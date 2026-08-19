"""
Client for golf club (grip + head) detection via Roboflow's hosted
single-model inference API (`https://serverless.roboflow.com/{project}/{version}`),
combined with the ViTPose **body** keypoints as a plausibility anchor.

Model history (why this isn't the first attempt)
--------------------------------------------------
v1: golfswingtrial/sample-skeleton (~68 training images). Got "stuck" on
backswing/top frames, confidently returning a static, visually-wrong point.
v2: myspace-riqj6/golf-pth94, a multi-class detector (player/golf club/club
head/golf ball). Better, but still a small/inconsistent community project --
occasionally matched the golf bag sitting elsewhere in frame instead of the
club actually being swung.
v3 (current): club-head-tracking/golf-club-tracking (v1) -- 6,750 source
images (11,479 with augmentation), real train/valid/test splits, and
published validation metrics: precision 90.9%, recall 84.6%, mAP50 89.9%.
Its classes are labeled just "0", "1", "3" (no human-readable names were set
by the uploader) -- confirmed empirically by plotting all three classes on
several frames and comparing against the visible club in the source image:
  - class "3" sits consistently 15-30px from the wrist -> grip/handle.
  - class "1" lands exactly on the visible club head/face (confirmed at
    address, where it sits right next to the ball, and at the top of the
    backswing, where it's the small dark shape at the end of the shaft).
  - class "0" is NOT the head -- it consistently lands mid-shaft, well short
    of the actual head. (First shipped as "head" by mistake, since it fires
    more often/confidently than class "1"; corrected after a screenshot
    showed it sitting mid-shaft instead of on the clubhead.) Left unused.

Why body-anchored, not raw detector output
--------------------------------------------
Even a good detector occasionally fires on the wrong thing (e.g. a face,
a club sitting in a bag elsewhere in frame). ViTPose's wrist tracking is
consistently accurate on the same footage, so it's used as a sanity anchor
rather than trusting the club detector blind:

  - grip: the class-"3" box closest to the wrist, if within a plausible
    distance; otherwise the wrist position itself is used directly
    (`source: "vitpose_wrist"`).
  - head: the highest-confidence class-"1" box that is (a) far enough from
    the wrist to plausibly be the *other end* of the club and (b) not
    implausibly far (e.g. a club sitting in a bag elsewhere in frame).
    Outside that distance ring, or with no candidates at all, head is
    reported as None rather than a guessed point -- class "1" fires less
    often than the mid-shaft class "0" did, so expect more honest "not
    detected" frames in exchange for the point being correct when present.

Distance thresholds are fractions of a body-scale estimate (vertical span of
confident body keypoints), so they hold up across zoom levels/frame sizes.
"""
import base64
import os
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import requests

ROBOFLOW_INFERENCE_URL = "https://serverless.roboflow.com"
PROJECT_ID = os.environ.get("GOLF_ROBOFLOW_PROJECT", "golf-club-tracking")
PROJECT_VERSION = os.environ.get("GOLF_ROBOFLOW_VERSION", "1")
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

# Detector classes (numeric labels as set by the dataset uploader -- see module
# docstring for how "0"/"3" were confirmed to mean club head / grip).
CLUB_HEAD_CLASSES = ("1",)
CLUB_GRIP_CLASSES = ("3",)
# Sits mid-shaft, short of the true head (see module docstring) -- last-resort
# stand-in only, used when the head class never fires at all for a frame (this
# detector's real-world recall on class "1" is ~85%, so some frames genuinely
# have no head box no matter how the distance ring is tuned).
CLUB_MIDSHAFT_CLASSES = ("0",)

# COCO keypoint indices (must match easy_ViTPose's "coco" dataset order).
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
MIN_KEYPOINT_SCORE = 0.3

# Distance thresholds, as a fraction of body scale (vertical span of confident keypoints).
GRIP_MAX_DIST_RATIO = 0.35   # grip detection must land within this fraction of body scale from the wrist
HEAD_MIN_DIST_RATIO = 0.12   # closer than this to the wrist -> probably the hands/grip itself, not the head
HEAD_MAX_DIST_RATIO = 1.15   # farther than this -> probably a different object (e.g. bag of clubs elsewhere)
# Second-pass ceiling, used only when the strict ring above finds nothing. Fast
# mid-swing frames (toe-up, mid-backswing, mid-downswing, mid-follow-through) blur
# the club and stretch the arm+shaft further from the wrist in image-space than the
# static address/top/impact/finish poses HEAD_MAX_DIST_RATIO was tuned against --
# those are exactly the frames golf_result_long.csv showed missing a head point.
HEAD_MAX_DIST_RATIO_RELAXED = 1.8

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2
DEFAULT_CONFIDENCE_PCT = 5  # keep low; the wrist-distance filter (not the API) does the real filtering


class GolfClubDetectionError(RuntimeError):
    """Raised when the Roboflow golf club detection model can't be reached or fails."""


class GolfClubConfigError(GolfClubDetectionError):
    """Raised when ROBOFLOW_API_KEY is missing or invalid."""


def _load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def is_api_key_configured() -> bool:
    _load_env_file()
    return bool(os.environ.get(API_KEY_ENV_VAR))


def _get_api_key() -> str:
    _load_env_file()
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise GolfClubConfigError(
            f"{API_KEY_ENV_VAR} belum di-set. Ambil Private API Key di "
            "app.roboflow.com/settings/api, lalu salah satu:\n"
            f"  1) copy webapp/.env.example jadi webapp/.env dan isi key-nya di situ, atau\n"
            f"  2) export {API_KEY_ENV_VAR}='rf_xxxxxxxxxxxx' sebelum start server"
        )
    return api_key


def _encode_image_to_b64(image: Union[str, "np.ndarray"]) -> str:
    if isinstance(image, np.ndarray):
        # VideoReader / cv2 frames are RGB in this codebase; the Roboflow API expects
        # standard encoded bytes, so make sure we hand it BGR before imencode.
        bgr = image[..., ::-1] if image.shape[-1] == 3 else image
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            raise GolfClubDetectionError("Gagal encode frame (numpy array) jadi JPEG.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    if isinstance(image, str) and os.path.isfile(image):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    raise GolfClubDetectionError(
        f"Tipe/isi input gambar tidak dikenali: {image!r} (harus path file lokal yang ada atau numpy array RGB)."
    )


def run_golf_club_detection(
    image: Union[str, "np.ndarray"],
    confidence: float = DEFAULT_CONFIDENCE_PCT,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> List[Dict[str, Any]]:
    """
    Run the golf object-detection model (player / golf club / club head / golf
    ball) on a single image.

    `image` may be a local file path or a numpy array (RGB, e.g. a frame from
    `easy_ViTPose.vit_utils.inference.VideoReader`).

    Returns the raw list of Roboflow predictions (each a dict with
    x/y/width/height/confidence/class for the box). Empty list if nothing
    detected.
    """
    api_key = _get_api_key()
    b64 = _encode_image_to_b64(image)
    url = f"{ROBOFLOW_INFERENCE_URL}/{PROJECT_ID}/{PROJECT_VERSION}"

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                params={"api_key": api_key, "confidence": confidence},
                data=b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("predictions", [])
        except (requests.RequestException, ValueError) as e:
            last_error = e
        if attempt < retries:
            time.sleep(2 ** attempt)

    raise GolfClubDetectionError(
        f"Gagal memanggil model deteksi golf club setelah {retries + 1} percobaan: {last_error}"
    ) from last_error


def _body_anchor(body_keypoints: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    From one person's ViTPose keypoints (Nx3 array of (y, x, score), COCO
    order), derive: the wrist to anchor the club against (whichever of
    left/right has the higher score) and a body-scale estimate (vertical
    span of confident keypoints) to turn the distance thresholds above into
    actual pixels. Returns None if there isn't enough confident signal.
    """
    if body_keypoints is None or len(body_keypoints) <= max(KP_LEFT_WRIST, KP_RIGHT_WRIST):
        return None

    lw, rw = body_keypoints[KP_LEFT_WRIST], body_keypoints[KP_RIGHT_WRIST]
    wrist = lw if lw[2] >= rw[2] else rw
    if wrist[2] < MIN_KEYPOINT_SCORE:
        return None

    confident_ys = [p[0] for p in body_keypoints if p[2] >= MIN_KEYPOINT_SCORE]
    if len(confident_ys) < 3:
        return None
    scale = max(confident_ys) - min(confident_ys)
    if scale < 1:
        return None

    return {"wrist_x": float(wrist[1]), "wrist_y": float(wrist[0]), "scale": float(scale)}


def select_grip_head(
    predictions: List[Dict[str, Any]],
    body_keypoints: Optional[np.ndarray],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Pick the grip and head points from the detector's candidate boxes,
    validated against the wrist position from ViTPose body keypoints.

    Returns {"grip": {...} or None, "head": {...} or None}, each point a
    dict with x, y, confidence, and source ("detector" or "vitpose_wrist"
    for the grip fallback).
    """
    head_boxes_seen = sum(1 for p in predictions if p.get("class") in CLUB_HEAD_CLASSES)

    anchor = _body_anchor(body_keypoints)
    if anchor is None:
        # No reliable body signal to validate against -- fall back to the single
        # highest-confidence box for head, and leave grip undetermined.
        best = max(predictions, key=lambda p: p.get("confidence", 0.0)) if predictions else None
        head = ({"x": float(best["x"]), "y": float(best["y"]), "confidence": float(best["confidence"]),
                 "source": "detector"} if best else None)
        return {"grip": None, "head": head, "head_candidates_seen": head_boxes_seen}

    wx, wy, scale = anchor["wrist_x"], anchor["wrist_y"], anchor["scale"]

    def dist(p):
        return ((p["x"] - wx) ** 2 + (p["y"] - wy) ** 2) ** 0.5

    def box_id(p):
        # detection_id is unique per box when present; (x, y, class) is a safe fallback.
        return p.get("detection_id", (p.get("x"), p.get("y"), p.get("class")))

    grip_candidates = sorted(
        (p for p in predictions if p.get("class") in CLUB_GRIP_CLASSES and dist(p) <= GRIP_MAX_DIST_RATIO * scale),
        key=dist,
    )
    grip = None
    grip_box_id = None
    if grip_candidates:
        p = grip_candidates[0]
        grip_box_id = box_id(p)
        grip = {"x": float(p["x"]), "y": float(p["y"]), "confidence": float(p["confidence"]), "source": "detector"}
    else:
        grip = {"x": wx, "y": wy, "confidence": float(body_keypoints[KP_LEFT_WRIST][2]
                                                       if body_keypoints[KP_LEFT_WRIST][2] >= body_keypoints[KP_RIGHT_WRIST][2]
                                                       else body_keypoints[KP_RIGHT_WRIST][2]),
                "source": "vitpose_wrist"}

    # Exclude whichever box was already claimed as the grip -- the two ends of the
    # club can't be the same box (this matters most in the overlap zone between
    # GRIP_MAX_DIST_RATIO and HEAD_MIN_DIST_RATIO, where a box near the hands is a
    # valid *grip* candidate but not a valid *head* candidate).
    head_candidates = [
        p for p in predictions
        if p.get("class") in CLUB_HEAD_CLASSES
        and HEAD_MIN_DIST_RATIO * scale <= dist(p) <= HEAD_MAX_DIST_RATIO * scale
        and box_id(p) != grip_box_id
    ]
    head_source = "detector"
    if not head_candidates:
        # Strict ring came up empty -- retry with the wider ceiling before giving up
        # (see HEAD_MAX_DIST_RATIO_RELAXED). Still excludes the grip box and the
        # too-close-to-hands zone, just tolerates the club reaching further out.
        head_candidates = [
            p for p in predictions
            if p.get("class") in CLUB_HEAD_CLASSES
            and HEAD_MIN_DIST_RATIO * scale <= dist(p) <= HEAD_MAX_DIST_RATIO_RELAXED * scale
            and box_id(p) != grip_box_id
        ]
        head_source = "detector_relaxed"

    if not head_candidates and head_boxes_seen == 0:
        # Class "1" never fired at all for this frame (not a distance-ring filtering
        # issue -- there's simply no head box to filter). Last resort: the mid-shaft
        # class "0" box, which sits short of the true head but is still on the club,
        # so it's a rough-but-useful stand-in rather than leaving head blank.
        head_candidates = [
            p for p in predictions
            if p.get("class") in CLUB_MIDSHAFT_CLASSES
            and HEAD_MIN_DIST_RATIO * scale <= dist(p) <= HEAD_MAX_DIST_RATIO_RELAXED * scale
            and box_id(p) != grip_box_id
        ]
        head_source = "detector_midshaft_approx"

    head = None
    if head_candidates:
        p = max(head_candidates, key=lambda p: p.get("confidence", 0.0))
        head = {"x": float(p["x"]), "y": float(p["y"]), "confidence": float(p["confidence"]), "source": head_source}

    return {"grip": grip, "head": head, "head_candidates_seen": head_boxes_seen}


def draw_club_keypoints(
    img_bgr: np.ndarray,
    club_points: Optional[Dict[str, Optional[Dict[str, Any]]]],
    grip_color=(0, 215, 255),   # BGR amber
    head_color=(60, 60, 255),   # BGR red
    line_color=(0, 255, 0),     # BGR green, matches the reference "object detection" panel
) -> np.ndarray:
    """Draw the grip-to-head club line + labeled points onto a BGR image in place.

    `club_points` is the dict returned by `select_grip_head` (or None).
    """
    if not club_points:
        return img_bgr
    grip, head = club_points.get("grip"), club_points.get("head")

    if grip and head:
        cv2.line(img_bgr, (int(grip["x"]), int(grip["y"])), (int(head["x"]), int(head["y"])), line_color, 3)
    for point, label, color in ((grip, "grip", grip_color), (head, "head", head_color)):
        if point is None:
            continue
        cx, cy = int(point["x"]), int(point["y"])
        cv2.circle(img_bgr, (cx, cy), 6, color, -1)
        suffix = {
            "vitpose_wrist": " (wrist)",
            "detector_relaxed": " (relaxed)",
            "detector_midshaft_approx": " (approx)",
            "interpolated_same_swing": " (interpolated)",
            "manual": " (manual)",
        }.get(point.get("source"), "")
        confidence = point.get("confidence")
        conf_label = f'{confidence:.2f}' if confidence is not None else ''
        cv2.putText(img_bgr, f'{label}{suffix} {conf_label}'.strip(), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img_bgr
