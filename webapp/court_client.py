"""
Client for the Roboflow public "badminton-court-keypoint-dataset" model
(workspace `learning-9i34b`, version 5) — detects the badminton court and
30 named line-intersection keypoints (grouped into 4 rows front-to-back:
`t`=far baseline, `u`=far service line, `l`=near service line, `b`=near
baseline; each row has `l`/`m`/`r` sub-points left/mid/right within that
line, with numeric suffixes whose exact meaning isn't documented by the
model author).

Confirmed via a real call on 2026-08-07: POST multipart file to
    https://serverless.roboflow.com/badminton-court-keypoint-dataset/5?api_key=...
returns `{"predictions": [{"class": "courtline", "keypoints": [...]}]}`,
each keypoint `{"x", "y", "confidence", "class_id", "class"}` in pixel
coords of the (internally resized) 640x640 input.
"""
import os
import socket
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import requests

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_HOST = "serverless.roboflow.com"
COURT_MODEL_ID = "badminton-court-keypoint-dataset/5"
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2

# This environment's local DNS resolver is intermittently unreliable for
# this specific domain (SERVFAIL even while `dig`/`curl` succeed via a
# public resolver moments later -- observed repeatedly, not a one-off).
# Cloudflare-fronted IPs confirmed working via `dig @8.8.8.8` on 2026-08-07;
# used as a last-resort fallback, the same trick as `curl --resolve`: force
# the TCP connection to a known-good IP while keeping the real hostname for
# TLS SNI / the Host header, so certificate validation still succeeds.
_DNS_FALLBACK_IPS = ["172.66.166.205", "104.20.41.123"]
_real_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_with_fallback(host, *args, **kwargs):
    try:
        return _real_getaddrinfo(host, *args, **kwargs)
    except socket.gaierror:
        if host != ROBOFLOW_API_HOST:
            raise
        last_error = None
        for ip in _DNS_FALLBACK_IPS:
            try:
                return _real_getaddrinfo(ip, *args, **kwargs)
            except socket.gaierror as e:
                last_error = e
        raise last_error


socket.getaddrinfo = _getaddrinfo_with_fallback


class CourtDetectionError(RuntimeError):
    """Raised when the Roboflow court-keypoint model can't be reached or fails."""


class CourtConfigError(CourtDetectionError):
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
        raise CourtConfigError(
            f"{API_KEY_ENV_VAR} belum di-set. Isi webapp/.env atau export "
            f"{API_KEY_ENV_VAR} sebelum pakai deteksi lapangan."
        )
    return api_key


def _image_to_jpeg_bytes(image: Union[str, "np.ndarray"]) -> bytes:
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise CourtDetectionError("Gagal encode frame (numpy array) jadi JPEG.")
        return buf.tobytes()

    if isinstance(image, str) and os.path.isfile(image):
        with open(image, "rb") as f:
            return f.read()

    raise CourtDetectionError(
        f"Tipe/isi input gambar tidak dikenali: {image!r} "
        "(harus path file lokal yang ada, atau numpy array)."
    )


def run_court_detection(
    image: Union[str, "np.ndarray"],
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """
    Run the court-keypoint model on a single image.

    Returns the raw response dict: `{"predictions": [...], "image": {...}, ...}`.
    Raises CourtConfigError / CourtDetectionError on failure.
    """
    api_key = _get_api_key()
    jpeg_bytes = _image_to_jpeg_bytes(image)
    url = f"{ROBOFLOW_API_URL}/{COURT_MODEL_ID}"

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                params={"api_key": api_key},
                files={"file": ("image.jpg", jpeg_bytes, "image/jpeg")},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
        if attempt < retries:
            import time
            time.sleep(2 ** attempt)

    raise CourtDetectionError(
        f"Gagal memanggil model deteksi lapangan setelah {retries + 1} percobaan: {last_error}"
    ) from last_error


def best_court_detection(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the highest-confidence court detection (with its keypoints), or None."""
    preds = raw.get("predictions") or []
    if not preds:
        return None
    return max(preds, key=lambda p: p.get("confidence", 0))


# Real-world badminton court reference (BWF standard, doubles), in meters.
# X: 0..6.1 (width, doubles sideline to doubles sideline)
# Y: 0..13.4 (length, far baseline to near baseline), net at Y=6.7
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = COURT_LENGTH_M / 2
SHORT_SERVICE_LINE_M = 1.98  # from net

_ROW_Y_METERS = {
    "t": 0.0,                                  # far baseline
    "u": NET_Y_M - SHORT_SERVICE_LINE_M,       # far short service line
    "l": NET_Y_M + SHORT_SERVICE_LINE_M,       # near short service line
    "b": COURT_LENGTH_M,                       # near baseline
}


def build_homography(
    keypoints: List[Dict[str, Any]], min_confidence: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Build a pixel->court(meters) homography using RANSAC over up to 4 of
    the model's line-rows (far baseline 't', far service line 'u', near
    service line 'l', near baseline 'b' -- this front/back Y grouping
    matches the visible line positions). Within each confident row, up
    to 3 name-agnostic correspondences are contributed: the min-x pixel
    -> X=0 (left doubles sideline), the max-x pixel -> X=COURT_WIDTH_M
    (right doubles sideline), and -- if any of that row's points has 'm'
    in its label -- their average -> X=COURT_WIDTH_M/2 (center line).
    The model's numeric sub-index semantics are NOT relied on beyond
    that (could not be reliably verified -- see module docstring).

    Using every available row (not just 2) and fitting with RANSAC
    instead of an exact 4-point solve matters in practice: a single
    mislabeled/outlier keypoint (observed to happen -- see module
    docstring's grounding notes) previously corrupted the whole
    transform when only 4 points were used with zero tolerance. With
    >=3 rows' worth of points, RANSAC can down-weight/reject a bad
    corner instead of being wrecked by it.

    Returns a 3x3 homography matrix (pixel -> meters), or None if fewer
    than 4 correspondences are available (RANSAC's minimum).
    """
    rows: Dict[str, List[Dict[str, Any]]] = {"t": [], "u": [], "l": [], "b": []}
    for kp in keypoints:
        name = kp.get("class", "")
        if not name or name[0] not in rows:
            continue
        if kp.get("confidence", 0) < min_confidence:
            continue
        rows[name[0]].append(kp)

    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    for row_key, pts in rows.items():
        if len(pts) < 2:
            continue
        y_m = _ROW_Y_METERS[row_key]
        leftmost = min(pts, key=lambda p: p["x"])
        rightmost = max(pts, key=lambda p: p["x"])
        src_pts.append([leftmost["x"], leftmost["y"]])
        dst_pts.append([0.0, y_m])
        src_pts.append([rightmost["x"], rightmost["y"]])
        dst_pts.append([COURT_WIDTH_M, y_m])

        mid_pts = [p for p in pts if "m" in p.get("class", "")]
        if mid_pts:
            mx = sum(p["x"] for p in mid_pts) / len(mid_pts)
            my = sum(p["y"] for p in mid_pts) / len(mid_pts)
            src_pts.append([mx, my])
            dst_pts.append([COURT_WIDTH_M / 2, y_m])

    if len(src_pts) < 4:
        return None

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=0.5)
    return homography


def build_homography_from_corners(corners_px: List[List[float]]) -> np.ndarray:
    """
    Build an exact pixel->court(meters) homography from 4 manually-picked
    pixel points, in order [far-left, far-right, near-right, near-left]
    of the doubles court's outer boundary. Unlike the auto-detected
    version, this has no model-precision error to average out -- it's
    only as accurate as the 4 clicks -- so an exact 4-point solve
    (no RANSAC) is appropriate.
    """
    src = np.array(corners_px, dtype=np.float32)
    dst = np.array([
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def project_points(homography: np.ndarray, points_xy: List[List[float]]) -> List[List[float]]:
    """Project a list of [x, y] pixel points to [x, y] court-meter points."""
    if not points_xy:
        return []
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, homography)
    return projected.reshape(-1, 2).tolist()


_SINGLES_INSET_M = 0.46  # doubles sideline -> singles sideline
_DOUBLES_SERVICE_LINE_M = 0.76  # from each baseline
_LINE_COLOR = (255, 255, 255)
_COURT_COLOR = (60, 130, 20)  # BGR, badminton-court green
_PLAYER_COLORS = [(60, 60, 230), (230, 130, 40)]  # BGR: red-ish, blue-ish
_SHUTTLE_COLOR = (0, 230, 255)  # BGR yellow


def draw_minimap(
    player_points_m: Optional[List[List[float]]] = None,
    shuttlecock_point_m: Optional[List[float]] = None,
    player_labels: Optional[List[str]] = None,
    scale: int = 50,
    margin: int = 100,
) -> np.ndarray:
    """
    Render a top-down badminton court diagram (BGR image) with optional
    player dots and a shuttlecock dot, given their positions in court
    meters (x in 0..COURT_WIDTH_M, y in 0..COURT_LENGTH_M).
    Points outside the court are still drawn (clamped visually by the
    canvas, not hidden) so an off-court reading is visible, not silently
    dropped.
    """
    w = int(COURT_WIDTH_M * scale) + margin * 2
    h = int(COURT_LENGTH_M * scale) + margin * 2
    img = np.full((h, w, 3), 40, dtype=np.uint8)

    def to_px(x_m: float, y_m: float) -> tuple:
        return (int(margin + x_m * scale), int(margin + y_m * scale))

    tl = to_px(0, 0)
    br = to_px(COURT_WIDTH_M, COURT_LENGTH_M)
    cv2.rectangle(img, tl, br, _COURT_COLOR, -1)

    def hline(y_m: float):
        cv2.line(img, to_px(0, y_m), to_px(COURT_WIDTH_M, y_m), _LINE_COLOR, 2)

    def vline(x_m: float, y0_m: float, y1_m: float):
        cv2.line(img, to_px(x_m, y0_m), to_px(x_m, y1_m), _LINE_COLOR, 2)

    # Outer boundary (doubles)
    cv2.rectangle(img, tl, br, _LINE_COLOR, 2)
    # Singles sidelines
    vline(_SINGLES_INSET_M, 0, COURT_LENGTH_M)
    vline(COURT_WIDTH_M - _SINGLES_INSET_M, 0, COURT_LENGTH_M)
    # Net
    hline(NET_Y_M)
    # Short service lines
    hline(NET_Y_M - SHORT_SERVICE_LINE_M)
    hline(NET_Y_M + SHORT_SERVICE_LINE_M)
    # Doubles long service lines
    hline(_DOUBLES_SERVICE_LINE_M)
    hline(COURT_LENGTH_M - _DOUBLES_SERVICE_LINE_M)
    # Center line (only within each service court, short-service-line to baseline)
    cx = COURT_WIDTH_M / 2
    vline(cx, 0, NET_Y_M - SHORT_SERVICE_LINE_M)
    vline(cx, NET_Y_M + SHORT_SERVICE_LINE_M, COURT_LENGTH_M)

    for i, p in enumerate(player_points_m or []):
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
        center = to_px(p[0], p[1])
        label = player_labels[i] if player_labels and i < len(player_labels) else f"P{i + 1}"
        cv2.circle(img, center, 8, color, -1)
        cv2.circle(img, center, 8, (255, 255, 255), 1)
        cv2.putText(img, label, (center[0] + 10, center[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if shuttlecock_point_m is not None:
        center = to_px(shuttlecock_point_m[0], shuttlecock_point_m[1])
        cv2.drawMarker(img, center, _SHUTTLE_COLOR, cv2.MARKER_STAR, 14, 2)

    return img
