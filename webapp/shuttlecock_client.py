"""
Client for the Roboflow "Shuttlecock Detection" workflow.

Deliberately implemented with plain `requests` instead of the official
`inference-sdk` package: `inference-sdk` pulls in `supervision` -> a recent
`scipy` that requires numpy>=2, which is incompatible with the numpy<2 this
project's torch/ViTPose stack needs (they share one venv). The wire protocol
below (endpoint shape, request/response JSON) was reverse-grounded from
`inference-sdk`'s own source (`http/client.py::_execute_workflow_request`,
`http/utils/loaders.py`, `http/utils/post_processing.py`) while it was
briefly installed in this project's venv, so it should be an exact match for
what the SDK itself sends/expects — not a guess.

Output shape — confirmed via a real run (smoke_test_shuttlecock.py) against
this workflow on 2026-08-05:

    {
      "output_image": "<base64 JPEG, already annotated by the workflow>",
      "predictions": {
        "image": {"width": ..., "height": ...},
        "predictions": [
          {"x", "y", "width", "height", "confidence", "class_id",
           "class": "Shuttle", "detection_id", "parent_id"},
          ...
        ]
      },
      "shuttlecock_count": <int>,
      "vision_events_status": "<str>"
    }

`split_workflow_outputs` still classifies fields by VALUE SHAPE rather than
these exact key names (image-looking base64? Roboflow-style
{"predictions": [...]}-shaped dict? plain list of dicts?) so it keeps working
if the workflow definition changes later — but the shape above is real, not
a guess.
"""
import base64
import binascii
import os
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import requests

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
WORKSPACE_NAME = "dewa-ahmad-septriansyah"
WORKFLOW_ID = "shuttlecock-detection-1785945414105"
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2

_IMAGE_MAGIC_BYTES = (
    b"\xff\xd8\xff",        # JPEG
    b"\x89PNG\r\n\x1a\n",   # PNG
    b"GIF87a",              # GIF
    b"GIF89a",
    b"RIFF",                # WEBP (RIFF....WEBP)
)


class ShuttlecockDetectionError(RuntimeError):
    """Raised when the Roboflow shuttlecock detection workflow can't be reached or fails."""


class ShuttlecockConfigError(ShuttlecockDetectionError):
    """Raised when ROBOFLOW_API_KEY is missing or invalid."""


def _load_env_file() -> None:
    """
    Populate os.environ from webapp/.env if it exists and the var isn't
    already set (a real env var / export always wins). Minimal hand-rolled
    parser (KEY=VALUE per line, '#' comments) so this doesn't need a new
    dependency like python-dotenv just for one optional file.
    """
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
    """Check whether ROBOFLOW_API_KEY is available (real env var or webapp/.env)."""
    _load_env_file()
    return bool(os.environ.get(API_KEY_ENV_VAR))


def _get_api_key() -> str:
    _load_env_file()
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise ShuttlecockConfigError(
            f"{API_KEY_ENV_VAR} belum di-set. Ambil Private API Key di "
            "app.roboflow.com/settings/api, lalu salah satu:\n"
            f"  1) copy webapp/.env.example jadi webapp/.env dan isi key-nya di situ, atau\n"
            f"  2) export {API_KEY_ENV_VAR}='rf_xxxxxxxxxxxx' sebelum start server"
        )
    return api_key


def _encode_image_input(image: Union[str, "np.ndarray"]) -> Dict[str, str]:
    """Build the `{"type": ..., "value": ...}` input shape the workflow endpoint expects."""
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ShuttlecockDetectionError("Gagal encode frame (numpy array) jadi JPEG.")
        return {"type": "base64", "value": base64.b64encode(buf.tobytes()).decode("ascii")}

    if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        if image.startswith("http://"):
            raise ShuttlecockDetectionError("URL input harus https://, http:// ditolak server.")
        return {"type": "url", "value": image}

    if isinstance(image, str) and os.path.isfile(image):
        with open(image, "rb") as f:
            raw_bytes = f.read()
        return {"type": "base64", "value": base64.b64encode(raw_bytes).decode("ascii")}

    raise ShuttlecockDetectionError(
        f"Tipe/isi input gambar tidak dikenali: {image!r} "
        "(harus path file lokal yang ada, URL https://, atau numpy array)."
    )


def run_shuttlecock_workflow(
    image: Union[str, "np.ndarray"],
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """
    Run the "Shuttlecock Detection" Roboflow workflow on a single image.

    `image` may be a local file path, an https:// image URL, or a numpy
    array (e.g. a video frame as read by cv2/VideoReader).

    Returns the raw per-image output dict for the (single) input image,
    with whatever keys the workflow itself defines. Retries with
    exponential backoff (1s, 2s, ...) on transient failures/timeouts.

    Raises:
        ShuttlecockConfigError: ROBOFLOW_API_KEY is not set.
        ShuttlecockDetectionError: all attempts failed (network, timeout,
            malformed response, or Roboflow-side error).
    """
    api_key = _get_api_key()
    image_input = _encode_image_input(image)
    url = f"{ROBOFLOW_API_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key": api_key,
        "use_cache": True,
        "inputs": {"image": image_input},
    }

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            outputs = data.get("outputs")
            if not outputs:
                raise ShuttlecockDetectionError(
                    f"Workflow mengembalikan hasil kosong (response keys: {list(data.keys())})."
                )
            return outputs[0]
        except (requests.RequestException, ValueError, ShuttlecockDetectionError) as e:
            last_error = e

        if attempt < retries:
            time.sleep(2 ** attempt)

    raise ShuttlecockDetectionError(
        f"Gagal memanggil workflow Shuttlecock Detection setelah {retries + 1} percobaan: {last_error}"
    ) from last_error


def _decode_if_base64_image(value: Any) -> Optional[bytes]:
    """Return decoded image bytes if `value` looks like a base64-encoded image, else None."""
    candidate = value
    if isinstance(value, dict) and value.get("type") == "base64":
        candidate = value.get("value")
    if not isinstance(candidate, str) or len(candidate) < 100:
        return None
    try:
        decoded = base64.b64decode(candidate, validate=False)
    except (binascii.Error, ValueError):
        return None
    if any(decoded.startswith(magic) for magic in _IMAGE_MAGIC_BYTES):
        return decoded
    return None


def _extract_detection_list(value: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Recognize Roboflow's standard object-detection output shape:
    `{"image": {"width": .., "height": ..}, "predictions": [ {...}, ... ]}`
    (confirmed via a real smoke_test_shuttlecock.py run against this
    workflow: its "predictions" output looks exactly like this, with each
    prediction dict having x/y/width/height/confidence/class/class_id).
    Returns the inner list if `value` matches this shape, else None.
    """
    if isinstance(value, dict) and isinstance(value.get("predictions"), list):
        preds = value["predictions"]
        if not preds or isinstance(preds[0], dict):
            return preds
    return None


def split_workflow_outputs(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Best-effort split of a raw workflow output dict, classified by VALUE
    SHAPE rather than known key names (see module docstring for why):

      - images: decoded bytes for values that are base64-encoded images
      - detections: values that are (or contain, Roboflow-style) a list of
        prediction dicts
      - other: everything else (counts, labels, scalars, ...)
    """
    images: Dict[str, bytes] = {}
    detections: Dict[str, List[Dict[str, Any]]] = {}
    other: Dict[str, Any] = {}

    for key, value in raw.items():
        decoded = _decode_if_base64_image(value)
        if decoded is not None:
            images[key] = decoded
            continue

        detection_list = _extract_detection_list(value)
        if detection_list is not None:
            detections[key] = detection_list
            continue

        if isinstance(value, list) and value and isinstance(value[0], dict):
            detections[key] = value
            continue

        other[key] = value

    return {"images": images, "detections": detections, "other": other}


def save_result_images(images: Dict[str, bytes], out_dir: str, run_id: str) -> Dict[str, str]:
    """
    Write decoded image outputs to `out_dir`. Returns {output_key: filename}.
    Bytes are not held onto after being written.
    """
    saved: Dict[str, str] = {}
    os.makedirs(out_dir, exist_ok=True)
    for key, raw_bytes in images.items():
        filename = f"{run_id}_{key}.jpg"
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(raw_bytes)
        saved[key] = filename
    return saved
