"""
Wraps a subprocess call to the vendored TrackNetV3 (../tracknet_v3/predict.py)
for shuttlecock trajectory tracking in video.

Unlike shuttlecock_client.py (remote Roboflow API, per-frame independent
detection), this runs fully local and uses temporal/motion context across
the whole video -- background estimation + heatmap regression, then a
trajectory-rectification pass (InpaintNet) that fills in frames where the
shuttlecock was occluded/undetected. Generally far more accurate for
small/fast/blurry shuttlecocks than a per-frame detector, at the cost of
being video-only (needs motion context) and slower to run.

Subprocess (not in-process import) is deliberate: TrackNetV3 vendors its
own `model.py` / `dataset.py` / `utils` modules whose names would collide
with this project's own modules (and with easy_ViTPose's) if imported
into the same process.

`tracknet_v3/predict.py` was patched in two ways to run here (see that
file's own comments for grounding):
  1. CPU inference: the released checkpoints were saved from a CUDA run;
     `.cuda()` calls and `torch.load` were patched for CPU-only machines.
  2. `num_workers` forced to 0: multiprocessing DataLoader workers combined
     with PyTorch's OpenMP thread pool deadlock at interpreter shutdown on
     macOS (observed directly: main process hung in Py_FinalizeEx ->
     os.waitpid on an unreaped worker, ~15+ min, despite the actual
     prediction finishing in under a minute).
"""
import csv
import os
import subprocess
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKNET_DIR = os.path.join(ROOT, 'tracknet_v3')
TRACKNET_CKPT = os.path.join(ROOT, 'models', 'tracknet', 'TrackNet_best.pt')
INPAINT_CKPT = os.path.join(ROOT, 'models', 'tracknet', 'InpaintNet_best.pt')

DEFAULT_TIMEOUT = 1800.0  # seconds; CPU inference is slow, see module docstring


class TrackNetError(RuntimeError):
    """Raised when the TrackNetV3 subprocess fails or its checkpoints are missing."""


def is_available() -> bool:
    """Whether the vendored TrackNetV3 code and checkpoints are present."""
    return (
        os.path.isfile(os.path.join(TRACKNET_DIR, 'predict.py'))
        and os.path.isfile(TRACKNET_CKPT)
        and os.path.isfile(INPAINT_CKPT)
    )


def run_tracknet_on_video(
    video_path: str,
    save_dir: str,
    output_video: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Run TrackNetV3 (trajectory prediction + inpainting rectification) on a
    video file.

    Returns:
        {
          'csv_path': str,
          'video_path': str or None (annotated video, if output_video=True),
          'trajectory': [{'frame': int, 'visible': bool, 'x': int, 'y': int}, ...],
        }
        `trajectory` has one entry per input frame, in order. `visible`
        False means no confident detection for that frame (x/y are 0).

    Raises:
        TrackNetError: checkpoints missing, subprocess failed, timed out,
            or its expected output CSV wasn't produced.
    """
    if not is_available():
        raise TrackNetError(
            f"Checkpoint TrackNetV3 belum ada di {os.path.dirname(TRACKNET_CKPT)}. "
            "Perlu TrackNet_best.pt & InpaintNet_best.pt."
        )

    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(TRACKNET_DIR, 'predict.py'),
        '--video_file', os.path.abspath(video_path),
        '--tracknet_file', TRACKNET_CKPT,
        '--inpaintnet_file', INPAINT_CKPT,
        '--save_dir', os.path.abspath(save_dir),
    ]
    if output_video:
        cmd.append('--output_video')

    try:
        result = subprocess.run(
            cmd, cwd=TRACKNET_DIR, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise TrackNetError(f"TrackNetV3 timeout setelah {timeout:.0f} detik.") from e

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '')[-2000:]
        raise TrackNetError(f"TrackNetV3 gagal (exit {result.returncode}): {tail}")

    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(save_dir, f'{base}_ball.csv')
    out_video_path = os.path.join(save_dir, f'{base}.mp4')

    if not os.path.isfile(csv_path):
        raise TrackNetError(f"TrackNetV3 selesai tapi CSV hasil ({csv_path}) tidak ditemukan.")

    trajectory: List[Dict[str, Any]] = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            trajectory.append({
                'frame': int(row['Frame']),
                'visible': bool(int(row['Visibility'])),
                'x': int(row['X']),
                'y': int(row['Y']),
            })

    return {
        'csv_path': csv_path,
        'video_path': out_video_path if os.path.isfile(out_video_path) else None,
        'trajectory': trajectory,
    }
