# GolfVitPose

Pose-estimation toolkit built on top of [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) (ViTPose + YOLOv8),
extended with a Flask web app and a batch pipeline for golf-swing analysis (GolfDB) and badminton
analysis (shuttlecock detection, court position tracking, TrackNetV3 trajectory tracking).

## What's in this repo

- **`easy_ViTPose/`** — the base pose-estimation library (ViTPose models + YOLOv8 person/animal
  detector). Human and animal 2D pose estimation, image/video/webcam, ONNX/Torch/TensorRT inference.
- **`webapp/`** — a Flask app with two parts:
  - an **upload page** (`/`) for one-off image/video processing across 6 modes (pose, shuttlecock,
    combined, court tracking, TrackNetV3, golf body+club);
  - a **GolfDB batch page** (`/golfdb`) for processing the [GolfDB](https://github.com/wmcnally/GolfDB)
    dataset in bulk — one swing, several, or "everything not done yet" — with per-frame progress, a
    click-to-edit UI for correcting club keypoints, and combined CSV exports for downstream analysis
    (clustering, ML).
- **`golf_inference.py`** — the standalone CLI the batch page runs under the hood (body pose + golf
  club grip/head detection, labeled against GolfDB's swing events).
- **`tracknet_v3/`** — vendored [TrackNetV3](https://github.com/qaz812345/TrackNetV3) (shuttlecock
  trajectory tracking: heatmap CNN + trajectory-rectification network), used locally (no external API)
  by the webapp's TrackNetV3 mode.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # add torch yourself first, see requirements_gpu.txt for CUDA/MPS
```

### Download models
Get the ViTPose + YOLOv8 checkpoints from [Huggingface](https://huggingface.co/JunkyByte/easy_ViTPose)
and place them under `models/` (e.g. `models/vitpose-l-coco.onnx`, `models/vitpose-l-ap10k.onnx`,
`models/yolov8l.pt`). For TrackNetV3, download `TrackNet_best.pt` + `InpaintNet_best.pt` and place them
under `models/tracknet/`.

### Roboflow API key (needed for Shuttlecock / Gabungan / Court / Golf modes)
Those modes call Roboflow-hosted detection models over HTTP. Local-only modes (plain pose, TrackNetV3)
don't need this.

```bash
cp webapp/.env.example webapp/.env
# edit webapp/.env and paste your Private API key (from app.roboflow.com/settings/api)
# after ROBOFLOW_API_KEY=
```

## Web UI

```bash
python webapp/app.py
# open http://127.0.0.1:5050
```

### Upload page (`/`) — 6 modes
| Mode | What it does | Runs locally / API |
|---|---|---|
| Manusia / Hewan | ViTPose body pose | local |
| Shuttlecock | Shuttlecock detection | Roboflow |
| Gabungan | Pose + shuttlecock combined | local + Roboflow |
| Posisi Lapangan | Player + shuttlecock position on a court minimap (homography) | local + Roboflow |
| TrackNetV3 | Shuttlecock trajectory across a whole video | local |
| Golf | Body pose + club grip/head keypoints | local + Roboflow |

Each result page shows the annotated image/video, a raw keypoint table, and downloadable JSON (video
modes also get a per-frame PNG download, single frame or all frames as a ZIP).

### GolfDB batch page (`/golfdb`)
Expects `golfDB.csv` (the dataset's metadata/events CSV) at the project root and each swing's extracted
event frames under `frames/frames/<0000-padded id>/` (10 frames per swing: GolfDB's `events` array —
clip_start, address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish,
clip_end).

- **Process** one swing, a selected set, or "N next pending" / "all pending" — resumable, since it
  always skips swings that already have output.
- **Live per-frame progress**, not just per-swing.
- **Per-swing detail page** (`/golfdb/<id>`) — one annotated image per event, a click-to-edit tool for
  manually placing/correcting the club grip/head point when detection missed it (redraws the image and
  updates the JSON/CSV immediately), "download all images as ZIP", and **reprocess** (wipes and redoes
  that swing from scratch).
- **Combined CSV exports** across every processed swing, generated on demand:
  - *per-frame* — one row per swing×event (same columns as one swing's own CSV, just stacked).
  - *wide* — one row per swing, every event's x/y features prefixed by event name
    (`address_nose_x`, `impact_club_head_y`, ...) — clustering-ready, numeric only.
  - *array* — one row per swing, each column holds all 10 events' values as an ordered list
    (`nose_x: [520.56, 521.08, ...]`).

  All three merge in GolfDB's own `events`/`bbox` columns and auto-fill missing club grip/head points:
  first by linear interpolation from the same swing's other detected events (grip only — the head
  moves through too fast/curved an arc for a straight-line estimate to be reliable, confirmed by
  rendering an example), then by falling back to the wrist keypoint. Every filled value is tagged in
  its `*_source` column (`interpolated_same_swing`, `wrist_fill_export`, ...) so it stays
  distinguishable from a real detection — manual correction via the detail page is the reliable fix
  for anything that still looks wrong.

### Output layout
```
hasil/<id>/
├── event-1/ ... event-10/     # annotated frame per GolfDB event (event-unmatched/ if unlabeled)
├── golf_result.json           # body keypoints + club points + GolfDB row, per frame
├── golf_result.csv            # same data, one row per frame, wide columns
└── golf_result_long.csv       # tidy format, one row per (frame, keypoint)
```

### Standalone CLI (what the batch page runs per swing)
```bash
python golf_inference.py --input frames/frames/0000 --golfdb golfDB.csv --golfdb-id 0
# or on a full video:
python golf_inference.py --input path/to/swing.mp4 --golfdb golfDB.csv --golfdb-id 0
```
`--output-path` defaults to `hasil/<input name>` if not given. `--no-club` skips Roboflow entirely
(body pose only — useful without an API key, or to process fast).

---

## Base library: ViTPose inference

### Skeleton reference
Multiple skeletons across datasets (AIC / MPII / COCO / COCO+FEET / COCO WHOLEBODY / APT36k / AP10k) —
see [`easy_ViTPose/vit_utils/visualization.py`](easy_ViTPose/vit_utils/visualization.py).

### CLI
```bash
python inference.py --input img.jpg --model models/vitpose-l-coco.onnx --yolo models/yolov8l.pt \
    --save-img --save-json
```
Run `python inference.py --help` for the full flag list (dataset, det-class, model-name, rotate,
single-pose, yolo-step, etc).

### From code
```python
import cv2
from easy_ViTPose import VitInference

img = cv2.cvtColor(cv2.imread('./examples/img1.jpg'), cv2.COLOR_BGR2RGB)
model = VitInference('models/vitpose-l-coco.onnx', 'models/yolov8l.pt', is_video=False)

keypoints = model.inference(img)  # {person_id: ndarray(N, 3) of (y, x, score)}
img = model.draw(show_yolo=True)
cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.waitKey(0)
```

### JSON output format
```json
{
  "keypoints": [
    { "0": [[121.19, 458.15, 0.99], "..."], "1": ["..."] }
  ],
  "skeleton": { "0": "nose", "1": "left_eye", "...": "..." }
}
```

## Finetuning
See the original project's guide for finetuning ViTPose on a custom dataset (checkpoint splitting via
`model_split.py`, COCO-format dataset prep, `train.py`, custom skeleton in `visualization.py`).

## Evaluation on COCO
```bash
python evaluation_on_coco.py --model_path ... --yolo_path ... --img_folder_path val2017/ \
    --annFile annotations/person_keypoints_val2017.json
```

## Docker
```bash
docker build . -t easy_vitpose
docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./models:/models -v ~/cats:/cats easy_vitpose \
    python inference.py --det-class cat --input /cats/image.jpg --output-path /cats \
    --save-img --model /models/vitpose-l-ap10k.onnx --yolo /models/yolov8l.pt
```

## Reference
Built on [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose) via
[JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose). Tracking: SORT
([abewley/sort](https://github.com/abewley/sort)). Shuttlecock trajectory:
[TrackNetV3](https://github.com/qaz812345/TrackNetV3). Golf swing dataset:
[GolfDB](https://github.com/wmcnally/GolfDB).
