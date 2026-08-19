"""
Golf swing inference: body pose (ViTPose, existing pipeline) + golf club
grip/head keypoints (Roboflow, webapp/golf_client.py), combined into one
annotated output + one JSON, following the reference pipeline:

    Input frames -> [Object Detection: club] + [Human Pose Estimation: body]
                 -> combined "Club + Body Keypoints" output

Works on:
  - a folder of frame images (e.g. sandra-dorset-01-noslow/, one PNG per
    GolfDB event frame) -- each frame is labeled with its GolfDB swing event
    name if a matching row in golfDB.csv can be found (by frame-index match).
  - a single video file -- every frame is processed and, if --golfdb-id is
    given, labeled against that row's event frame indices.

Usage:
    python golf_inference.py --input sandra-dorset-01-noslow \
        --golfdb golfDB.csv --output-path hasil/golf

    python golf_inference.py --input path/to/swing.mp4 \
        --golfdb golfDB.csv --golfdb-id 0 --output-path hasil/golf
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp"))

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
from easy_ViTPose.vit_utils.visualization import joints_dict

from golf_client import (  # noqa: E402
    GolfClubDetectionError,
    draw_club_keypoints,
    is_api_key_configured,
    run_golf_club_detection,
    select_grip_head,
)

# GolfDB's 10-value "events" array = [clip_start, 8 named events..., clip_end]
# (see the GolfDB paper / dataset: Address, Toe-up, Mid-backswing, Top,
# Mid-downswing, Impact, Mid-follow-through, Finish).
GOLFDB_EVENT_LABELS = [
    "clip_start", "address", "toe-up", "mid-backswing", "top",
    "mid-downswing", "impact", "mid-follow-through", "finish", "clip_end",
]

FRAME_INDEX_RE = re.compile(r"(\d+)")


def parse_int_array(cell: str):
    """Parse a numpy-repr-like CSV cell such as '[408 455 473 ... 545]' into a list of ints."""
    return [int(v) for v in re.findall(r"-?\d+", cell)]


def load_golfdb_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["events"] = parse_int_array(row["events"])
            row["bbox"] = [float(v) for v in re.findall(r"[\d.eE+-]+", row["bbox"])]
            rows.append(row)
    return rows


def find_matching_golfdb_row(rows, frame_indices):
    """Best-effort: the row whose event-frame set overlaps the given frame indices the most."""
    if not frame_indices:
        return None
    frame_set = set(frame_indices)
    best_row, best_overlap = None, 0
    for row in rows:
        overlap = len(frame_set & set(row["events"]))
        if overlap > best_overlap:
            best_row, best_overlap = row, overlap
    return best_row if best_overlap > 0 else None


def event_label_for_frame(golfdb_row, frame_idx):
    if golfdb_row is None or frame_idx not in golfdb_row["events"]:
        return None
    pos = golfdb_row["events"].index(frame_idx)
    if pos < len(GOLFDB_EVENT_LABELS):
        return GOLFDB_EVENT_LABELS[pos]
    return None


def event_no_for_frame(golfdb_row, frame_idx):
    """1-based position of frame_idx in golfdb_row['events'] (1=clip_start ... 10=clip_end), or None."""
    if golfdb_row is None or frame_idx not in golfdb_row["events"]:
        return None
    return golfdb_row["events"].index(frame_idx) + 1


def event_folder_name(event_no):
    return f"event-{event_no}" if event_no is not None else "event-unmatched"


def list_frame_files(folder):
    files = glob.glob(os.path.join(folder, "*"))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    files.sort()
    return files


def frame_index_from_filename(path):
    """
    Last run of digits in the filename is the frame index. Using the *last* match
    (not the first) matters for names like '0001_00_f814.png' (GolfDB id, event
    position, then 'f<frame_index>' -- as extracted under frames/frames/<id>/):
    the first number there is the id (0001), not the frame index (814).
    """
    matches = FRAME_INDEX_RE.findall(os.path.basename(path))
    return int(matches[-1]) if matches else None


def keypoints_to_dict(keypoints: dict, dataset: str):
    names = joints_dict()[dataset]["keypoints"]
    people = []
    for pid, kp in keypoints.items():
        points = [
            {"name": names.get(idx, str(idx)), "y": round(float(y), 2), "x": round(float(x), 2),
             "score": round(float(score), 3)}
            for idx, (y, x, score) in enumerate(kp)
        ]
        people.append({"id": int(pid), "points": points})
    return people


def process_frame(model, img_rgb, club_confidence, club_enabled):
    """Run body pose + club detection on one RGB frame. Returns (annotated_bgr, record_dict)."""
    body_keypoints = model.inference(img_rgb)
    body_rgb = model.draw(show_yolo=False)
    annotated_bgr = cv2.cvtColor(body_rgb, cv2.COLOR_RGB2BGR)
    # Single-pose pipeline: the first (lowest-id) detected person anchors the club search.
    first_person = body_keypoints[min(body_keypoints)] if body_keypoints else None

    club_points, club_error, head_candidates_seen = {"grip": None, "head": None}, None, None
    if club_enabled:
        try:
            preds = run_golf_club_detection(img_rgb, confidence=club_confidence)
            selection = select_grip_head(preds, first_person)
            club_points = {"grip": selection.get("grip"), "head": selection.get("head")}
            head_candidates_seen = selection.get("head_candidates_seen")
        except GolfClubDetectionError as e:
            club_error = str(e)
    draw_club_keypoints(annotated_bgr, club_points)

    detector_source_names = ("detector", "detector_relaxed", "detector_midshaft_approx")
    detector_confidences = [
        club_points[k]["confidence"] for k in ("grip", "head")
        if club_points.get(k) and club_points[k].get("source") in detector_source_names
    ]
    record = {
        "body": keypoints_to_dict(body_keypoints, model.dataset),
        "club": club_points,
        "club_head_candidates_seen": head_candidates_seen,
        "club_box_confidence": round(max(detector_confidences), 3) if detector_confidences else None,
        "club_error": club_error,
    }
    return annotated_bgr, record


def run_on_frame_folder(args, model, progress_cb=None):
    frame_files = list_frame_files(args.input)
    assert frame_files, f"Tidak ada gambar (.png/.jpg) di folder {args.input}"

    golfdb_row = None
    if args.golfdb:
        rows = load_golfdb_rows(args.golfdb)
        if args.golfdb_id is not None:
            golfdb_row = next((r for r in rows if r["id"] == str(args.golfdb_id)), None)
        else:
            frame_indices = [frame_index_from_filename(f) for f in frame_files]
            golfdb_row = find_matching_golfdb_row(rows, [i for i in frame_indices if i is not None])
        if golfdb_row:
            print(f">>> Matched golfDB.csv row id={golfdb_row['id']} "
                  f"player={golfdb_row['player']} club={golfdb_row['club']} view={golfdb_row['view']}")
        else:
            print(">>> No matching golfDB.csv row found for these frames (event labels will be omitted)")

    os.makedirs(args.output_path, exist_ok=True)
    frames_out = []
    for i, frame_path in enumerate(tqdm.tqdm(frame_files, desc="Frames")):
        if progress_cb:
            progress_cb(i, len(frame_files), os.path.basename(frame_path))

        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        annotated_bgr, record = process_frame(model, img_rgb, args.club_confidence, not args.no_club)

        frame_idx = frame_index_from_filename(frame_path)
        event_no = event_no_for_frame(golfdb_row, frame_idx)
        record["frame_file"] = os.path.basename(frame_path)
        record["frame_index"] = frame_idx
        record["event"] = event_label_for_frame(golfdb_row, frame_idx)

        # One subfolder per GolfDB swing event (hasil/<run>/event-<no>/<frame>_result.png)
        # so all frames for a given event are easy to find without scrolling a flat folder.
        event_dir = os.path.join(args.output_path, event_folder_name(event_no))
        os.makedirs(event_dir, exist_ok=True)
        out_name = os.path.basename(frame_path).rsplit(".", 1)[0] + "_result.png"
        out_path = os.path.join(event_dir, out_name)
        cv2.imwrite(out_path, annotated_bgr)
        record["output_image"] = os.path.relpath(out_path, args.output_path)

        frames_out.append(record)

    if progress_cb:
        progress_cb(len(frame_files), len(frame_files), None)

    write_json(args, model, golfdb_row, frames_out)
    write_csv(args, model, golfdb_row, frames_out)
    write_csv_long(args, golfdb_row, frames_out)


def run_on_video(args, model):
    reader = VideoReader(args.input, 0)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    golfdb_row = None
    if args.golfdb:
        rows = load_golfdb_rows(args.golfdb)
        if args.golfdb_id is not None:
            golfdb_row = next((r for r in rows if r["id"] == str(args.golfdb_id)), None)
            if golfdb_row:
                print(f">>> Using golfDB.csv row id={golfdb_row['id']} "
                      f"player={golfdb_row['player']} club={golfdb_row['club']} view={golfdb_row['view']}")

    os.makedirs(args.output_path, exist_ok=True)
    base_name = os.path.basename(args.input).rsplit(".", 1)[0]
    out_video_path = os.path.join(args.output_path, f"{base_name}_result.mp4")
    writer = None

    frames_out = []
    for idx, img_rgb in enumerate(tqdm.tqdm(reader, total=total_frames, desc="Frames")):
        annotated_bgr, record = process_frame(model, img_rgb, args.club_confidence, not args.no_club)

        if writer is None:
            h, w = annotated_bgr.shape[:2]
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(annotated_bgr)

        record["frame_file"] = None
        record["frame_index"] = idx
        record["event"] = event_label_for_frame(golfdb_row, idx)

        event_no = event_no_for_frame(golfdb_row, idx)
        if event_no is not None:
            # Matched GolfDB event frame -- also save it as its own image under
            # event-<no>/, same layout as the frame-folder pipeline, so event stills
            # are easy to find without scrubbing through the annotated video.
            event_dir = os.path.join(args.output_path, event_folder_name(event_no))
            os.makedirs(event_dir, exist_ok=True)
            out_img_path = os.path.join(event_dir, f"{base_name}_frame{idx:05d}_result.png")
            cv2.imwrite(out_img_path, annotated_bgr)
            record["output_image"] = os.path.relpath(out_img_path, args.output_path)

        frames_out.append(record)

    if writer is not None:
        writer.release()
    print(f">>> Saved annotated video: {out_video_path}")

    write_json(args, model, golfdb_row, frames_out)
    write_csv(args, model, golfdb_row, frames_out)
    write_csv_long(args, golfdb_row, frames_out)


def write_json(args, model, golfdb_row, frames_out):
    out_json = os.path.join(args.output_path, "golf_result.json")
    payload = {
        "skeleton": joints_dict()[model.dataset]["keypoints"],
        "club_keypoints": ["grip", "head"],
        "golfdb_row": golfdb_row,
        "frames": frames_out,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, cls=NumpyEncoder, indent=2)
    print(f">>> Saved combined body+club JSON: {out_json}")

    n_club_ok = sum(1 for fr in frames_out if fr["club"]["grip"] and fr["club"]["head"])
    print(f">>> Club grip+head detected in {n_club_ok}/{len(frames_out)} frames")

    missing_head = [fr for fr in frames_out if not fr["club"]["head"]]
    if missing_head:
        never_seen = sum(1 for fr in missing_head if not fr.get("club_head_candidates_seen"))
        print(f">>> Head missing in {len(missing_head)} frame(s): {never_seen} where the detector "
              f"never fired, {len(missing_head) - never_seen} where candidate boxes existed but were "
              f"filtered out by distance (see club_head_candidates_seen / club_head_source in the CSV).")


# Columns copied straight from golfDB.csv, kept constant across every row of one run
# (one video/frame-set = one golfer/swing, so these don't vary frame-to-frame).
GOLFDB_META_COLUMNS = ["id", "youtube_id", "player", "sex", "club", "view", "slow", "split"]


def write_csv(args, model, golfdb_row, frames_out):
    """
    Write one row per frame: golfDB metadata + frame/event label + every
    ViTPose body keypoint (x, y, score) as its own column + the club grip
    and head points (x, y, confidence) + overall club box confidence.

    This is golfDB.csv's row shape extended with exactly the two things this
    pipeline adds on top of it: body keypoints (ViTPose) and club keypoints
    (grip + head, via golf_client.py).
    """
    keypoint_names = [name for _, name in sorted(joints_dict()[model.dataset]["keypoints"].items())]

    fieldnames = (
        list(GOLFDB_META_COLUMNS)
        + ["frame_file", "frame_index", "event", "num_people_detected"]
        + [f"{name}_{axis}" for name in keypoint_names for axis in ("x", "y", "score")]
        + ["club_grip_x", "club_grip_y", "club_grip_confidence", "club_grip_source",
           "club_head_x", "club_head_y", "club_head_confidence", "club_head_source",
           "club_head_candidates_seen", "club_box_confidence", "club_error", "output_image"]
    )

    out_csv = os.path.join(args.output_path, "golf_result.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fr in frames_out:
            row = {col: (golfdb_row.get(col) if golfdb_row else "") for col in GOLFDB_META_COLUMNS}
            row["frame_file"] = fr.get("frame_file") or ""
            row["frame_index"] = fr.get("frame_index")
            row["event"] = fr.get("event") or ""
            row["num_people_detected"] = len(fr["body"])

            # Single-pose pipeline: take the first (lowest-id) detected person, if any.
            person = min(fr["body"], key=lambda p: p["id"]) if fr["body"] else None
            points_by_name = {p["name"]: p for p in person["points"]} if person else {}
            for name in keypoint_names:
                pt = points_by_name.get(name)
                row[f"{name}_x"] = pt["x"] if pt else ""
                row[f"{name}_y"] = pt["y"] if pt else ""
                row[f"{name}_score"] = pt["score"] if pt else ""

            grip, head = fr["club"]["grip"], fr["club"]["head"]
            row["club_grip_x"] = grip["x"] if grip else ""
            row["club_grip_y"] = grip["y"] if grip else ""
            row["club_grip_confidence"] = grip["confidence"] if grip else ""
            row["club_grip_source"] = grip["source"] if grip else ""
            row["club_head_x"] = head["x"] if head else ""
            row["club_head_y"] = head["y"] if head else ""
            row["club_head_confidence"] = head["confidence"] if head else ""
            row["club_head_source"] = head["source"] if head else ""
            row["club_head_candidates_seen"] = fr.get("club_head_candidates_seen")
            if row["club_head_candidates_seen"] is None:
                row["club_head_candidates_seen"] = ""
            row["club_box_confidence"] = fr.get("club_box_confidence") or ""
            row["club_error"] = fr.get("club_error") or ""
            row["output_image"] = fr.get("output_image") or ""
            writer.writerow(row)

    print(f">>> Saved combined body+club CSV: {out_csv} ({len(fieldnames)} columns, {len(frames_out)} rows)")


LONG_FIELDNAMES = ["event_no", "frame_index", "event", "category", "point", "x", "y", "score", "source"]


def write_csv_long(args, golfdb_row, frames_out):
    """
    Long/tidy format: one row per (event, point) instead of one row per
    event with dozens of columns -- easier to scroll through top-to-bottom
    in a plain editor than golf_result.csv's 71-column wide layout.

    Each ViTPose body keypoint and each club point (grip, head) gets its own
    row, tagged with which event/frame it came from (event_no is this
    swing's 1-based position in golfDB's event list, e.g. 2=address,
    6=impact) so rows stay traceable back to golf_result.json.
    """
    out_csv = os.path.join(args.output_path, "golf_result_long.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_FIELDNAMES)
        writer.writeheader()
        for fr in frames_out:
            frame_idx = fr.get("frame_index")
            event_no = event_no_for_frame(golfdb_row, frame_idx)
            base = {"event_no": event_no if event_no is not None else "", "frame_index": frame_idx,
                    "event": fr.get("event") or ""}

            person = min(fr["body"], key=lambda p: p["id"]) if fr["body"] else None
            for pt in (person["points"] if person else []):
                writer.writerow({**base, "category": "body", "point": pt["name"],
                                  "x": pt["x"], "y": pt["y"], "score": pt["score"], "source": "vitpose"})

            for label in ("grip", "head"):
                pt = fr["club"][label]
                writer.writerow({
                    **base, "category": "club", "point": label,
                    "x": pt["x"] if pt else "", "y": pt["y"] if pt else "",
                    "score": pt["confidence"] if pt else "", "source": pt["source"] if pt else "",
                })

    n_rows = sum((len(fr["body"][0]["points"]) if fr["body"] else 0) + 2 for fr in frames_out)
    print(f">>> Saved long-format body+club CSV: {out_csv} ({len(LONG_FIELDNAMES)} columns, {n_rows} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of frame images, or a video file (.mp4/.avi/.mov)")
    parser.add_argument("--output-path", default=None,
                         help="Where to write annotated frames/video + JSON "
                              "(default: hasil/<input name>, one folder per swing)")
    parser.add_argument("--golfdb", default="golfDB.csv", help="Path to golfDB.csv (used to label swing events)")
    parser.add_argument("--golfdb-id", type=int, default=None, help="golfDB.csv row id to use (skip auto-match)")
    parser.add_argument("--model", default="models/vitpose-l-coco.onnx", help="ViTPose checkpoint path")
    parser.add_argument("--yolo", default="models/yolov8l.pt", help="YOLO checkpoint path (person detector)")
    parser.add_argument("--club-confidence", type=float, default=10,
                         help="Min confidence (0-100) for the club detection model")
    parser.add_argument("--no-club", action="store_true", help="Skip club detection, body pose only")
    args = parser.parse_args()

    if args.output_path is None:
        run_name = os.path.basename(args.input.rstrip("/\\"))
        if os.path.isfile(args.input):
            run_name = run_name.rsplit(".", 1)[0]
        args.output_path = os.path.join("hasil", run_name)

    if not args.no_club and not is_api_key_configured():
        print(">>> WARNING: ROBOFLOW_API_KEY not set (webapp/.env or env var) -- "
              "running body pose only. Copy webapp/.env.example to webapp/.env and fill it in "
              "to enable club grip/head detection.")
        args.no_club = True

    model = VitInference(args.model, args.yolo, det_class="human", is_video=os.path.isfile(args.input),
                          single_pose=True)
    print(f">>> Model loaded: {args.model}")

    if os.path.isdir(args.input):
        run_on_frame_folder(args, model)
    else:
        assert os.path.isfile(args.input), f"Input tidak ditemukan: {args.input}"
        run_on_video(args, model)


if __name__ == "__main__":
    main()
