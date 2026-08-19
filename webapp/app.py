import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
from flask import (
    Flask, abort, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from easy_ViTPose import VitInference  # noqa: E402
from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader  # noqa: E402
from easy_ViTPose.vit_utils.visualization import joints_dict  # noqa: E402

from shuttlecock_client import (  # noqa: E402
    ShuttlecockDetectionError,
    is_api_key_configured,
    run_shuttlecock_workflow,
    save_result_images,
    split_workflow_outputs,
)
from court_client import (  # noqa: E402
    CourtDetectionError,
    best_court_detection,
    build_homography,
    build_homography_from_corners,
    draw_minimap,
    project_points,
    run_court_detection,
)
from tracknet_client import TrackNetError, run_tracknet_on_video  # noqa: E402
from tracknet_client import is_available as is_tracknet_available  # noqa: E402
import golfdb_batch  # noqa: E402
from golf_client import (  # noqa: E402
    GolfClubDetectionError,
    draw_club_keypoints,
    run_golf_club_detection,
    select_grip_head,
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'bmp'}
VIDEO_EXTS = {'mp4', 'avi', 'mov'}

MODEL_CONFIG = {
    'human': dict(
        model=os.path.join(ROOT, 'models', 'vitpose-l-coco.onnx'),
        yolo=os.path.join(ROOT, 'models', 'yolov8l.pt'),
        det_class='human',
        label='Manusia',
    ),
    'animal': dict(
        model=os.path.join(ROOT, 'models', 'vitpose-l-ap10k.onnx'),
        yolo=os.path.join(ROOT, 'models', 'yolov8l.pt'),
        det_class='animals',
        label='Hewan',
    ),
}

SHUTTLECOCK_MODE = 'shuttlecock'
SHUTTLECOCK_LABEL = 'Shuttlecock / Kok Badminton (Roboflow)'

COMBINED_MODE = 'combined'
COMBINED_LABEL = 'Manusia + Shuttlecock (Gabungan)'
SHUTTLECOCK_BOX_COLOR = (0, 215, 255)  # BGR

COURT_MODE = 'court'
COURT_LABEL = 'Posisi Pemain + Kok di Lapangan (gambar saja)'
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16

TRACKNET_MODE = 'tracknet'
TRACKNET_LABEL = 'Shuttlecock - Trajectory Tracking / TrackNetV3 (video saja, lokal)'

GOLF_MODE = 'golf'
GOLF_LABEL = 'Golf: Body + Club (Grip & Head, Roboflow)'

app = Flask(__name__)
_model_cache = {}
JOBS = {}
_jobs_lock = threading.Lock()


def get_model(mode: str, is_video: bool) -> VitInference:
    key = (mode, is_video)
    if key not in _model_cache:
        cfg = MODEL_CONFIG[mode]
        _model_cache[key] = VitInference(
            cfg['model'], cfg['yolo'], det_class=cfg['det_class'], is_video=is_video,
        )
    return _model_cache[key]


def ext_of(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


def trim_video_to_pct(src_path: str, pct: int) -> str:
    """
    Trim a video to its first `pct`% (by duration), in place (overwrites
    src_path). No-op if pct >= 100, ffprobe/ffmpeg fails, or duration can't
    be determined -- the original file is left untouched and used as-is,
    since trimming is a nice-to-have speed optimization, not something
    that should block processing if it fails.
    """
    if pct >= 100:
        return src_path
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', src_path],
            check=True, capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip())
    except Exception:
        return src_path

    target_duration = max(duration * pct / 100.0, 0.5)
    trimmed_path = src_path + '.trimmed.mp4'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', src_path, '-t', f'{target_duration:.2f}',
             '-c', 'copy', trimmed_path],
            check=True, capture_output=True, timeout=60,
        )
        os.replace(trimmed_path, src_path)
    except Exception:
        if os.path.isfile(trimmed_path):
            os.remove(trimmed_path)
    return src_path


def keypoints_to_persons(keypoints: dict, dataset: str) -> list:
    names = joints_dict()[dataset]['keypoints']
    persons = []
    for pid, kp in keypoints.items():
        points = [
            {
                'name': names.get(idx, str(idx)),
                'y': round(float(y), 2),
                'x': round(float(x), 2),
                'score': round(float(score), 3),
            }
            for idx, (y, x, score) in enumerate(kp)
        ]
        persons.append({'id': int(pid), 'points': points})
    return persons


def flatten_detections(detections_by_key: dict) -> list:
    flat = []
    for values in detections_by_key.values():
        flat.extend(values)
    return flat


def draw_shuttlecock_boxes(img_bgr: np.ndarray, detections: list,
                            color=SHUTTLECOCK_BOX_COLOR) -> np.ndarray:
    """Draw Roboflow-style center-box detections ({x,y,width,height,...}) onto a BGR image in place."""
    for det in detections:
        x, y, w, h = det.get('x'), det.get('y'), det.get('width'), det.get('height')
        if None in (x, y, w, h):
            continue
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        conf = det.get('confidence')
        label = det.get('class', 'shuttlecock')
        text = f'{label} {conf:.2f}' if isinstance(conf, (int, float)) else str(label)
        cv2.putText(img_bgr, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img_bgr


def process_image(src_path: str, mode: str, run_id: str, progress_cb=None) -> dict:
    model = get_model(mode, is_video=False)
    if progress_cb:
        progress_cb(0, 1, 'Memuat model...')

    img_bgr = cv2.imread(src_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if progress_cb:
        progress_cb(0, 1, 'Mendeteksi & menghitung keypoint...')
    keypoints = model.inference(img_rgb)
    result_rgb = model.draw(show_yolo=False)

    result_name = f'{run_id}_result.png'
    result_path = os.path.join(RESULT_DIR, result_name)
    cv2.imwrite(result_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {
        'keypoints': {str(k): v for k, v in keypoints.items()},
        'skeleton': skeleton_names,
    }
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(1, 1, 'Selesai')

    return {
        'kind': 'image',
        'result_image': result_name,
        'json_file': json_name,
        'persons': keypoints_to_persons(keypoints, model.dataset),
        'num_detected': len(keypoints),
    }


def process_video(src_path: str, mode: str, run_id: str, progress_cb=None) -> dict:
    model = get_model(mode, is_video=True)
    model.reset()
    if progress_cb:
        progress_cb(0, 1, 'Memuat model...')

    reader = VideoReader(src_path)
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    raw_name = f'{run_id}_raw.mp4'
    raw_path = os.path.join(RESULT_DIR, raw_name)
    writer = None

    all_keypoints = []
    num_frames = 0
    max_persons = 0

    for img in reader:
        frame_keypoints = model.inference(img)
        drawn = model.draw(show_yolo=False)[..., ::-1]  # RGB -> BGR for cv2

        if writer is None:
            h, w = drawn.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(drawn)

        all_keypoints.append(frame_keypoints)
        max_persons = max(max_persons, len(frame_keypoints))
        num_frames += 1
        if progress_cb:
            progress_cb(num_frames, total_frames, f'Memproses frame {num_frames}/{total_frames}')

    if writer is not None:
        writer.release()

    result_name = f'{run_id}_result.mp4'
    result_path = os.path.join(RESULT_DIR, result_name)
    transcoded = False
    if progress_cb:
        progress_cb(num_frames, total_frames, 'Mengonversi video (ffmpeg)...')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', result_path],
            check=True, capture_output=True, timeout=600,
        )
        transcoded = True
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, result_path)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {'keypoints': all_keypoints, 'skeleton': skeleton_names}
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(total_frames, total_frames, 'Selesai')

    return {
        'kind': 'video',
        'result_video': result_name,
        'json_file': json_name,
        'num_frames': num_frames,
        'total_frames': total_frames,
        'max_persons': max_persons,
        'transcoded': transcoded,
    }


def process_image_golf(src_path: str, run_id: str, progress_cb=None) -> dict:
    model = get_model('human', is_video=False)
    if progress_cb:
        progress_cb(0, 2, 'Memuat model...')

    img_bgr = cv2.imread(src_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if progress_cb:
        progress_cb(0, 2, 'Mendeteksi pose tubuh (ViTPose)...')
    keypoints = model.inference(img_rgb)
    pose_rgb = model.draw(show_yolo=False)
    combined_bgr = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)

    if progress_cb:
        progress_cb(1, 2, 'Mendeteksi grip & head club (Roboflow)...')
    first_person = keypoints[min(keypoints)] if keypoints else None
    club_points = {'grip': None, 'head': None}
    club_error = None
    try:
        preds = run_golf_club_detection(img_rgb)
        club_points = select_grip_head(preds, first_person)
    except GolfClubDetectionError as e:
        club_error = str(e)
    draw_club_keypoints(combined_bgr, club_points)

    result_name = f'{run_id}_result.png'
    result_path = os.path.join(RESULT_DIR, result_name)
    cv2.imwrite(result_path, combined_bgr)

    detector_confidences = [p['confidence'] for p in club_points.values() if p and p.get('source') == 'detector']
    club_box_confidence = round(max(detector_confidences), 3) if detector_confidences else None
    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {
        'keypoints': {str(k): v for k, v in keypoints.items()},
        'skeleton': skeleton_names,
        'club': club_points,
        'club_box_confidence': club_box_confidence,
        'club_error': club_error,
    }
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(2, 2, 'Selesai')

    return {
        'kind': 'golf_image',
        'result_image': result_name,
        'json_file': json_name,
        'persons': keypoints_to_persons(keypoints, model.dataset),
        'num_detected': len(keypoints),
        'club': club_points,
        'club_box_confidence': json_payload['club_box_confidence'],
        'club_error': club_error,
    }


def process_video_golf(src_path: str, run_id: str, progress_cb=None) -> dict:
    model = get_model('human', is_video=True)
    model.reset()
    if progress_cb:
        progress_cb(0, 1, 'Memuat model...')

    reader = VideoReader(src_path)
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    raw_name = f'{run_id}_raw.mp4'
    raw_path = os.path.join(RESULT_DIR, raw_name)
    writer = None

    all_keypoints = []
    all_club = []
    num_frames = 0
    max_persons = 0
    n_club_detected = 0
    last_error = None

    for img in reader:
        frame_keypoints = model.inference(img)
        pose_rgb = model.draw(show_yolo=False)
        drawn = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)
        first_person = frame_keypoints[min(frame_keypoints)] if frame_keypoints else None

        club_points = {'grip': None, 'head': None}
        try:
            preds = run_golf_club_detection(img)
            club_points = select_grip_head(preds, first_person)
        except GolfClubDetectionError as e:
            last_error = str(e)
        draw_club_keypoints(drawn, club_points)

        if club_points['grip'] and club_points['head']:
            n_club_detected += 1

        if writer is None:
            h, w = drawn.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(drawn)

        all_keypoints.append(frame_keypoints)
        all_club.append(club_points)
        max_persons = max(max_persons, len(frame_keypoints))
        num_frames += 1
        if progress_cb:
            progress_cb(num_frames, total_frames,
                        f'Frame {num_frames}/{total_frames}: pose + club (grip/head)...')

    if writer is not None:
        writer.release()

    result_name = f'{run_id}_result.mp4'
    result_path = os.path.join(RESULT_DIR, result_name)
    transcoded = False
    if progress_cb:
        progress_cb(num_frames, total_frames, 'Mengonversi video (ffmpeg)...')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', result_path],
            check=True, capture_output=True, timeout=600,
        )
        transcoded = True
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, result_path)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {
        'keypoints': all_keypoints,
        'skeleton': skeleton_names,
        'club': all_club,
    }
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(total_frames, total_frames, 'Selesai')

    return {
        'kind': 'golf_video',
        'result_video': result_name,
        'json_file': json_name,
        'num_frames': num_frames,
        'total_frames': total_frames,
        'max_persons': max_persons,
        'n_club_detected': n_club_detected,
        'transcoded': transcoded,
        'last_error': last_error,
    }


COURT_PLAUSIBLE_MARGIN_M = 1.5


def is_plausible_court_point(x_m: float, y_m: float, margin: float = COURT_PLAUSIBLE_MARGIN_M) -> bool:
    """
    Homography extrapolation can blow up (wildly wrong values) for points
    far outside the calibrated line-pair's pixel span -- e.g. a player
    standing behind the near baseline when only the near *service* line
    was confidently detected. Flag positions that land far outside the
    court's plausible extended bounds instead of presenting a precise-
    looking but nonsensical number.
    """
    from court_client import COURT_LENGTH_M, COURT_WIDTH_M
    return (-margin <= x_m <= COURT_WIDTH_M + margin) and (-margin <= y_m <= COURT_LENGTH_M + margin)


def foot_pixel_position(kp: np.ndarray, min_score: float = 0.3):
    """Average of left/right ankle pixel positions (whichever are confident enough), or None."""
    y_l, x_l, s_l = kp[LEFT_ANKLE_IDX]
    y_r, x_r, s_r = kp[RIGHT_ANKLE_IDX]
    pts = []
    if s_l >= min_score:
        pts.append((float(x_l), float(y_l)))
    if s_r >= min_score:
        pts.append((float(x_r), float(y_r)))
    if not pts:
        return None
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def process_image_shuttlecock(src_path: str, run_id: str, progress_cb=None) -> dict:
    if progress_cb:
        progress_cb(0, 1, 'Memanggil Roboflow workflow...')

    raw = run_shuttlecock_workflow(src_path)
    parsed = split_workflow_outputs(raw)
    saved_images = save_result_images(parsed['images'], RESULT_DIR, run_id)

    json_name = f'{run_id}_result.json'
    with open(os.path.join(RESULT_DIR, json_name), 'w') as f:
        json.dump(
            {'detections': parsed['detections'], 'other': parsed['other'],
             'raw_output_keys': list(raw.keys())},
            f, indent=2, default=str,
        )

    if progress_cb:
        progress_cb(1, 1, 'Selesai')

    return {
        'kind': 'shuttlecock_image',
        'result_image': next(iter(saved_images.values()), None),
        'json_file': json_name,
        'num_detected': sum(len(v) for v in parsed['detections'].values()),
        'output_keys': list(raw.keys()),
        'detections': parsed['detections'],
        'detections_flat': flatten_detections(parsed['detections']),
        'other': parsed['other'],
    }


def process_video_shuttlecock(src_path: str, run_id: str, progress_cb=None) -> dict:
    if progress_cb:
        progress_cb(0, 1, 'Memuat video...')

    reader = VideoReader(src_path)
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    raw_name = f'{run_id}_raw.mp4'
    raw_path = os.path.join(RESULT_DIR, raw_name)
    writer = None
    out_size = None

    all_detections = []
    num_frames = 0
    any_image_output = False
    last_error = None

    for frame_rgb in reader:
        parsed = {'images': {}, 'detections': {}, 'other': {}}
        try:
            raw = run_shuttlecock_workflow(frame_rgb)
            parsed = split_workflow_outputs(raw)
        except ShuttlecockDetectionError as e:
            last_error = str(e)

        drawn = frame_rgb[..., ::-1]  # fallback: original frame, RGB -> BGR
        if parsed['images']:
            any_image_output = True
            arr = np.frombuffer(next(iter(parsed['images'].values())), dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                drawn = decoded

        if out_size is None:
            out_size = (drawn.shape[1], drawn.shape[0])
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
        elif (drawn.shape[1], drawn.shape[0]) != out_size:
            drawn = cv2.resize(drawn, out_size)
        writer.write(drawn)

        all_detections.append({str(k): v for k, v in parsed['detections'].items()})
        num_frames += 1
        if progress_cb:
            progress_cb(num_frames, total_frames,
                        f'Memanggil Roboflow: frame {num_frames}/{total_frames}')

    if writer is not None:
        writer.release()

    result_name = f'{run_id}_result.mp4'
    result_path = os.path.join(RESULT_DIR, result_name)
    transcoded = False
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', result_path],
            check=True, capture_output=True, timeout=600,
        )
        transcoded = True
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, result_path)

    json_name = f'{run_id}_result.json'
    with open(os.path.join(RESULT_DIR, json_name), 'w') as f:
        json.dump({'frames': all_detections}, f, indent=2, default=str)

    if progress_cb:
        progress_cb(total_frames, total_frames, 'Selesai')

    return {
        'kind': 'shuttlecock_video',
        'result_video': result_name,
        'json_file': json_name,
        'num_frames': num_frames,
        'total_frames': total_frames,
        'transcoded': transcoded,
        'any_image_output': any_image_output,
        'last_error': last_error,
    }


def process_image_combined(src_path: str, run_id: str, progress_cb=None) -> dict:
    model = get_model('human', is_video=False)
    if progress_cb:
        progress_cb(0, 2, 'Memuat model...')

    img_bgr = cv2.imread(src_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if progress_cb:
        progress_cb(0, 2, 'Mendeteksi pose manusia...')
    keypoints = model.inference(img_rgb)
    pose_rgb = model.draw(show_yolo=False)
    combined_bgr = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)

    if progress_cb:
        progress_cb(1, 2, 'Memanggil Roboflow untuk deteksi shuttlecock...')
    detections = []
    output_keys = []
    shuttlecock_error = None
    try:
        raw = run_shuttlecock_workflow(src_path)
        parsed = split_workflow_outputs(raw)
        detections = flatten_detections(parsed['detections'])
        output_keys = list(raw.keys())
    except ShuttlecockDetectionError as e:
        shuttlecock_error = str(e)

    draw_shuttlecock_boxes(combined_bgr, detections)

    result_name = f'{run_id}_result.png'
    result_path = os.path.join(RESULT_DIR, result_name)
    cv2.imwrite(result_path, combined_bgr)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {
        'keypoints': {str(k): v for k, v in keypoints.items()},
        'skeleton': skeleton_names,
        'shuttlecock_detections': detections,
        'shuttlecock_error': shuttlecock_error,
    }
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(2, 2, 'Selesai')

    return {
        'kind': 'combined_image',
        'result_image': result_name,
        'json_file': json_name,
        'persons': keypoints_to_persons(keypoints, model.dataset),
        'num_detected': len(keypoints),
        'shuttlecock_count': len(detections),
        'shuttlecock_error': shuttlecock_error,
        'output_keys': output_keys,
        'shuttlecock_detections': detections,
    }


def process_video_combined(src_path: str, run_id: str, progress_cb=None) -> dict:
    model = get_model('human', is_video=True)
    model.reset()
    if progress_cb:
        progress_cb(0, 1, 'Memuat model...')

    reader = VideoReader(src_path)
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    raw_name = f'{run_id}_raw.mp4'
    raw_path = os.path.join(RESULT_DIR, raw_name)
    writer = None

    all_keypoints = []
    all_detections = []
    num_frames = 0
    max_persons = 0
    max_shuttlecocks = 0
    last_error = None

    for img in reader:
        frame_keypoints = model.inference(img)
        pose_rgb = model.draw(show_yolo=False)
        drawn = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)

        frame_detections = []
        try:
            raw = run_shuttlecock_workflow(img)
            parsed = split_workflow_outputs(raw)
            frame_detections = flatten_detections(parsed['detections'])
        except ShuttlecockDetectionError as e:
            last_error = str(e)

        draw_shuttlecock_boxes(drawn, frame_detections)

        if writer is None:
            h, w = drawn.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(drawn)

        all_keypoints.append(frame_keypoints)
        all_detections.append(frame_detections)
        max_persons = max(max_persons, len(frame_keypoints))
        max_shuttlecocks = max(max_shuttlecocks, len(frame_detections))
        num_frames += 1
        if progress_cb:
            progress_cb(num_frames, total_frames,
                        f'Frame {num_frames}/{total_frames}: pose + Roboflow shuttlecock...')

    if writer is not None:
        writer.release()

    result_name = f'{run_id}_result.mp4'
    result_path = os.path.join(RESULT_DIR, result_name)
    transcoded = False
    if progress_cb:
        progress_cb(num_frames, total_frames, 'Mengonversi video (ffmpeg)...')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', result_path],
            check=True, capture_output=True, timeout=600,
        )
        transcoded = True
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, result_path)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_payload = {
        'keypoints': all_keypoints,
        'skeleton': skeleton_names,
        'shuttlecock_detections': all_detections,
    }
    json_name = f'{run_id}_result.json'
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(total_frames, total_frames, 'Selesai')

    return {
        'kind': 'combined_video',
        'result_video': result_name,
        'json_file': json_name,
        'num_frames': num_frames,
        'total_frames': total_frames,
        'max_persons': max_persons,
        'max_shuttlecocks': max_shuttlecocks,
        'transcoded': transcoded,
        'last_error': last_error,
    }


def process_image_court(src_path: str, run_id: str, progress_cb=None, manual_corners=None) -> dict:
    model = get_model('human', is_video=False)
    if progress_cb:
        progress_cb(0, 3, 'Memuat model...')

    img_bgr = cv2.imread(src_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if progress_cb:
        progress_cb(0, 3, 'Mendeteksi pose pemain...')
    keypoints = model.inference(img_rgb)
    pose_rgb = model.draw(show_yolo=False)
    combined_bgr = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)

    if progress_cb:
        progress_cb(1, 3, 'Mendeteksi shuttlecock...')
    shuttlecock_detections = []
    shuttlecock_error = None
    try:
        raw_sc = run_shuttlecock_workflow(src_path)
        parsed_sc = split_workflow_outputs(raw_sc)
        shuttlecock_detections = flatten_detections(parsed_sc['detections'])
    except ShuttlecockDetectionError as e:
        shuttlecock_error = str(e)
    draw_shuttlecock_boxes(combined_bgr, shuttlecock_detections)

    if progress_cb:
        progress_cb(2, 3, 'Mendeteksi lapangan & menghitung posisi...')
    court_error = None
    player_positions = []
    shuttlecock_position = None
    minimap_name = None
    court_confidence = None
    court_keypoints = None
    court_manual = manual_corners is not None
    try:
        homography = None
        if court_manual:
            homography = build_homography_from_corners(manual_corners)
        else:
            raw_court = run_court_detection(src_path)
            det = best_court_detection(raw_court)
            if det is None:
                court_error = 'Lapangan tidak terdeteksi di gambar ini.'
            else:
                court_confidence = round(det['confidence'], 3)
                court_keypoints = det['keypoints']
                homography = build_homography(det['keypoints'])
                if homography is None:
                    court_error = 'Sudut lapangan tidak cukup jelas untuk dihitung posisinya.'

        if homography is not None:
                foot_pixels = []
                person_ids = []
                for pid, kp in keypoints.items():
                    foot = foot_pixel_position(kp)
                    if foot is not None:
                        foot_pixels.append(foot)
                        person_ids.append(int(pid))
                foot_courts = project_points(homography, foot_pixels)
                player_positions = []
                for pid, p in zip(person_ids, foot_courts):
                    x_m, y_m = round(p[0], 2), round(p[1], 2)
                    player_positions.append({
                        'id': pid, 'x_m': x_m, 'y_m': y_m,
                        'reliable': is_plausible_court_point(x_m, y_m),
                    })

                if shuttlecock_detections:
                    best_shuttle = max(shuttlecock_detections, key=lambda d: d.get('confidence', 0))
                    shuttle_court = project_points(homography, [[best_shuttle['x'], best_shuttle['y']]])[0]
                    sx_m, sy_m = round(shuttle_court[0], 2), round(shuttle_court[1], 2)
                    shuttlecock_position = {
                        'x_m': sx_m, 'y_m': sy_m,
                        'reliable': is_plausible_court_point(sx_m, sy_m),
                    }

                reliable_players = [p for p in player_positions if p['reliable']]
                minimap = draw_minimap(
                    player_points_m=[[p['x_m'], p['y_m']] for p in reliable_players],
                    player_labels=[f"P{p['id']}" for p in reliable_players],
                    shuttlecock_point_m=([shuttlecock_position['x_m'], shuttlecock_position['y_m']]
                                          if shuttlecock_position and shuttlecock_position['reliable'] else None),
                )
                minimap_name = f'{run_id}_minimap.png'
                cv2.imwrite(os.path.join(RESULT_DIR, minimap_name), minimap)
    except CourtDetectionError as e:
        court_error = str(e)

    result_name = f'{run_id}_result.png'
    cv2.imwrite(os.path.join(RESULT_DIR, result_name), combined_bgr)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    persons = keypoints_to_persons(keypoints, model.dataset)
    json_payload = {
        'keypoints': {str(k): v for k, v in keypoints.items()},
        'skeleton': skeleton_names,
        'shuttlecock_detections': shuttlecock_detections,
        'shuttlecock_error': shuttlecock_error,
        'court_confidence': court_confidence,
        'court_error': court_error,
        'court_manual': court_manual,
        'court_keypoints': court_keypoints,
        'player_positions_m': player_positions,
        'shuttlecock_position_m': shuttlecock_position,
    }
    json_name = f'{run_id}_result.json'
    with open(os.path.join(RESULT_DIR, json_name), 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(3, 3, 'Selesai')

    return {
        'kind': 'court_image',
        'result_image': result_name,
        'minimap_image': minimap_name,
        'json_file': json_name,
        'num_detected': len(keypoints),
        'persons': persons,
        'shuttlecock_count': len(shuttlecock_detections),
        'shuttlecock_detections': shuttlecock_detections,
        'shuttlecock_error': shuttlecock_error,
        'court_confidence': court_confidence,
        'court_error': court_error,
        'court_manual': court_manual,
        'court_keypoints': court_keypoints,
        'player_positions': player_positions,
        'shuttlecock_position': shuttlecock_position,
    }


def process_video_court(src_path: str, run_id: str, progress_cb=None, manual_corners=None) -> dict:
    model = get_model('human', is_video=True)
    model.reset()
    if progress_cb:
        progress_cb(0, 1, 'Memuat model...')

    reader = VideoReader(src_path)
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    raw_name = f'{run_id}_raw.mp4'
    raw_path = os.path.join(RESULT_DIR, raw_name)
    minimap_raw_name = f'{run_id}_minimap_raw.mp4'
    minimap_raw_path = os.path.join(RESULT_DIR, minimap_raw_name)
    writer = None
    minimap_writer = None

    court_manual = manual_corners is not None
    homography = build_homography_from_corners(manual_corners) if court_manual else None
    court_confidence = None
    court_error = None

    frames_timeline = []  # one entry per frame: {frame, keypoints, shuttlecock_detections, player_positions, shuttlecock_position}
    num_frames = 0
    max_persons = 0
    max_shuttlecocks = 0
    last_shuttlecock_error = None

    for img in reader:
        frame_keypoints = model.inference(img)
        pose_rgb = model.draw(show_yolo=False)
        drawn = cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)

        frame_detections = []
        try:
            raw_sc = run_shuttlecock_workflow(img)
            parsed_sc = split_workflow_outputs(raw_sc)
            frame_detections = flatten_detections(parsed_sc['detections'])
        except ShuttlecockDetectionError as e:
            last_shuttlecock_error = str(e)
        draw_shuttlecock_boxes(drawn, frame_detections)

        if homography is None and not court_manual and num_frames == 0:
            try:
                raw_court = run_court_detection(img)
                det = best_court_detection(raw_court)
                if det is None:
                    court_error = 'Lapangan tidak terdeteksi di frame pertama.'
                else:
                    court_confidence = round(det['confidence'], 3)
                    homography = build_homography(det['keypoints'])
                    if homography is None:
                        court_error = 'Sudut lapangan tidak cukup jelas untuk dihitung posisinya (frame pertama).'
            except CourtDetectionError as e:
                court_error = str(e)

        frame_player_positions = []
        frame_shuttlecock_position = None
        minimap_frame = None
        if homography is not None:
            foot_pixels = []
            person_ids = []
            for pid, kp in frame_keypoints.items():
                foot = foot_pixel_position(kp)
                if foot is not None:
                    foot_pixels.append(foot)
                    person_ids.append(int(pid))
            foot_courts = project_points(homography, foot_pixels)
            for pid, p in zip(person_ids, foot_courts):
                x_m, y_m = round(p[0], 2), round(p[1], 2)
                frame_player_positions.append({
                    'id': pid, 'x_m': x_m, 'y_m': y_m,
                    'reliable': is_plausible_court_point(x_m, y_m),
                })

            if frame_detections:
                best_shuttle = max(frame_detections, key=lambda d: d.get('confidence', 0))
                shuttle_court = project_points(homography, [[best_shuttle['x'], best_shuttle['y']]])[0]
                sx_m, sy_m = round(shuttle_court[0], 2), round(shuttle_court[1], 2)
                frame_shuttlecock_position = {
                    'x_m': sx_m, 'y_m': sy_m,
                    'reliable': is_plausible_court_point(sx_m, sy_m),
                }

            reliable_players = [p for p in frame_player_positions if p['reliable']]
            minimap_frame = draw_minimap(
                player_points_m=[[p['x_m'], p['y_m']] for p in reliable_players],
                player_labels=[f"P{p['id']}" for p in reliable_players],
                shuttlecock_point_m=([frame_shuttlecock_position['x_m'], frame_shuttlecock_position['y_m']]
                                      if frame_shuttlecock_position and frame_shuttlecock_position['reliable']
                                      else None),
            )

        if writer is None:
            h, w = drawn.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(drawn)

        if minimap_frame is not None:
            if minimap_writer is None:
                mh, mw = minimap_frame.shape[:2]
                minimap_writer = cv2.VideoWriter(minimap_raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mw, mh))
            minimap_writer.write(minimap_frame)

        frames_timeline.append({
            'frame': num_frames,
            'keypoints': {str(k): v for k, v in frame_keypoints.items()},
            'shuttlecock_detections': frame_detections,
            'player_positions_m': frame_player_positions,
            'shuttlecock_position_m': frame_shuttlecock_position,
        })
        max_persons = max(max_persons, len(frame_keypoints))
        max_shuttlecocks = max(max_shuttlecocks, len(frame_detections))
        num_frames += 1
        if progress_cb:
            progress_cb(num_frames, total_frames,
                        f'Frame {num_frames}/{total_frames}: pose + shuttlecock + posisi lapangan...')

    if writer is not None:
        writer.release()
    if minimap_writer is not None:
        minimap_writer.release()

    result_name = f'{run_id}_result.mp4'
    result_path = os.path.join(RESULT_DIR, result_name)
    transcoded = False
    if progress_cb:
        progress_cb(num_frames, total_frames, 'Mengonversi video (ffmpeg)...')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', result_path],
            check=True, capture_output=True, timeout=600,
        )
        transcoded = True
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, result_path)

    minimap_name = None
    if os.path.exists(minimap_raw_path):
        minimap_name = f'{run_id}_minimap.mp4'
        minimap_path = os.path.join(RESULT_DIR, minimap_name)
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', minimap_raw_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', minimap_path],
                check=True, capture_output=True, timeout=600,
            )
            os.remove(minimap_raw_path)
        except Exception:
            os.replace(minimap_raw_path, minimap_path)

    skeleton_names = joints_dict()[model.dataset]['keypoints']
    json_name = f'{run_id}_result.json'
    json_payload = {
        'skeleton': skeleton_names,
        'court_confidence': court_confidence,
        'court_error': court_error,
        'court_manual': court_manual,
        'frames': frames_timeline,
    }
    with open(os.path.join(RESULT_DIR, json_name), 'w') as f:
        json.dump(json_payload, f, cls=NumpyEncoder, indent=2)

    if progress_cb:
        progress_cb(total_frames, total_frames, 'Selesai')

    return {
        'kind': 'court_video',
        'result_video': result_name,
        'minimap_video': minimap_name,
        'json_file': json_name,
        'num_frames': num_frames,
        'total_frames': total_frames,
        'max_persons': max_persons,
        'max_shuttlecocks': max_shuttlecocks,
        'transcoded': transcoded,
        'court_confidence': court_confidence,
        'court_error': court_error,
        'court_manual': court_manual,
        'last_shuttlecock_error': last_shuttlecock_error,
    }


def process_video_tracknet(src_path: str, run_id: str, progress_cb=None) -> dict:
    if progress_cb:
        progress_cb(0, 1, 'Menjalankan TrackNetV3 (deteksi + rektifikasi trajektori kok)...')

    save_dir = os.path.join(RESULT_DIR, f'{run_id}_tracknet_tmp')
    result = run_tracknet_on_video(src_path, save_dir, output_video=True)
    trajectory = result['trajectory']
    visible_count = sum(1 for t in trajectory if t['visible'])

    result_name = None
    if result['video_path'] and os.path.isfile(result['video_path']):
        result_name = f'{run_id}_result.mp4'
        os.replace(result['video_path'], os.path.join(RESULT_DIR, result_name))

    json_name = f'{run_id}_result.json'
    with open(os.path.join(RESULT_DIR, json_name), 'w') as f:
        json.dump({'trajectory': trajectory}, f, indent=2)

    shutil.rmtree(save_dir, ignore_errors=True)

    if progress_cb:
        progress_cb(1, 1, 'Selesai')

    return {
        'kind': 'tracknet_video',
        'result_video': result_name,
        'json_file': json_name,
        'num_frames': len(trajectory),
        'visible_count': visible_count,
        'visible_pct': round(100 * visible_count / len(trajectory), 1) if trajectory else 0,
        'trajectory': trajectory,
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', modes=MODEL_CONFIG, shuttlecock_mode=SHUTTLECOCK_MODE,
                            shuttlecock_label=SHUTTLECOCK_LABEL, combined_mode=COMBINED_MODE,
                            combined_label=COMBINED_LABEL, court_mode=COURT_MODE,
                            court_label=COURT_LABEL, tracknet_mode=TRACKNET_MODE,
                            tracknet_label=TRACKNET_LABEL, golf_mode=GOLF_MODE,
                            golf_label=GOLF_LABEL)


def run_job(job_id: str, src_path: str, mode: str, ext: str, manual_corners=None):
    def cb(current, total, message=''):
        with _jobs_lock:
            JOBS[job_id].update(
                current=current, total=total, message=message,
                percent=int(current / total * 100) if total else 0,
            )

    t0 = time.time()
    try:
        if mode == SHUTTLECOCK_MODE:
            mode_label = SHUTTLECOCK_LABEL
            if ext in IMAGE_EXTS:
                result = process_image_shuttlecock(src_path, job_id, progress_cb=cb)
            else:
                result = process_video_shuttlecock(src_path, job_id, progress_cb=cb)
        elif mode == COMBINED_MODE:
            mode_label = COMBINED_LABEL
            if ext in IMAGE_EXTS:
                result = process_image_combined(src_path, job_id, progress_cb=cb)
            else:
                result = process_video_combined(src_path, job_id, progress_cb=cb)
        elif mode == COURT_MODE:
            mode_label = COURT_LABEL
            if ext in IMAGE_EXTS:
                result = process_image_court(src_path, job_id, progress_cb=cb, manual_corners=manual_corners)
            else:
                result = process_video_court(src_path, job_id, progress_cb=cb, manual_corners=manual_corners)
        elif mode == TRACKNET_MODE:
            mode_label = TRACKNET_LABEL
            if ext not in IMAGE_EXTS:
                result = process_video_tracknet(src_path, job_id, progress_cb=cb)
            else:
                raise ValueError('Mode TrackNetV3 cuma mendukung video, bukan gambar (butuh info gerakan antar-frame).')
        elif mode == GOLF_MODE:
            mode_label = GOLF_LABEL
            if ext in IMAGE_EXTS:
                result = process_image_golf(src_path, job_id, progress_cb=cb)
            else:
                result = process_video_golf(src_path, job_id, progress_cb=cb)
        else:
            mode_label = MODEL_CONFIG[mode]['label']
            if ext in IMAGE_EXTS:
                result = process_image(src_path, mode, job_id, progress_cb=cb)
            else:
                result = process_video(src_path, mode, job_id, progress_cb=cb)
        result['job_id'] = job_id
        with _jobs_lock:
            JOBS[job_id].update(
                status='done', result=result, elapsed=round(time.time() - t0, 2),
                mode_label=mode_label,
            )
    except Exception as e:
        with _jobs_lock:
            JOBS[job_id].update(status='error', error=str(e))


@app.route('/trim_preview', methods=['POST'])
def trim_preview():
    """Trim an uploaded video server-side and hand back a URL to preview/download it,
    without running any AI processing. Lets the user check the cut before committing."""
    file = request.files.get('media')
    if not file or file.filename == '':
        return jsonify(error='Pilih file dulu.'), 400

    filename = secure_filename(file.filename)
    ext = ext_of(filename)
    if ext not in VIDEO_EXTS:
        return jsonify(error='Preview potongan cuma buat video.'), 400

    try:
        trim_pct = int(request.form.get('trim_pct', 100))
    except (TypeError, ValueError):
        trim_pct = 100
    trim_pct = max(10, min(100, trim_pct))

    preview_name = f'preview_{uuid.uuid4().hex[:10]}_{filename}'
    preview_path = os.path.join(UPLOAD_DIR, preview_name)
    file.save(preview_path)
    trim_video_to_pct(preview_path, trim_pct)

    return jsonify(url=url_for('static', filename=f'uploads/{preview_name}'))


@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('media')
    mode = request.form.get('mode', 'human')
    if mode not in MODEL_CONFIG and mode not in (SHUTTLECOCK_MODE, COMBINED_MODE, COURT_MODE, TRACKNET_MODE, GOLF_MODE):
        mode = 'human'

    common_kwargs = dict(modes=MODEL_CONFIG, shuttlecock_mode=SHUTTLECOCK_MODE,
                          shuttlecock_label=SHUTTLECOCK_LABEL, combined_mode=COMBINED_MODE,
                          combined_label=COMBINED_LABEL, court_mode=COURT_MODE,
                          court_label=COURT_LABEL, tracknet_mode=TRACKNET_MODE,
                          tracknet_label=TRACKNET_LABEL, golf_mode=GOLF_MODE,
                          golf_label=GOLF_LABEL)

    if not file or file.filename == '':
        return render_template('index.html', error='Pilih file dulu ya.', **common_kwargs)

    filename = secure_filename(file.filename)
    ext = ext_of(filename)
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return render_template('index.html',
                                error=f'Format .{ext} belum didukung. Pakai jpg/png/mp4/mov/avi.',
                                **common_kwargs)

    if mode in (SHUTTLECOCK_MODE, COMBINED_MODE, COURT_MODE, GOLF_MODE) and not is_api_key_configured():
        return render_template(
            'index.html',
            error=("ROBOFLOW_API_KEY belum di-set. Isi webapp/.env (copy dari .env.example) "
                   "atau export ROBOFLOW_API_KEY, lalu restart server sebelum pakai mode "
                   "Shuttlecock / Gabungan / Posisi Lapangan / Golf."),
            **common_kwargs,
        )

    if mode == TRACKNET_MODE and ext in IMAGE_EXTS:
        return render_template(
            'index.html',
            error='Mode TrackNetV3 cuma mendukung video, bukan gambar (butuh info gerakan antar-frame).',
            **common_kwargs,
        )

    if mode == TRACKNET_MODE and not is_tracknet_available():
        return render_template(
            'index.html',
            error=('Model TrackNetV3 belum siap (checkpoint tidak ditemukan di models/tracknet/). '
                   'Cek TrackNet_best.pt & InpaintNet_best.pt.'),
            **common_kwargs,
        )


    manual_corners = None
    if mode == COURT_MODE:
        corners_raw = request.form.get('corners', '')
        if corners_raw:
            try:
                parsed = json.loads(corners_raw)
                if (isinstance(parsed, list) and len(parsed) == 4
                        and all(isinstance(pt, list) and len(pt) == 2 for pt in parsed)):
                    manual_corners = [[float(x), float(y)] for x, y in parsed]
            except (ValueError, TypeError):
                manual_corners = None

    job_id = uuid.uuid4().hex[:10]
    src_path = os.path.join(UPLOAD_DIR, f'{job_id}_{filename}')
    file.save(src_path)

    if ext in VIDEO_EXTS:
        try:
            trim_pct = int(request.form.get('trim_pct', 100))
        except (TypeError, ValueError):
            trim_pct = 100
        trim_pct = max(10, min(100, trim_pct))
        trim_video_to_pct(src_path, trim_pct)

    with _jobs_lock:
        JOBS[job_id] = dict(status='processing', current=0, total=1, percent=0,
                             message='Memulai...', kind='video' if ext in VIDEO_EXTS else 'image')

    threading.Thread(target=run_job, args=(job_id, src_path, mode, ext, manual_corners), daemon=True).start()

    return render_template('processing.html', job_id=job_id)


@app.route('/progress/<job_id>', methods=['GET'])
def progress(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(status='unknown'), 404
    return jsonify(
        status=job.get('status'), current=job.get('current', 0), total=job.get('total', 1),
        percent=job.get('percent', 0), message=job.get('message', ''), kind=job.get('kind'),
        error=job.get('error'),
    )


@app.route('/result/<job_id>', methods=['GET'])
def show_result(job_id):
    job = JOBS.get(job_id)
    if job is None or job.get('status') != 'done':
        return redirect(url_for('index'))
    return render_template('result.html', result=job['result'], elapsed=job['elapsed'],
                            mode_label=job['mode_label'])


def _result_video_path(job_id: str) -> str:
    path = os.path.join(RESULT_DIR, f'{job_id}_result.mp4')
    if not os.path.isfile(path):
        abort(404)
    return path


@app.route('/frame/<job_id>/<int:idx>.png', methods=['GET'])
def download_frame(job_id, idx):
    video_path = _result_video_path(job_id)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx < 0 or idx >= total:
        cap.release()
        abort(404)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(404)
    ok, buf = cv2.imencode('.png', frame)
    if not ok:
        abort(500)
    return send_file(
        io.BytesIO(buf.tobytes()), mimetype='image/png',
        as_attachment=True, download_name=f'{job_id}_frame_{idx:04d}.png',
    )


@app.route('/frames_zip/<job_id>', methods=['GET'])
def download_frames_zip(job_id):
    video_path = _result_video_path(job_id)
    cap = cv2.VideoCapture(video_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok2, img_buf = cv2.imencode('.png', frame)
            if ok2:
                zf.writestr(f'{job_id}_frame_{idx:04d}.png', img_buf.tobytes())
            idx += 1
    cap.release()
    buf.seek(0)
    return send_file(
        buf, mimetype='application/zip',
        as_attachment=True, download_name=f'{job_id}_frames.zip',
    )


@app.route('/golfdb', methods=['GET'])
def golfdb_page():
    rows = golfdb_batch.list_rows_with_status()
    n_done = sum(1 for r in rows if r['done'])
    n_ready = sum(1 for r in rows if r['has_frames'] and not r['done'])
    return render_template(
        'golfdb.html', rows=rows, n_total=len(rows), n_done=n_done, n_ready=n_ready,
        api_key_configured=is_api_key_configured(),
    )


@app.route('/golfdb/process', methods=['POST'])
def golfdb_process():
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get('mode')
    no_club = bool(data.get('no_club', False))
    club_confidence = float(data.get('club_confidence', 10))

    if mode == 'pending':
        limit = data.get('limit')
        rows = golfdb_batch.list_rows_with_status()
        ids = [r['id'] for r in rows if r['has_frames'] and not r['done']]
        if limit:
            ids = ids[:int(limit)]
    else:
        ids = [str(i) for i in data.get('ids', [])]

    if not ids:
        return jsonify(error='Tidak ada id untuk diproses (semua sudah selesai, atau tidak ada yang dipilih).'), 400

    try:
        count = golfdb_batch.start_batch(ids, no_club=no_club, club_confidence=club_confidence)
    except (RuntimeError, ValueError) as e:
        return jsonify(error=str(e)), 409
    return jsonify(started=True, count=count)


@app.route('/golfdb/stop', methods=['POST'])
def golfdb_stop():
    golfdb_batch.request_stop()
    return jsonify(stopping=True)


@app.route('/golfdb/job_status', methods=['GET'])
def golfdb_job_status():
    return jsonify(golfdb_batch.JOB_STATE)


@app.route('/golfdb/<row_id>', methods=['GET'])
def golfdb_row_detail(row_id):
    folder = golfdb_batch.row_id_folder(row_id)
    json_path = os.path.join(golfdb_batch.HASIL_DIR, folder, 'golf_result.json')
    if not os.path.isfile(json_path):
        return redirect(url_for('golfdb_page'))
    with open(json_path) as f:
        data = json.load(f)
    images = []
    for fr in data.get('frames', []):
        img = fr.get('output_image')
        if not img:
            continue
        club = fr.get('club') or {}
        images.append({
            'event': fr.get('event') or f"frame {fr.get('frame_index')}",
            'frame_index': fr.get('frame_index'),
            'url': url_for('hasil_files', subpath=f'{folder}/{img}'),
            'grip': club.get('grip'),
            'head': club.get('head'),
        })
    return render_template('golfdb_detail.html', row=data.get('golfdb_row') or {}, images=images, row_id=row_id,
                            json_url=url_for('hasil_files', subpath=f'{folder}/golf_result.json'),
                            csv_url=url_for('hasil_files', subpath=f'{folder}/golf_result.csv'),
                            zip_url=url_for('golfdb_download_zip', row_id=row_id))


@app.route('/hasil_files/<path:subpath>', methods=['GET'])
def hasil_files(subpath):
    return send_from_directory(golfdb_batch.HASIL_DIR, subpath)


@app.route('/golfdb/wide_csv', methods=['GET'])
def golfdb_wide_csv():
    try:
        path = golfdb_batch.build_wide_csv()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return send_file(path, as_attachment=True, download_name='golfdb_wide.csv')


@app.route('/golfdb/long_csv', methods=['GET'])
def golfdb_long_csv():
    try:
        path = golfdb_batch.build_long_csv()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return send_file(path, as_attachment=True, download_name='golfdb_long.csv')


@app.route('/golfdb/array_csv', methods=['GET'])
def golfdb_array_csv():
    try:
        path = golfdb_batch.build_array_csv()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return send_file(path, as_attachment=True, download_name='golfdb_array.csv')


@app.route('/golfdb/<row_id>/point', methods=['POST'])
def golfdb_update_point(row_id):
    data = request.get_json(force=True, silent=True) or {}
    try:
        frame_index = int(data['frame_index'])
        label = data['label']
        clear = bool(data.get('clear', False))
        x, y = data.get('x'), data.get('y')
        frame_record = golfdb_batch.update_point(row_id, frame_index, label, x, y, clear=clear)
    except (KeyError, ValueError, FileNotFoundError) as e:
        return jsonify(error=str(e)), 400

    folder = golfdb_batch.row_id_folder(row_id)
    return jsonify(
        ok=True,
        club=frame_record.get('club'),
        image_url=url_for('hasil_files', subpath=f"{folder}/{frame_record['output_image']}") + f'?t={int(time.time())}',
    )


@app.route('/golfdb/<row_id>/zip', methods=['GET'])
def golfdb_download_zip(row_id):
    folder = golfdb_batch.row_id_folder(row_id)
    out_dir = os.path.join(golfdb_batch.HASIL_DIR, folder)
    if not os.path.isdir(out_dir):
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, arcname=os.path.relpath(fpath, out_dir))
    buf.seek(0)
    return send_file(
        buf, mimetype='application/zip',
        as_attachment=True, download_name=f'golfdb_{folder}_images.zip',
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    # Defaults to loopback-only (safe for local use). Override with HOST=0.0.0.0
    # when running inside a container/remote VM (RunPod, Docker, etc.) whose
    # port-forwarding needs the process listening on all interfaces to reach it.
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5050'))
    app.run(host=host, port=port, debug=False, threaded=True)
