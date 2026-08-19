"""
GolfDB batch page backend: lists every row in golfDB.csv against the locally
extracted event frames in frames/frames/<0000-padded id>/, tracks which ones
have already been run through golf_inference.py's pipeline (hasil/<id>/), and
runs the pipeline for one id / a chosen set / "everything not done yet" as a
single background job (one shared model load, not one per id -- reloading
ViTPose+YOLO per row would dominate runtime over 1400 rows).

Resumable by design: "process what's pending" always re-scans hasil/<id>/ for
a finished golf_result.json, so stopping the job partway through and running
it again later just picks up where it left off.
"""
import json
import os
import shutil
import sys
import threading
import time
import types

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import golf_inference as gi  # noqa: E402
from easy_ViTPose.inference import VitInference  # noqa: E402
from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict  # noqa: E402
from golf_client import draw_club_keypoints, is_api_key_configured  # noqa: E402

GOLFDB_CSV_PATH = os.path.join(ROOT, 'golfDB.csv')
FRAMES_DIR = os.path.join(ROOT, 'frames', 'frames')
HASIL_DIR = os.path.join(ROOT, 'hasil')
VITPOSE_MODEL = os.path.join(ROOT, 'models', 'vitpose-l-coco.onnx')
YOLO_MODEL = os.path.join(ROOT, 'models', 'yolov8l.pt')

_model = None
_model_lock = threading.Lock()


def get_shared_model() -> VitInference:
    global _model
    with _model_lock:
        if _model is None:
            _model = VitInference(VITPOSE_MODEL, YOLO_MODEL, det_class='human',
                                   is_video=False, single_pose=True)
    return _model


def row_id_folder(row_id) -> str:
    return str(row_id).zfill(4)


_rows_cache = None
_rows_cache_mtime = None


def load_rows():
    """golfDB.csv rows, cached until the file's mtime changes (1400 rows, cheap to reparse anyway)."""
    global _rows_cache, _rows_cache_mtime
    mtime = os.path.getmtime(GOLFDB_CSV_PATH)
    if _rows_cache is None or mtime != _rows_cache_mtime:
        _rows_cache = gi.load_golfdb_rows(GOLFDB_CSV_PATH)
        _rows_cache_mtime = mtime
    return _rows_cache


def row_status(row_id) -> dict:
    folder = row_id_folder(row_id)
    frames_dir = os.path.join(FRAMES_DIR, folder)
    out_dir = os.path.join(HASIL_DIR, folder)
    json_path = os.path.join(out_dir, 'golf_result.json')

    has_frames = os.path.isdir(frames_dir)
    done = os.path.isfile(json_path)
    head_ok = total_frames = None
    if done:
        try:
            with open(json_path) as f:
                data = json.load(f)
            frames = data.get('frames', [])
            total_frames = len(frames)
            head_ok = sum(1 for fr in frames if fr.get('club', {}).get('head'))
        except Exception:
            done = False
    return {'has_frames': has_frames, 'done': done, 'head_ok': head_ok, 'total_frames': total_frames}


def list_rows_with_status() -> list:
    out = []
    for row in load_rows():
        st = row_status(row['id'])
        out.append({
            'id': row['id'], 'player': row['player'], 'sex': row['sex'],
            'club': row['club'], 'view': row['view'], 'slow': row['slow'], 'split': row['split'],
            **st,
        })
    return out


JOB_LOCK = threading.Lock()
JOB_STATE = {
    'running': False,
    'stop_requested': False,
    'current_id': None,
    'current_player': None,
    'current_frame': 0,
    'current_frame_total': 0,
    'current_frame_name': None,
    'done_count': 0,
    'total': 0,
    'frames_done': 0,
    'frames_total': 0,
    'ok_count': 0,
    'error_count': 0,
    'errors': [],
    'started_at': None,
    'finished_at': None,
    'no_club': False,
    'avg_seconds_per_id': None,
    'avg_seconds_per_frame': None,
}


def _run_batch(ids, no_club, club_confidence):
    model = get_shared_model()
    rows_by_id = {r['id']: r for r in load_rows()}

    # Frame count per id (usually 10, but count for real rather than assume it).
    per_id_frames = {rid: len(gi.list_frame_files(os.path.join(FRAMES_DIR, row_id_folder(rid)))) for rid in ids}
    frames_total = sum(per_id_frames.values())

    JOB_STATE.update(running=True, stop_requested=False, done_count=0, total=len(ids),
                      frames_done=0, frames_total=frames_total,
                      ok_count=0, error_count=0, errors=[], started_at=time.time(),
                      finished_at=None, no_club=no_club, avg_seconds_per_id=None, avg_seconds_per_frame=None)

    frames_done_before_current = 0
    for rid in ids:
        if JOB_STATE['stop_requested']:
            break
        row = rows_by_id.get(str(rid))
        folder = row_id_folder(rid)
        JOB_STATE['current_id'] = rid
        JOB_STATE['current_player'] = row['player'] if row else None
        JOB_STATE['current_frame'] = 0
        JOB_STATE['current_frame_total'] = per_id_frames.get(rid, 0)
        JOB_STATE['current_frame_name'] = None

        def _on_frame(i, total, frame_name):
            JOB_STATE['current_frame'] = i
            JOB_STATE['current_frame_total'] = total
            JOB_STATE['current_frame_name'] = frame_name
            JOB_STATE['frames_done'] = frames_done_before_current + i
            elapsed = time.time() - JOB_STATE['started_at']
            if JOB_STATE['frames_done'] > 0:
                JOB_STATE['avg_seconds_per_frame'] = round(elapsed / JOB_STATE['frames_done'], 2)

        output_path = os.path.join(HASIL_DIR, folder)
        args = types.SimpleNamespace(
            input=os.path.join(FRAMES_DIR, folder),
            output_path=output_path,
            golfdb=GOLFDB_CSV_PATH, golfdb_id=int(rid),
            club_confidence=club_confidence, no_club=no_club,
        )
        try:
            # Full replace, not merge: wipe any existing output first so a
            # reprocess can't leave stale files behind (e.g. a frame that lands
            # in a different event-<no>/ folder than last time, or a manual
            # edit from before -- reprocessing is meant to start clean).
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            gi.run_on_frame_folder(args, model, progress_cb=_on_frame)
            JOB_STATE['ok_count'] += 1
        except Exception as e:
            JOB_STATE['error_count'] += 1
            JOB_STATE['errors'].append({'id': rid, 'error': str(e)})
        JOB_STATE['done_count'] += 1
        frames_done_before_current += per_id_frames.get(rid, 0)
        JOB_STATE['frames_done'] = frames_done_before_current

        elapsed = time.time() - JOB_STATE['started_at']
        JOB_STATE['avg_seconds_per_id'] = round(elapsed / JOB_STATE['done_count'], 1)
        if frames_done_before_current > 0:
            JOB_STATE['avg_seconds_per_frame'] = round(elapsed / frames_done_before_current, 2)

    JOB_STATE['running'] = False
    JOB_STATE['current_id'] = None
    JOB_STATE['current_player'] = None
    JOB_STATE['current_frame_name'] = None
    JOB_STATE['finished_at'] = time.time()


def start_batch(ids, no_club=False, club_confidence=10.0):
    ids = [str(i) for i in ids]
    if not ids:
        raise ValueError('Tidak ada id untuk diproses.')
    if not no_club and not is_api_key_configured():
        raise ValueError(
            'ROBOFLOW_API_KEY belum di-set -- deteksi club butuh ini. '
            'Isi webapp/.env, atau centang "lewati deteksi club" untuk lanjut tanpa itu.'
        )
    with JOB_LOCK:
        if JOB_STATE['running']:
            raise RuntimeError('Batch job sudah berjalan -- tunggu sampai selesai atau hentikan dulu.')
        t = threading.Thread(target=_run_batch, args=(ids, no_club, club_confidence), daemon=True)
        t.start()
    return len(ids)


def request_stop():
    JOB_STATE['stop_requested'] = True


GOLFDB_EXTRA_COLS = ['events', 'bbox']
# golfDB.csv columns that golf_result.csv doesn't carry on its own (it only copies
# GOLFDB_META_COLUMNS from golf_inference.py) -- merged back in from golfDB.csv
# itself so the combined export has everything golfDB.csv has, not a subset.
# These stay swing-level metadata (one value per swing, not per event) in the
# wide/array exports, same as player/club/view -- not pivoted or listed per event.
ID_COLS = ['id', 'youtube_id', 'player', 'sex', 'club', 'view', 'slow', 'split'] + GOLFDB_EXTRA_COLS


def _interpolate_missing_club(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing club_grip x,y (grip only -- see below) using linear interpolation
    between the SAME swing's other, already-detected events (ordered by
    frame_index). The grip stays close to the body/hands and moves comparatively
    slowly, so a gap bracketed on both sides by a real detection is reasonably
    well-estimated by the straight line between them.

    club_head is deliberately excluded: checked visually against a real frame
    (id 0, mid-follow-through, bracketed by impact and finish -- as close as an
    interior gap gets), the straight-line estimate landed off in open fairway,
    disconnected from the visible club. The clubhead sweeps through a fast,
    curved arc even between adjacent events, which a straight line in pixel-space
    doesn't model -- confidently wrong is worse than admittedly missing, so head
    keeps falling through to the wrist fallback (or stays for manual edit on the
    detail page, which is the reliable way to fix these).

    limit_area='inside' is the safety rail for grip too: only interior gaps (a
    real detection before AND after) get filled -- edges/fully-missing swings
    are left for _fill_missing_club_with_wrist.
    """
    cols = [c for c in ('club_grip_x', 'club_grip_y') if c in df.columns]
    if not cols:
        return df
    df = df.sort_values(['id', 'frame_index'])
    was_missing = df[cols].isna()
    df[cols] = df.groupby('id')[cols].transform(
        lambda s: s.interpolate(method='linear', limit_area='inside')
    )
    if 'club_grip_source' in df.columns:
        filled = was_missing['club_grip_x'] & df['club_grip_x'].notna()
        df.loc[filled, 'club_grip_source'] = 'interpolated_same_swing'
    return df


def _fill_missing_club_with_wrist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where club_grip/club_head is missing, fill it with the wrist keypoint (whichever
    side ViTPose is more confident about) instead of leaving it blank -- the grip is
    physically held in the hand and the head passes right by the wrist at address, so
    the wrist position is a reasonable stand-in where the club detector found nothing
    at all. Marks the fill in the *_source column so it stays distinguishable from a
    real detection.
    """
    if 'left_wrist_x' not in df.columns:
        return df
    use_left = df['left_wrist_score'].fillna(0) >= df['right_wrist_score'].fillna(0)
    wrist_x = df['left_wrist_x'].where(use_left, df['right_wrist_x'])
    wrist_y = df['left_wrist_y'].where(use_left, df['right_wrist_y'])

    for label in ('grip', 'head'):
        xcol, ycol, scol = f'club_{label}_x', f'club_{label}_y', f'club_{label}_source'
        if xcol not in df.columns:
            continue
        missing = df[xcol].isna()
        df.loc[missing, xcol] = wrist_x[missing]
        df.loc[missing, ycol] = wrist_y[missing]
        if scol in df.columns:
            df.loc[missing, scol] = 'wrist_fill_export'
    return df


def _load_all_processed(drop_score_confidence=True, fill_missing_club=True) -> pd.DataFrame:
    """Every processed swing's golf_result.csv (one row per event, per swing), stacked
    and merged with golfDB.csv's own columns (events, bbox) that aren't in golf_result.csv."""
    frames = []
    for row in load_rows():
        st = row_status(row['id'])
        if not st['done']:
            continue
        csv_path = os.path.join(HASIL_DIR, row_id_folder(row['id']), 'golf_result.csv')
        if os.path.isfile(csv_path):
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise ValueError('Belum ada swing yang selesai diproses -- tidak ada data untuk digabung.')

    df = pd.concat(frames, ignore_index=True)
    df = df[df['event'].notna() & (df['event'] != '')]

    if fill_missing_club:
        df = _interpolate_missing_club(df)
        df = _fill_missing_club_with_wrist(df)

    if drop_score_confidence:
        drop_cols = [c for c in df.columns if c.endswith('_score') or c.endswith('_confidence')]
        df = df.drop(columns=drop_cols)

    golfdb_raw = pd.read_csv(GOLFDB_CSV_PATH, usecols=['id'] + GOLFDB_EXTRA_COLS)
    df = df.merge(golfdb_raw, on='id', how='left')
    return df


def build_long_csv() -> str:
    """
    All processed swings' per-frame rows stacked into one file, unchanged shape
    (one row per swing x event, same ~60 columns as a single golf_result.csv) --
    far fewer columns than the wide/pivoted version, at the cost of one swing's
    10 events being 10 separate rows instead of 1.

    Writes hasil/golfdb_long.csv and returns its path.
    """
    df = _load_all_processed()
    out_path = os.path.join(HASIL_DIR, 'golfdb_long.csv')
    df.to_csv(out_path, index=False)
    return out_path


def _feature_cols(df: pd.DataFrame) -> list:
    """
    Only the actual pose/club coordinate columns (every body keypoint's *_x/*_y,
    plus club_grip_x/y and club_head_x/y) -- not the bookkeeping columns that also
    live in golf_result.csv (frame_file, output_image, num_people_detected,
    club_*_source, club_head_candidates_seen, club_error). Those aren't numeric
    features and have no business in a clustering-ready table: mixed in, they'd
    show up as a handful of small integers or blank/text cells scattered across
    the pivoted columns, which is exactly what looked broken when pivoted wide.
    """
    return [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]


def _to_bracket_list(series) -> str:
    parts = []
    for v in series:
        if pd.isna(v):
            parts.append('null')
        else:
            parts.append(str(v))
    return '[' + ', '.join(parts) + ']'


def build_array_csv() -> str:
    """
    One row per swing, like build_wide_csv, but instead of exploding each event
    into its own column (nose_x -> address_nose_x, impact_nose_x, ...), every
    per-frame column keeps its own name and holds all 10 events' values as one
    bracketed list, in event order (nose_x -> "[520.56, 521.08, ...]") -- same
    idea as golfDB.csv's own "events" column, just applied to every pose/club
    column instead of only frame indices.

    Writes hasil/golfdb_array.csv and returns its path.
    """
    df = _load_all_processed()
    df = df.sort_values(['id', 'frame_index'])

    # 'event' isn't a coordinate, but it's worth keeping as a list here (unlike in
    # the wide export) so the row order of every other list is self-documenting.
    value_cols = ['event'] + _feature_cols(df)
    rows = []
    for id_val, g in df.groupby('id', sort=False):
        row = {col: g[col].iloc[0] for col in ID_COLS}
        for col in value_cols:
            row[col] = _to_bracket_list(g[col])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(HASIL_DIR, 'golfdb_array.csv')
    out_df.to_csv(out_path, index=False)
    return out_path


def build_wide_csv() -> str:
    """
    Combine every processed swing's golf_result.csv (one row per event, per swing)
    into a single wide table: one row per swing, with every pose/club x/y feature
    prefixed by its event name (address_nose_x, impact_club_head_y, ...) -- the
    "pivot" shape asked for. Only coordinate columns are included (see
    _feature_cols) -- score/confidence and bookkeeping columns (source, error,
    file paths) are left out so every column is a clustering-ready number.

    Writes hasil/golfdb_wide.csv and returns its path.
    """
    df = _load_all_processed()

    value_cols = _feature_cols(df)
    wide = df.pivot(index='id', columns='event', values=value_cols)
    wide.columns = [f'{event}_{col}' for col, event in wide.columns]
    wide = wide.reset_index()

    meta = df[ID_COLS].drop_duplicates(subset='id')
    wide = meta.merge(wide, on='id')

    out_path = os.path.join(HASIL_DIR, 'golfdb_wide.csv')
    wide.to_csv(out_path, index=False)
    return out_path


def _find_source_frame(row_id, frame_file, frame_index=None):
    folder = row_id_folder(row_id)
    path = os.path.join(FRAMES_DIR, folder, frame_file)
    if os.path.isfile(path):
        return path
    # Filename in JSON didn't match what's on disk (e.g. older runs used
    # frame_00473.png while frames/frames/<id>/ uses 0000_02_f473.png) -- fall
    # back to matching by the frame index embedded in the filename instead.
    if frame_index is None:
        return None
    for candidate in gi.list_frame_files(os.path.join(FRAMES_DIR, folder)):
        if gi.frame_index_from_filename(candidate) == frame_index:
            return candidate
    return None


def _redraw_frame(row_id, frame_record):
    """
    Re-render one frame's output PNG from scratch (original source frame + the body
    skeleton stored in JSON + whatever club points are currently set), so a manual
    grip/head edit is reflected in the image without needing ViTPose re-inference.
    """
    src_path = _find_source_frame(row_id, frame_record['frame_file'], frame_record.get('frame_index'))
    if src_path is None:
        raise FileNotFoundError(f"Source frame tidak ditemukan: {frame_record['frame_file']}")
    img_bgr = cv2.imread(src_path)

    people = frame_record.get('body') or []
    coco = joints_dict()['coco']
    skeleton = coco['skeleton']
    name_to_index = {name: idx for idx, name in coco['keypoints'].items()}
    for idx, person in enumerate(people):
        points = sorted(person['points'], key=lambda p: name_to_index[p['name']])
        pts_arr = np.array([[p['y'], p['x'], p['score']] for p in points], dtype=np.float32)
        img_bgr = draw_points_and_skeleton(
            img_bgr, pts_arr, skeleton, person_index=idx,
            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
            points_palette_samples=10, confidence_threshold=0.3,
        )

    draw_club_keypoints(img_bgr, frame_record.get('club'))

    folder = row_id_folder(row_id)
    out_path = os.path.join(HASIL_DIR, folder, frame_record['output_image'])
    cv2.imwrite(out_path, img_bgr)


def update_point(row_id, frame_index, label, x, y, clear=False):
    """
    Manually set (or clear) the club grip/head point for one frame, redraw that
    frame's PNG, and keep golf_result.json/csv/csv_long in sync. Used by the
    click-to-edit UI on the per-swing detail page.
    """
    if label not in ('grip', 'head'):
        raise ValueError("label harus 'grip' atau 'head'")

    folder = row_id_folder(row_id)
    json_path = os.path.join(HASIL_DIR, folder, 'golf_result.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f'Belum ada hasil untuk id {row_id}.')

    with open(json_path) as f:
        data = json.load(f)

    frame_record = next((fr for fr in data['frames'] if fr.get('frame_index') == frame_index), None)
    if frame_record is None:
        raise ValueError(f'Frame dengan frame_index={frame_index} tidak ditemukan.')

    club = frame_record.setdefault('club', {'grip': None, 'head': None})
    if clear:
        club[label] = None
    else:
        club[label] = {'x': float(x), 'y': float(y), 'confidence': 1.0, 'source': 'manual'}

    if not frame_record.get('output_image'):
        out_name = frame_record['frame_file'].rsplit('.', 1)[0] + '_result.png'
        event_no = None
        golfdb_row = data.get('golfdb_row')
        if golfdb_row:
            event_no = gi.event_no_for_frame(golfdb_row, frame_record.get('frame_index'))
        rel_dir = gi.event_folder_name(event_no)
        os.makedirs(os.path.join(HASIL_DIR, folder, rel_dir), exist_ok=True)
        frame_record['output_image'] = os.path.join(rel_dir, out_name)

    _redraw_frame(row_id, frame_record)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    args = types.SimpleNamespace(output_path=os.path.join(HASIL_DIR, folder))
    model_stub = types.SimpleNamespace(dataset='coco')
    gi.write_csv(args, model_stub, data.get('golfdb_row'), data['frames'])
    gi.write_csv_long(args, data.get('golfdb_row'), data['frames'])

    return frame_record
