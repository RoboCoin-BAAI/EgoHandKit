"""Detectron2/ViTPose input adapter; selection never changes crop geometry."""

from decimal import Decimal
import gc
import hashlib
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from .failure_diagnostics import jsonable
from .pre_hamer_observation_frontend import select_pre_hamer_observations
from .physical_hand_temporal_association import (
    DEFAULT_CONFIG as DEFAULT_ASSOCIATION_CONFIG,
    PhysicalHandTemporalAssociationConfig,
)


def extract_observations(image_paths, body_detector, vitpose, fps=30):
    frames = []
    image_size = None
    for frame_idx, path in enumerate(tqdm(image_paths, desc='All-person hand observations')):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError('Observation frontend requires consistent image dimensions')
        image_size = size
        instances = body_detector(image)['instances']
        valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
        boxes = instances.pred_boxes.tensor[valid].detach().cpu().numpy()
        scores = instances.scores[valid].detach().cpu().numpy()
        observations = []
        # One person at a time bounds ViTPose activation memory, without top-1 filtering.
        for person_index, (box, score) in enumerate(zip(boxes, scores)):
            detections = np.concatenate([box, [score]])[None]
            poses = vitpose.predict_pose(image[:, :, ::-1], [detections])
            if not poses:
                continue
            points = np.asarray(poses[0]['keypoints'])
            if points.shape != (133, 3) or not np.isfinite(points).all():
                raise ValueError(f'Invalid ViTPose keypoints in frame {frame_idx}')
            for side_index, (side, keypoints) in enumerate([
                ('left', points[-42:-21]), ('right', points[-21:]),
            ]):
                reliable = keypoints[:, 2] > 0.5
                # Keep EgoHandKit's existing gate and raw hand-bbox construction.
                if reliable.sum() <= 3:
                    continue
                xy = keypoints[reliable, :2]
                bbox = np.concatenate([xy.min(axis=0), xy.max(axis=0)])
                observations.append({
                    'frame_idx': frame_idx,
                    'candidate_id': 2 * person_index + side_index,
                    'person_index': person_index,
                    'person_score': float(score),
                    'person_bbox_xyxy': box.tolist(),
                    'handedness': side,
                    'bbox_xyxy': bbox.tolist(),
                    'vitpose_keypoints_2d': keypoints.tolist(),
                    'official_gate_passed': True,
                    'source': 'detectron2_vitpose',
                })
        frames.append({
            'frame_idx': frame_idx, 'img_path': str(path),
            'time_s': frame_idx / fps, 'observations': observations,
        })
    if not frames:
        raise ValueError('Observation frontend requires at least one frame')
    timestamps = Path(image_paths[0]).parent / 'timestamps.txt'
    if timestamps.exists():
        rows = [line.split() for line in timestamps.read_text().splitlines() if line.strip()]
        if len(rows) != len(frames):
            raise ValueError('timestamps.txt must match the full input sequence')
        for index, row in enumerate(rows):
            if len(row) != 2 or int(row[0]) != index:
                raise ValueError('timestamps.txt must contain consecutive zero-based frame indices')
            frames[index]['timestamp_ns'] = int(Decimal(row[1]) * 1_000_000_000)
    return {'frames': frames, 'image_size': image_size}


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def cache_signature(
    image_paths,
    repo_root,
    fps,
    association_config: PhysicalHandTemporalAssociationConfig = DEFAULT_ASSOCIATION_CONFIG,
):
    root = Path(repo_root)
    assets = [
        root / '_DATA/detectron2/model_final_f05665.pkl',
        root / 'hmr_backends/configs/cascade_mask_rcnn_vitdet_h_75ep.py',
        root / '_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth',
    ]
    assets.extend(sorted((root / '_DATA/vitpose_ckpts/configs').rglob('*.py')))
    timestamps = Path(image_paths[0]).parent / 'timestamps.txt'
    if timestamps.exists():
        assets.append(timestamps)
    modules = sorted(Path(__file__).parent.glob('*.py')) + [root / 'vitpose_model.py']
    payload = {
        'images': [[str(path), *_file_identity(path)] for path in image_paths],
        'assets': [_file_identity(path) for path in assets],
        'code': [hashlib.sha256(path.read_bytes()).hexdigest() for path in modules],
        'fps': fps,
        'association_config': association_config.as_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_cache(path, signature):
    if path.exists():
        with path.open('rb') as handle:
            cached = pickle.load(handle)
        if cached.get('signature') == signature:
            return cached['result']
    return None


def _save_cache(path, signature, result):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump({'signature': signature, 'result': result}, handle)
    temporary.replace(path)


def _detect_with_models(image_paths, repo_root, device, fps):
    import torch
    from detectron2.config import LazyConfig
    from hmr_backends.utils.utils_detectron2 import DefaultPredictor_Lazy
    from vitpose_model import ViTPoseModel

    root = Path(repo_root)
    cfg = LazyConfig.load(str(root / 'hmr_backends/configs/cascade_mask_rcnn_vitdet_h_75ep.py'))
    cfg.train.init_checkpoint = str(root / '_DATA/detectron2/model_final_f05665.pkl')
    for head in cfg.model.roi_heads.box_predictors:
        head.test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(cfg, device=device)
    pose = ViTPoseModel(str(root), device)
    try:
        return extract_observations(image_paths, detector, pose, fps)
    finally:
        del detector, pose
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def run_observation_frontend(
    image_paths,
    out_dir,
    repo_root,
    device,
    fps=30,
    force=False,
    association_config: PhysicalHandTemporalAssociationConfig = DEFAULT_ASSOCIATION_CONFIG,
):
    if not image_paths or fps <= 0:
        raise ValueError('Observation frontend requires images and a positive FPS')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(image_paths, repo_root, fps, association_config)
    selected_path = out_dir / 'observations_selected.pkl'
    selected = None if force else _load_cache(selected_path, signature)
    if selected is None:
        raw_path = out_dir / 'observations_raw.pkl'
        raw = None if force else _load_cache(raw_path, signature)
        if raw is None:
            raw = _detect_with_models(image_paths, repo_root, device, fps)
            _save_cache(raw_path, signature, raw)
        selected = select_pre_hamer_observations(
            raw['frames'], image_size=raw['image_size'], association_config=association_config,
        )
        _save_cache(selected_path, signature, selected)
    else:
        print(f'Loading selected observation cache: {selected_path}')
    (out_dir / 'observation_selection.json').write_text(
        json.dumps(jsonable(selected), indent=2) + '\n'
    )
    print(f'Observation selection: {selected["summary"]}')
    return selected['frames']
