"""Hand mesh recovery pipeline — the core orchestration layer.

Given cleaned bbox data from Pass 1/2, this module:
  1. Collects and structures per-frame hand instances
  2. Runs a backend model (HAMER / HTM / WiLoR / HAWOR) to predict MANO params
  3. Assembles frame-level result dictionaries
  4. Renders mesh overlays onto the original images
"""

from pathlib import Path
from copy import deepcopy
import os
import cv2
import numpy as np

from hmr_backends.runners.schema import HandInstance, Pass3Inputs, BackendOutputInstance
from hmr_backends.runners.factory import build_runner
from hmr_backends.utils.render_policy import build_left_hand_policy
from bbox_utils import create_video_from_images


# ---------------------------------------------------------------------------
# Rendering colors
# ---------------------------------------------------------------------------
COLOR_GREEN = (0.2, 0.8, 0.3)
COLOR_BLUE = (0.2, 0.4, 0.9)


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def _resolve_hawor_focal(args, seq_folder: Path | None):
    """Resolve focal length for HaWoR backend (est_focal.txt → default 600)."""
    img_focal = args.img_focal
    if img_focal is None and seq_folder is not None:
        focal_path = seq_folder / 'est_focal.txt'
        if focal_path.exists():
            img_focal = float(focal_path.read_text().strip())
    if img_focal is None:
        img_focal = 600.0
        if seq_folder is not None:
            try:
                (seq_folder / 'est_focal.txt').write_text(str(img_focal))
            except Exception:
                pass
    return img_focal


def _collect_observation_inputs(frames, args, backend_name):
    img_paths = [Path(frame['img_path']) for frame in frames]
    image = cv2.imread(str(img_paths[0]))
    if image is None:
        raise ValueError(f'Cannot read image: {img_paths[0]}')
    instances, segments, by_key, last_by_track = [], [], {}, {}
    for frame in frames:
        for observation in frame['selected_for_hamer']:
            side = observation['handedness']
            if side not in ('left', 'right'):
                raise ValueError(f'Unsupported original handedness: {side}')
            track = int(observation['physical_track_id'])
            fragment = int(observation['physical_track_fragment_id'])
            index = frame['frame_idx']
            key = (index, f'physical_{track}')
            if key in by_key:
                raise ValueError(f'Duplicate physical track in frame {index}: {track}')
            bbox = np.asarray(observation['bbox_xyxy'], dtype=np.float32).copy()
            center = (bbox[:2] + bbox[2:]) / 2
            size = max(bbox[2:] - bbox[:2])
            inst = HandInstance(
                frame_idx=index, img_path=frame['img_path'], hand_side=side,
                bbox=bbox, bbox_square=np.array([*center, size, size], dtype=np.float32),
                keypoints=np.asarray(observation['vitpose_keypoints_2d'], dtype=np.float32).copy(),
                observation_meta={'frontend': 'observations', **deepcopy(observation)},
            )
            instances.append(inst)
            by_key[key] = inst
            previous = last_by_track.get(track)
            # HaWoR must not cross missing frames, fragments, or crop-flip changes.
            if (previous is None or previous[0] != index - 1
                    or previous[1] != fragment or previous[2] != side):
                segment = []
                segments.append(segment)
            else:
                segment = previous[3]
            segment.append(inst)
            last_by_track[track] = (index, fragment, side, segment)
    # The frame folder itself owns its focal override.  Using its parent would
    # make every sequence under test_data/images share one est_focal.txt.
    focal = _resolve_hawor_focal(args, img_paths[0].parent) if backend_name == 'hawor' else None
    return Pass3Inputs(
        img_paths=img_paths, image_size=image.shape[:2], instances=instances,
        instances_by_key=by_key, segments_by_hand={'left': [], 'right': []},
        img_focal=focal, temporal_segments=segments,
    )


def _collect_inputs(raw_data, args, backend_name):
    """Build Pass3Inputs from cleaned bbox data."""
    if not raw_data:
        return Pass3Inputs(img_paths=[], image_size=(0, 0), instances=[],
                           instances_by_key={}, segments_by_hand={'left': [], 'right': []})
    if any('selected_for_hamer' in frame for frame in raw_data):
        if not all('selected_for_hamer' in frame for frame in raw_data):
            raise ValueError('Cannot mix observation and legacy frontend frames')
        return _collect_observation_inputs(raw_data, args, backend_name)
    img_paths = [Path(frame['img_path']) for frame in raw_data]
    first_img = cv2.imread(str(img_paths[0]))
    h, w = first_img.shape[:2]
    instances = []
    instances_by_key = {}
    segments_by_hand = {'left': [], 'right': []}
    frames_present = {'left': [], 'right': []}

    for frame_data in raw_data:
        frame_idx = frame_data['frame_idx']
        for hand_side in ('left', 'right'):
            bbox = frame_data.get(f'{hand_side}_raw_bbox')
            if bbox is None:
                bbox = frame_data.get(f'{hand_side}_bbox')
            if bbox is None:
                continue
            keypoints = frame_data.get(f'{hand_side}_keypoints')
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            side = max(x2 - x1, y2 - y1)
            bbox_square = np.array([cx, cy, side, side], dtype=np.float32)
            inst = HandInstance(
                frame_idx=frame_idx,
                img_path=frame_data['img_path'],
                hand_side=hand_side,
                bbox=np.array(bbox, dtype=np.float32),
                bbox_square=bbox_square,
                keypoints=keypoints,
            )
            instances.append(inst)
            instances_by_key[(frame_idx, hand_side)] = inst
            frames_present[hand_side].append(frame_idx)

    for hand_side in ('left', 'right'):
        frames = sorted(frames_present[hand_side])
        if not frames:
            continue
        cur = [frames[0]]
        for idx in frames[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                segments_by_hand[hand_side].append(cur)
                cur = [idx]
        segments_by_hand[hand_side].append(cur)

    img_focal = _resolve_hawor_focal(
        args, Path(raw_data[0]['img_path']).parent if raw_data else None
    ) if backend_name == 'hawor' else None
    return Pass3Inputs(
        img_paths=img_paths,
        image_size=(h, w),
        instances=instances,
        instances_by_key=instances_by_key,
        segments_by_hand=segments_by_hand,
        img_focal=img_focal,
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _assemble_results(raw_outputs, cleaned_data):
    """Convert per-instance outputs to per-frame result dicts."""
    results_dict = {}
    by_frame = {}
    for out in raw_outputs:
        by_frame.setdefault(out.img_path, []).append(out)
    for frame_data in cleaned_data:
        img_path = frame_data['img_path']
        frame_outputs = by_frame.get(img_path, [])
        mano_list, cam_list, tracked_ids, extra_data, backend_meta = [], [], [], [], []
        for fo in sorted(frame_outputs, key=lambda x: x.raw_backend_meta.get(
                'physical_track_id', 1 if x.hand_side == 'right' else 0)):
            tracked_id = fo.raw_backend_meta.get('physical_track_id', 1 if fo.hand_side == 'right' else 0)
            mano_list.append(fo.mano_params)
            cam_list.append(fo.cam_trans)
            tracked_ids.append(tracked_id)
            extra_data.append(fo.pred_keypoints_2d.tolist())
            backend_meta.append(fo.raw_backend_meta)
        results_dict[img_path] = {
            'mano': mano_list,
            'cam_trans': cam_list,
            'tracked_ids': tracked_ids,
            'tracked_time': [0] * len(tracked_ids),
            'extra_data': extra_data,
            'tid': np.array(tracked_ids),
            'shot': 0,
            'backend_meta': backend_meta,
        }
        if 'selected_for_hamer' in frame_data:
            results_dict[img_path]['track_id_semantics'] = 'anonymous_physical_slot'
            results_dict[img_path]['physical_tracks'] = deepcopy(frame_data.get('physical_tracks', []))
            for field in ('frame_idx', 'timestamp_ns', 'time_s'):
                if field in frame_data:
                    results_dict[img_path][field] = frame_data[field]
    return results_dict


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_results(raw_outputs, cleaned_data, renderer, args, out_dir, fps, left_policy, backend_bundle):
    """Render mesh overlays and stitch into a video."""
    by_frame = {}
    for out in raw_outputs:
        by_frame.setdefault(out.img_path, []).append(out)
    render_dir = os.path.join(str(out_dir), f'render_{args.backend}')
    if args.render:
        os.makedirs(render_dir, exist_ok=True)
    for frame_data in cleaned_data:
        img_path = frame_data['img_path']
        img_cv2 = cv2.imread(img_path)
        img_fn = os.path.splitext(os.path.basename(img_path))[0]
        frame_outputs = by_frame.get(img_path, [])
        if not args.render:
            continue
        if not frame_outputs:
            # Keep the source timeline even when neither hand is detected.
            cv2.imwrite(os.path.join(render_dir, f'{img_fn}.jpg'), img_cv2)
            continue
        all_verts, cam_list, render_is_right = [], [], []
        for fo in sorted(frame_outputs, key=lambda x: x.hand_side):
            tracked_id = 1 if fo.hand_side == 'right' else 0
            render_verts, render_right = left_policy.apply(fo.pred_vertices, tracked_id == 1, fo.mano_params, fo.cam_trans)
            all_verts.append(render_verts)
            cam_list.append(fo.cam_trans)
            render_is_right.append(render_right)
        hand_colors = [COLOR_BLUE if r else COLOR_GREEN for r in render_is_right]

        # Keep rendering intrinsics consistent with the camera recovery path.
        # - HaWoR recovers trans_full directly in full-image coordinates using img_focal,
        #   and its renderer is initialized with the same focal.
        # - HaMeR / HTM / WiLoR recover cam_t_full using focal scaled to the current full-image size,
        #   so rendering must use that same scaled focal instead of the crop-space config constant.
        if backend_bundle.backend_name == 'hawor':
            render_focal_length = renderer.focal_length
        else:
            img_h, img_w = img_cv2.shape[:2]
            render_focal_length = (
                backend_bundle.model_cfg.EXTRA.FOCAL_LENGTH
                / backend_bundle.model_cfg.MODEL.IMAGE_SIZE
                * max(img_w, img_h)
            )

        cam_view, _ = renderer.render_rgba_multiple(
            all_verts,
            cam_t=cam_list,
            render_res=np.array([img_cv2.shape[1], img_cv2.shape[0]]),
            is_right=render_is_right,
            mesh_base_color=hand_colors,
            scene_bg_color=(1, 1, 1),
            focal_length=render_focal_length,
        )
        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        cv2.imwrite(os.path.join(render_dir, f'{img_fn}.jpg'), 255 * input_img_overlay[:, :, ::-1])
    if args.render and os.path.exists(render_dir):
        video_path = os.path.join(str(out_dir), f'render_{args.backend}.mp4')
        create_video_from_images(render_dir, video_path, fps=fps)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_mesh_recovery(cleaned_data, backend_bundle, renderer, args, out_dir, fps=15, device=None):
    """Run the hand mesh recovery pipeline on cleaned bbox data.

    This is the main orchestration function that:
      1. Collects structured inputs from raw per-frame bbox data
      2. Runs the selected backend model (HAMER/HTM/WiLoR/HAWOR) for MANO param prediction
      3. Assembles per-frame result dictionaries
      4. Renders mesh overlays onto original images

    Args:
        cleaned_data: List of per-frame dicts with bboxes from Pass 2
        backend_bundle: BackendBundle with model, config, and render config
        renderer: Renderer instance for mesh overlay
        args: CLI args (backend, render, etc.)
        out_dir: Output directory for rendered frames/video
        fps: Output video FPS
        device: torch device
    Returns:
        Per-frame result dict with mano params, cam_trans, tracked_ids, etc.
    """
    backend_name = backend_bundle.backend_name
    print("\n" + "=" * 80)
    print(f"Pass 3 - MESH RECOVERY: Running {backend_name.upper()} on cleaned bboxes")
    print("=" * 80)
    if device is None:
        raise ValueError('run_mesh_recovery requires an explicit device')

    inputs = _collect_inputs(cleaned_data, args, backend_name)
    if backend_name == 'hawor' and inputs.img_focal is not None:
        # Keep projection, camera translation recovery, and overlay rendering on
        # the same per-sequence focal, including est_focal.txt overrides.
        renderer.focal_length = inputs.img_focal
    if not inputs.instances:
        print("No hand instances; preserving empty results and the full video timeline.")
        _render_results([], cleaned_data, renderer, args, out_dir, fps, None, backend_bundle)
        return _assemble_results([], cleaned_data)
    runner = build_runner(backend_bundle, args, device)
    left_policy = build_left_hand_policy(device=device)
    raw_outputs = runner.infer(inputs)
    results = _assemble_results(raw_outputs, cleaned_data)
    _render_results(raw_outputs, cleaned_data, renderer, args, out_dir, fps, left_policy, backend_bundle)

    print("\n" + "=" * 80)
    print(f"Pass 3 - MESH RECOVERY: {backend_name.upper()} reconstruction finished")
    print("=" * 80)
    return results
