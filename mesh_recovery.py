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
import json
import cv2
import numpy as np

from hmr_backends.runners.schema import HandInstance, Pass3Inputs, BackendOutputInstance
from hmr_backends.runners.factory import build_runner
from hmr_backends.utils.render_policy import build_left_hand_policy
from hmr_backends.utils.mint_style_smoother import smooth_hand_sequence
from hmr_backends.utils.endpoint_wrist_gate import apply_endpoint_wrist_gate
from hmr_backends.utils.mint_3d_consistency import (
    apply_mint_3d_consistency_gate,
    attach_depth_anchors,
    convert_hmr_to_camera_joints,
)
from bbox_utils import create_video_from_images
from pipeline_artifacts import ArtifactStore, serialize_backend_outputs
from observation_frontend.depth_gate import infer_camera_side


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
        for observation in frame['hands']:
            if observation.get('meta', {}).get('force_frontend_fallback'):
                continue
            # ViTPose handedness is retained in metadata, but HaWoR needs a
            # stable crop flip for an entire physical track.  The frontend
            # supplies a track-level side that is robust to transient flips.
            side = observation.get('backend_handedness', observation['handedness'])
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
                keypoints=np.asarray(observation['keypoints_2d'], dtype=np.float32).copy(),
                observation_meta=deepcopy(observation),
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
    if all('hands' in frame for frame in raw_data):
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
        mano_list, cam_list, tracked_ids, extra_data, backend_meta, joints_3d = [], [], [], [], [], []
        for fo in sorted(frame_outputs, key=lambda x: x.raw_backend_meta.get(
                'physical_track_id', 1 if x.hand_side == 'right' else 0)):
            tracked_id = fo.raw_backend_meta.get('physical_track_id', 1 if fo.hand_side == 'right' else 0)
            mano_list.append(fo.mano_params)
            cam_list.append(fo.cam_trans)
            tracked_ids.append(tracked_id)
            extra_data.append(fo.pred_keypoints_2d.tolist())
            backend_meta.append(fo.raw_backend_meta)
            camera_joints = convert_hmr_to_camera_joints(fo)
            fo.camera_joints_3d = camera_joints
            joints_3d.append(camera_joints)
        results_dict[img_path] = {
            'mano': mano_list,
            'cam_trans': cam_list,
            'tracked_ids': tracked_ids,
            'tracked_time': [0] * len(tracked_ids),
            'extra_data': extra_data,
            'tid': np.array(tracked_ids),
            'shot': 0,
            'backend_meta': backend_meta,
            'joints_3d': joints_3d,
        }
        if 'hands' in frame_data:
            results_dict[img_path]['track_id_semantics'] = 'anonymous_physical_slot'
            results_dict[img_path]['physical_tracks'] = deepcopy(frame_data.get('physical_tracks', []))
            for field in ('frame_idx', 'timestamp_ns', 'time_s'):
                if field in frame_data:
                    results_dict[img_path][field] = frame_data[field]
    return results_dict


def _numpy_value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rotmat_to_6d(value):
    matrix = _numpy_value(value)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _smooth_raw_outputs(raw_outputs, backend_bundle, device, args):
    """Apply MINT's camera-frame UKF/RTS smoother to HaMeR track outputs."""
    grouped = {}
    for output in raw_outputs:
        track = output.raw_backend_meta.get("physical_track_id", 1 if output.hand_side == "right" else 0)
        fragment = output.raw_backend_meta.get("physical_track_fragment_id", 0)
        grouped.setdefault((int(track), output.hand_side, int(fragment)), []).append(output)

    # Grouping by physical fragment prevents smoothing across reacquisition boundaries.
    for (_track, _side, _fragment), sequence in grouped.items():
        sequence.sort(key=lambda item: item.frame_idx)
        # Missing observations are hard segment boundaries.
        runs, current = [], []
        for output in sequence:
            if current and output.frame_idx != current[-1].frame_idx + 1:
                runs.append(current)
                current = []
            current.append(output)
        if current:
            runs.append(current)

        for smooth_run in runs:
            if len(smooth_run) < 4:
                continue
            values = []
            for output in smooth_run:
                params = output.mano_params
                global_orient = _numpy_value(params["global_orient"]).reshape(3, 3)
                hand_pose = _numpy_value(params["hand_pose"]).reshape(15, 3, 3)
                betas = _numpy_value(params["betas"]).reshape(10)
                values.append(np.concatenate([
                    np.asarray(output.cam_trans, dtype=np.float32).reshape(3),
                    _rotmat_to_6d(global_orient),
                    _rotmat_to_6d(hand_pose).reshape(90),
                    betas,
                ]))

            smoothed = smooth_hand_sequence(
                np.stack(values), np.asarray([item.frame_idx for item in smooth_run]),
                q=args.smoother_q, r=args.smoother_r, beta=args.smoother_beta,
            )
            for output, value in zip(smooth_run, smoothed):
                global_orient = _rotation6d_to_matrix(value[3:9])
                hand_pose = _rotation6d_to_matrix(value[9:99].reshape(15, 6))
                params = dict(output.mano_params)
                params["global_orient"] = global_orient.astype(np.float32)
                params["hand_pose"] = hand_pose.astype(np.float32)
                params["betas"] = value[99:109].astype(np.float32)
                output.mano_params = params
                output.cam_trans = value[:3].astype(np.float32)
                output.pred_vertices, output.pred_joints_3d = _recompute_geometry(
                    backend_bundle.model, params, device)
                output.raw_backend_meta["temporal_smoother"] = {
                    "q": float(args.smoother_q),
                    "r": float(args.smoother_r),
                    "beta": float(args.smoother_beta),
                    "track_id": int(_track),
                    "track_fragment_id": int(_fragment),
                }

def _rotation6d_to_matrix(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1, 6)
    a0, a1 = value[:, :3], value[:, 3:]
    b0 = a0 / np.maximum(np.linalg.norm(a0, axis=1, keepdims=True), 1e-8)
    a1 = a1 - np.sum(b0 * a1, axis=1, keepdims=True) * b0
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=1)


def _recompute_geometry(model, params, device):
    import torch
    rotmat = np.concatenate([
        _numpy_value(params['global_orient']).reshape(1, 3, 3),
        _numpy_value(params['hand_pose']).reshape(15, 3, 3),
    ], axis=0)
    with torch.no_grad():
        output = model.mano.query({
            'pred_rotmat': torch.from_numpy(rotmat).unsqueeze(0).float().to(device),
            'pred_shape': torch.from_numpy(_numpy_value(params['betas']).reshape(1, 10)).float().to(device),
        })
    return (
        output.vertices[0].detach().cpu().numpy(),
        output.joints[0, :21].detach().cpu().numpy(),
    )


def _reproject_smoothed_outputs(raw_outputs, backend_bundle):
    """Keep final 2D rays consistent with smoothed MANO/camera parameters."""
    for output in raw_outputs:
        joints = np.asarray(output.pred_joints_3d, dtype=np.float64).copy()
        if joints.shape != (21, 3):
            continue
        if output.hand_side == "left":
            joints[:, 0] *= -1
        camera_joints = joints + np.asarray(output.cam_trans, dtype=np.float64)
        if not np.isfinite(camera_joints).all() or np.any(camera_joints[:, 2] <= 1e-8):
            continue
        image = cv2.imread(str(output.img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image for smoothed projection: {output.img_path}")
        height, width = image.shape[:2]
        if backend_bundle.backend_name == "hawor":
            focal = float(output.raw_backend_meta["img_focal"])
        else:
            focal = (
                backend_bundle.model_cfg.EXTRA.FOCAL_LENGTH
                / backend_bundle.model_cfg.MODEL.IMAGE_SIZE
                * max(width, height)
            )
        projected = np.column_stack((
            focal * camera_joints[:, 0] / camera_joints[:, 2] + width / 2.0,
            focal * camera_joints[:, 1] / camera_joints[:, 2] + height / 2.0,
            np.ones(21, dtype=np.float64),
        ))
        output.pred_keypoints_2d = projected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_results(raw_outputs, cleaned_data, renderer, args, out_dir, fps, left_policy, backend_bundle):
    """Render mesh overlays and stitch into a video."""
    by_frame = {}
    for out in raw_outputs:
        by_frame.setdefault(out.img_path, []).append(out)
    render_dir = os.path.join(str(out_dir), 'final', 'render_frames')
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
        rendered_bgr = np.clip(
            255.0 * input_img_overlay[:, :, ::-1], 0, 255
        ).astype(np.uint8)
        cv2.imwrite(os.path.join(render_dir, f'{img_fn}.jpg'), rendered_bgr)
    if args.render and os.path.exists(render_dir):
        video_path = os.path.join(str(out_dir), 'final', 'render.mp4')
        create_video_from_images(render_dir, video_path, fps=fps)


def _export_hand_tracking_if_enabled(cleaned_data, results, args, out_dir):
    if not getattr(args, 'hand_tracking_parquet', False):
        return None
    from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet
    return export_hand_tracking_parquet(
        cleaned_data, results, Path(out_dir) / "final" / "hand_tracking.parquet",
        depth_dir=args.depth_dir,
        expected_reference_camera=infer_camera_side(getattr(args, "input", None)),
    )


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
        if getattr(args, 'mint_3d_consistency_gate', False):
            _, consistency_report = apply_mint_3d_consistency_gate(
                [],
                wrist_distance_max_m=args.mint_wrist_distance_max_m,
                wrist_vector_angle_max_deg=args.mint_wrist_vector_angle_max_deg,
                hand_scale_min=args.mint_hand_scale_min,
                hand_scale_max=args.mint_hand_scale_max,
            )
            artifacts = ArtifactStore(out_dir)
            artifacts.write_json(
                artifacts.stage_dir("45_mint_3d_consistency") / "mint_3d_consistency.json",
                consistency_report,
            )
            artifacts.write_pickle(
                artifacts.stage_dir("45_mint_3d_consistency") / "checked_outputs.pkl", []
            )
            artifacts.write_pickle(
                artifacts.stage_dir("45_mint_3d_consistency") / "outputs.pkl", []
            )
        if getattr(args, 'endpoint_wrist_gate', False):
            _, pose_report = apply_endpoint_wrist_gate([], endpoint_wrist_max_deg=args.endpoint_wrist_max_deg)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            artifacts = ArtifactStore(out_dir)
            artifacts.write_json(artifacts.stage_dir("50_endpoint_wrist_gate") / "report.json", pose_report)
        results = _assemble_results([], cleaned_data)
        _export_hand_tracking_if_enabled(cleaned_data, results, args, out_dir)
        _render_results([], cleaned_data, renderer, args, out_dir, fps, None, backend_bundle)
        return results
    runner = build_runner(backend_bundle, args, device)
    left_policy = build_left_hand_policy(device=device)
    raw_outputs = runner.infer(inputs)
    if (getattr(args, 'mint_3d_consistency_gate', False)
            or getattr(args, 'hand_tracking_parquet', False)):
        attach_depth_anchors(
            raw_outputs, args.depth_dir, len(cleaned_data),
            expected_reference_camera=infer_camera_side(getattr(args, "input", None)),
        )
    artifacts = ArtifactStore(out_dir)
    artifacts.write_pickle(artifacts.stage_dir("40_backend_raw") / "outputs.pkl", serialize_backend_outputs(raw_outputs))
    artifacts.write_json(artifacts.stage_dir("40_backend_raw") / "summary.json", {
        "schema_version": "egohand.backend_outputs.v1", "count": len(raw_outputs), "stage": "pre_gate_pre_smoother"})
    if getattr(args, 'mint_3d_consistency_gate', False):
        checked_outputs = raw_outputs
        raw_outputs, consistency_report = apply_mint_3d_consistency_gate(
            checked_outputs,
            wrist_distance_max_m=args.mint_wrist_distance_max_m,
            wrist_vector_angle_max_deg=args.mint_wrist_vector_angle_max_deg,
            hand_scale_min=args.mint_hand_scale_min,
            hand_scale_max=args.mint_hand_scale_max,
        )
        consistency_stage = artifacts.stage_dir("45_mint_3d_consistency")
        artifacts.write_pickle(
            consistency_stage / "checked_outputs.pkl",
            serialize_backend_outputs(checked_outputs),
        )
        artifacts.write_pickle(
            consistency_stage / "outputs.pkl", serialize_backend_outputs(raw_outputs)
        )
        artifacts.write_json(
            consistency_stage / "mint_3d_consistency.json", consistency_report
        )
        print(
            "MINT 3D consistency: "
            f"kept {len(raw_outputs)}/{len(checked_outputs)} HMR hands "
            f"(rejected {consistency_report['rejected_hand_count']})"
        )
    if getattr(args, 'endpoint_wrist_gate', False):
        raw_outputs, pose_report = apply_endpoint_wrist_gate(
            raw_outputs, endpoint_wrist_max_deg=args.endpoint_wrist_max_deg)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        artifacts.write_pickle(artifacts.stage_dir("50_endpoint_wrist_gate") / "outputs.pkl", serialize_backend_outputs(raw_outputs))
        artifacts.write_json(artifacts.stage_dir("50_endpoint_wrist_gate") / "report.json", pose_report)
    if getattr(args, 'temporal_smoother', False):
        _smooth_raw_outputs(raw_outputs, backend_bundle, device, args)
        if (getattr(args, 'mint_3d_consistency_gate', False)
                or getattr(args, 'hand_tracking_parquet', False)):
            _reproject_smoothed_outputs(raw_outputs, backend_bundle)
            attach_depth_anchors(
                raw_outputs, args.depth_dir, len(cleaned_data),
                expected_reference_camera=infer_camera_side(getattr(args, "input", None)),
            )
        artifacts.write_pickle(artifacts.stage_dir("60_smoother") / "outputs.pkl", serialize_backend_outputs(raw_outputs))
        artifacts.write_json(artifacts.stage_dir("60_smoother") / "summary.json", {
            "schema_version": "egohand.smoother.v1", "q": args.smoother_q,
            "r": args.smoother_r, "beta": args.smoother_beta, "count": len(raw_outputs)})
    results = _assemble_results(raw_outputs, cleaned_data)
    _export_hand_tracking_if_enabled(cleaned_data, results, args, out_dir)
    _render_results(raw_outputs, cleaned_data, renderer, args, out_dir, fps, left_policy, backend_bundle)

    print("\n" + "=" * 80)
    print(f"Pass 3 - MESH RECOVERY: {backend_name.upper()} reconstruction finished")
    print("=" * 80)
    return results
