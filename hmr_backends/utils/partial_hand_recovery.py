"""Optional HMR shape recovery when a sensor wrist anchor is unavailable."""

import numpy as np

from observation_frontend.depth_gate import depth_for_image_point, resolve_depth_frames
from hmr_backends.utils.mint_3d_consistency import convert_mint_to_camera_joints


def visible_hand(observation, image_shape, min_joints=4):
    """Require actual image support; an off-screen wrist alone is not a veto."""
    points = np.asarray(observation.get('keypoints_2d'), dtype=np.float64)
    box = np.asarray(observation.get('bbox_xyxy'), dtype=np.float64)
    if points.shape != (21, 3) or box.shape != (4,) or not np.isfinite(box).all():
        return False
    height, width = image_shape[:2]
    visible = (np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
               & (points[:, 0] >= 0) & (points[:, 0] < width)
               & (points[:, 1] >= 0) & (points[:, 1] < height))
    overlap = np.minimum(box[2:], [width, height]) - np.maximum(box[:2], [0, 0])
    return bool(visible.sum() >= min_joints and np.all(overlap > 0))


def attach_partial_hand_anchors(outputs, depth_dir, frame_count, *, depth_spread_max_m=0.08,
                                max_depth_m=None):
    """Keep HMR relative geometry; recover only its absolute wrist position.

    Three consistent visible MCP samples can estimate wrist Z by subtracting
    HMR's relative joint Z. Otherwise use the available MINT wrist position.
    These assisted outputs are not independent evidence for a MINT/HMR gate.
    """
    import cv2
    depth_paths = None
    report = []
    shapes = {}
    for output in outputs:
        meta = output.raw_backend_meta
        meta.pop('partial_hand_recovery', None)
        anchor = meta.get('depth_anchor', {})
        requested = bool(meta.get('meta', {}).get('force_frontend_fallback'))
        if not requested and anchor.get('hmr_wrist_depth_m') is not None:
            continue
        diagnostic = {'frame_idx': int(output.frame_idx), 'side': output.hand_side,
                      'status': 'unavailable', 'source': None, 'wrist_camera': None,
                      'sensor_calibrated': False, 'visible_depth_joint_ids': [],
                      'estimated_wrist_depths_m': []}
        meta['partial_hand_recovery'] = diagnostic
        report.append(diagnostic)
        joints = np.asarray(output.pred_joints_3d, dtype=np.float64)
        if (joints.shape != (21, 3) or not np.isfinite(joints).all()
                or np.linalg.norm(joints[9] - joints[0]) < 1e-6):
            diagnostic['reason'] = 'invalid_hmr_geometry'
            continue
        relative = joints - joints[:1]
        points = np.asarray(output.pred_keypoints_2d, dtype=np.float64)
        k = np.asarray(anchor.get('camera_intrinsics'), dtype=np.float64)
        can_project = (points.shape == (21, 3) and np.isfinite(points[0]).all()
                       and points[0, 2] > 0 and k.shape == (3, 3)
                       and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0)
        wrist = None
        depth = anchor.get('hmr_wrist_depth_m')
        if depth is None and anchor.get('mint_wrist_depth_m') is not None:
            mint = convert_mint_to_camera_joints(
                meta, wrist_depth_m=anchor['mint_wrist_depth_m'],
                camera_intrinsics=anchor.get('camera_intrinsics'))
            if mint is not None:
                wrist = mint[0]
                diagnostic.update(source='mint_sensor_wrist', sensor_calibrated=True)
        if can_project and depth is not None:
            diagnostic.update(source='hmr_sensor_wrist', sensor_calibrated=True)
        elif can_project and wrist is None:
            if depth_paths is None:
                depth_paths = resolve_depth_frames(depth_dir, frame_count)
            if output.img_path not in shapes:
                image = cv2.imread(str(output.img_path))
                if image is None:
                    raise ValueError(f'Cannot read partial hand image: {output.img_path}')
                shapes[output.img_path] = image.shape[:2]
            estimates, indices = [], []
            for index in (5, 9, 13, 17):
                if not np.isfinite(points[index]).all() or points[index, 2] <= 0:
                    continue
                sample = depth_for_image_point(depth_paths[int(output.frame_idx)],
                                               points[index, :2], shapes[output.img_path])
                if sample is not None and sample - relative[index, 2] > 0:
                    estimates.append(float(sample - relative[index, 2]))
                    indices.append(index)
            diagnostic.update(visible_depth_joint_ids=indices, estimated_wrist_depths_m=estimates)
            if len(estimates) >= 3 and np.ptp(estimates) <= depth_spread_max_m:
                depth = float(np.median(estimates))
                diagnostic.update(source='visible_mcp_sensor', sensor_calibrated=True)
        if can_project and depth is not None:
            wrist = np.array([(points[0, 0] - k[0, 2]) * depth / k[0, 0],
                              (points[0, 1] - k[1, 2]) * depth / k[1, 1], depth])
        if wrist is None:
            mint = convert_mint_to_camera_joints(
                meta, wrist_depth_m=anchor.get('mint_wrist_depth_m'),
                camera_intrinsics=anchor.get('camera_intrinsics'))
            if mint is not None:
                wrist = mint[0]
                calibrated = anchor.get('mint_wrist_depth_m') is not None
                diagnostic.update(source='mint_sensor_wrist' if calibrated else 'mint_frontend_wrist',
                                  sensor_calibrated=calibrated)
        if wrist is None or not np.isfinite(wrist).all() or np.any(wrist[2] + relative[:, 2] <= 0):
            diagnostic['reason'] = 'missing_valid_wrist_anchor'
            continue
        diagnostic['wrist_camera'] = wrist.tolist()
        if diagnostic['sensor_calibrated'] and max_depth_m is not None and wrist[2] > max_depth_m:
            diagnostic.update(status='depth_rejected', reason='absolute_depth_limit')
        else:
            diagnostic['status'] = 'recovered'
    return report
