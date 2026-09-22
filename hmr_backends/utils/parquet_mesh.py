"""Visualization-only MANO fitting from final camera-space joint tracks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from observation_frontend.depth_gate import (
    resolve_depth_camera_intrinsics, resolve_depth_frames, scale_camera_intrinsics,
)


COLOR_GREEN = (0.2, 0.8, 0.3)
COLOR_BLUE = (0.2, 0.4, 0.9)
OPENPOSE21_PAIRS = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
], dtype=np.int32)
PARQUET_RENDER_MESH_ALPHA = 0.22


def project_camera_joints(joints, camera_intrinsics):
    """Project OpenCV camera-space OpenPose21 joints to image keypoints."""
    joints = np.asarray(joints, dtype=np.float64)
    k = np.asarray(camera_intrinsics, dtype=np.float64)
    if joints.shape != (21, 3) or k.shape != (3, 3):
        raise ValueError('Expected joints [21,3] and camera intrinsics [3,3]')
    keypoints = np.full((21, 3), np.nan, dtype=np.float32)
    valid = np.isfinite(joints).all(axis=1) & (joints[:, 2] > 1e-8)
    keypoints[valid, 0] = k[0, 0] * joints[valid, 0] / joints[valid, 2] + k[0, 2]
    keypoints[valid, 1] = k[1, 1] * joints[valid, 1] / joints[valid, 2] + k[1, 2]
    keypoints[valid, 2] = 1.0
    return keypoints


def blend_rgba_overlay(image, rgba, *, mesh_alpha=PARQUET_RENDER_MESH_ALPHA):
    """Blend RGB/RGBA renderer output over a BGR image with scaled opacity."""
    if not 0 <= mesh_alpha <= 1:
        raise ValueError('mesh_alpha must lie in [0,1]')
    image = np.asarray(image, dtype=np.float32)
    rgba = np.asarray(rgba, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or rgba.shape[:2] != image.shape[:2] or rgba.shape[2] != 4:
        raise ValueError('Expected BGR image [H,W,3] and RGBA overlay [H,W,4]')
    alpha = np.clip(rgba[:, :, 3:] * mesh_alpha, 0.0, 1.0)
    overlay_bgr = rgba[:, :, :3][:, :, ::-1] * 255.0
    return np.rint(np.clip(image * (1 - alpha) + overlay_bgr * alpha, 0, 255)).astype(np.uint8)


def draw_openpose21_joints(image, keypoints, *, side, alpha=0.92):
    """Draw final OpenPose21 skeleton/joints on a BGR image."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (21, 3):
        raise ValueError('Expected OpenPose21 keypoints with shape [21,3]')
    color = (230, 110, 35) if side == 'right' else (55, 190, 70)
    line_color = tuple(int(c * 0.75) for c in color)
    overlay = image.copy()
    valid = np.isfinite(keypoints[:, :2]).all(axis=1) & (keypoints[:, 2] > 0)
    for a, b in OPENPOSE21_PAIRS:
        if valid[a] and valid[b]:
            pa = tuple(np.round(keypoints[a, :2]).astype(int))
            pb = tuple(np.round(keypoints[b, :2]).astype(int))
            cv2.line(overlay, pa, pb, line_color, 1, cv2.LINE_AA)
    for index, point in enumerate(keypoints):
        if not valid[index]:
            continue
        center = tuple(np.round(point[:2]).astype(int))
        radius = 2 if index in (0, 5, 9, 13, 17) else 1
        cv2.circle(overlay, center, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def fit_mano_joints(mano, joints, sides, *, steps=200):
    """Fit each sample independently; fix wrist and use a neutral shared shape.

    Left targets are mirrored into the right MANO convention before fitting.
    No backend parameters, adjacent frames, or frontend observations are used.
    """
    device = mano.v_template.device
    target = torch.as_tensor(np.asarray(joints), dtype=torch.float32, device=device)
    count = len(target)
    if target.shape != (count, 21, 3) or count == 0 or not torch.isfinite(target).all():
        raise ValueError('Expected nonempty finite [N,21,3] joints')
    if len(sides) != count or any(s not in ('left', 'right') for s in sides):
        raise ValueError('Expected one left/right side per hand')
    if steps < 1:
        raise ValueError('MANO fit steps must be positive')
    from smplx.lbs import batch_rodrigues

    mirror = torch.ones((count, 1, 3), device=device)
    mirror[:, :, 0] = torch.tensor([-1 if s == 'left' else 1 for s in sides], device=device)[:, None]
    wrist = target[:, :1].clone()
    relative = (target - wrist) * mirror
    identity = torch.eye(3, device=device).expand(count, 16, 3, 3).clone()
    betas = torch.zeros((count, 10), device=device)
    with torch.no_grad():
        neutral = mano.query({'pred_rotmat': identity, 'pred_shape': betas})
        base = neutral.joints[:, :21] - neutral.joints[:, :1]
        # Rigid palm alignment initializes global orientation without Euler angles.
        palm = [5, 9, 17]
        covariance = base[:, palm].transpose(1, 2) @ relative[:, palm]
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.eye(3, device=device).repeat(count, 1, 1)
        correction[:, 2, 2] = torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2))
        initial_rotation = vh.transpose(1, 2) @ correction @ u.transpose(1, 2)
        scale = (relative[:, 9].norm(dim=1) / base[:, 9].norm(dim=1)).clamp(0.3, 3.0)
    pose = torch.zeros((count, 16, 3), device=device, requires_grad=True)
    log_scale = scale.log().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose, log_scale], lr=0.03)
    parameters = list(mano.parameters())
    flags = [p.requires_grad for p in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)

    def geometry():
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(count, 16, 3, 3)
        rotations = torch.cat(((initial_rotation @ rotations[:, 0])[:, None], rotations[:, 1:]), dim=1)
        output = mano.query({'pred_rotmat': rotations, 'pred_shape': betas})
        scale_value = log_scale.clamp(np.log(0.3), np.log(3.0)).exp()[:, None, None]
        root = output.joints[:, :1]
        return ((output.vertices - root) * scale_value,
                (output.joints[:, :21] - root) * scale_value)

    try:
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()
                _, predicted = geometry()
                loss = ((predicted - relative) * 1000).square().mean()
                loss = loss + 0.05 * pose[:, 1:].square().mean()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            vertices, predicted = geometry()
            rmse = (predicted - relative).square().sum(dim=-1).mean(dim=-1).sqrt()
            vertices = vertices * mirror + wrist
            predicted = predicted * mirror + wrist
        return vertices.cpu().numpy(), predicted.cpu().numpy(), rmse.cpu().numpy()
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def read_tracking(path):
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if (metadata.get(b'coordinate_system') != b'opencv_x_right_y_down_z_forward'
            or metadata.get(b'joint_order') != b'openpose21'):
        raise ValueError('Parquet must declare OpenCV camera coordinates and openpose21 joints')
    rows = table.to_pylist()
    indices = [row['frame_idx'] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate Parquet frame indices')
    return {row['frame_idx']: row for row in rows}


def render_tracking_parquet(path, frames, mano, renderer, out_dir, *, depth_dir,
                            expected_reference_camera=None, fps=30, steps=200,
                            max_rmse_m=0.03, batch_size=64,
                            mesh_alpha=PARQUET_RENDER_MESH_ALPHA):
    """Render Parquet geometry only; frames supply image paths, never hand data."""
    from bbox_utils import create_video_from_images

    if steps < 1 or batch_size < 1 or not np.isfinite(max_rmse_m) or max_rmse_m <= 0:
        raise ValueError('Invalid MANO fitting settings')
    rows = read_tracking(path)
    if set(rows) != {int(frame['frame_idx']) for frame in frames}:
        raise ValueError('Parquet and image timeline frame indices differ')
    if set(rows) != set(range(len(frames))):
        raise ValueError('Expected contiguous zero-based frame indices')
    intrinsic, _ = resolve_depth_camera_intrinsics(
        depth_dir, expected_reference_camera=expected_reference_camera)
    if intrinsic is None:
        raise ValueError('Parquet mesh rendering requires registered camera calibration')
    depth_paths = resolve_depth_frames(depth_dir, len(frames))
    records, targets, sides = [], [], []
    diagnostics = []
    for frame in frames:
        index = int(frame['frame_idx'])
        row = rows[index]
        for side in ('left', 'right'):
            if not row[f'{side}_present']:
                continue
            joints = np.asarray(row[f'{side}_joints_3d'], dtype=np.float32)
            record = {'frame_idx': index, 'side': side, 'source': row['source'],
                      'rendered': False, 'rmse_m': None, 'reason': None}
            diagnostics.append(record)
            if joints.shape != (21, 3) or not np.isfinite(joints).all() or (joints[:, 2] <= 0).any():
                record['reason'] = 'invalid_camera_joints'
                continue
            records.append(record)
            targets.append(joints)
            sides.append(side)
    meshes = {}
    for start in range(0, len(targets), batch_size):
        stop = start + batch_size
        batch_targets = targets[start:stop]
        vertices, _, errors = fit_mano_joints(mano, batch_targets, sides[start:stop], steps=steps)
        for record, mesh, joints, error in zip(records[start:stop], vertices, batch_targets, errors):
            if not np.isfinite(error) or not np.isfinite(mesh).all():
                record['reason'] = 'nonfinite_fit'
            else:
                record['rmse_m'] = float(error)
                if error > max_rmse_m:
                    record['reason'] = 'fit_error_exceeds_limit'
                elif (mesh[:, 2] <= 0).any():
                    record['reason'] = 'mesh_behind_camera'
                else:
                    record['rendered'] = True
                    meshes.setdefault(record['frame_idx'], []).append((record['side'], mesh, joints))
    out_dir = Path(out_dir)
    render_dir = out_dir / 'parquet_render_frames'
    render_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in render_dir.glob('[0-9][0-9][0-9][0-9][0-9][0-9].jpg'):
        old_frame.unlink()
    for frame in frames:
        index = int(frame['frame_idx'])
        image = cv2.imread(str(frame['img_path']))
        if image is None:
            raise ValueError(f"Cannot read image: {frame['img_path']}")
        hands = meshes.get(index, [])
        if hands:
            depth = cv2.imread(str(depth_paths[index]), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f'Cannot read depth dimensions for frame {index}')
            k = scale_camera_intrinsics(intrinsic, depth.shape[:2], image.shape[:2])
            rgba, _ = renderer.render_rgba_multiple(
                [mesh for _, mesh, _ in hands], cam_t=[np.zeros(3) for _ in hands],
                render_res=np.array([image.shape[1], image.shape[0]]),
                is_right=[side == 'right' for side, _, _ in hands],
                mesh_base_color=[COLOR_BLUE if side == 'right' else COLOR_GREEN for side, _, _ in hands],
                camera_intrinsics=k,
            )
            image = blend_rgba_overlay(image, rgba, mesh_alpha=mesh_alpha)
            for side, _, joints in hands:
                keypoints = project_camera_joints(joints, k)
                image = draw_openpose21_joints(image, keypoints, side=side)
        if not cv2.imwrite(str(render_dir / f'{index:06d}.jpg'), image):
            raise OSError(f'Cannot write rendered frame {index}')
    report = {'source': str(path), 'fit_steps': steps, 'max_rmse_m': max_rmse_m,
              'mesh_alpha': mesh_alpha,
              'present_hands': len(diagnostics),
              'rendered_hands': sum(d['rendered'] for d in diagnostics), 'hands': diagnostics}
    (out_dir / 'parquet_mesh_fit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    create_video_from_images(str(render_dir), str(out_dir / 'render.mp4'), fps=fps)
    return report
