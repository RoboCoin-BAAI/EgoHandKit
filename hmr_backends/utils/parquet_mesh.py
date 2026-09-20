"""Visualization-only MANO fitting from final camera-space joint tracks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from smplx.lbs import batch_rodrigues
from hmr_backends.utils.render_policy import COLOR_GREEN, COLOR_BLUE

from observation_frontend.depth_gate import (
    resolve_depth_camera_intrinsics, resolve_depth_frames, scale_camera_intrinsics,
)


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
                            max_rmse_m=0.03, batch_size=64):
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
        vertices, _, errors = fit_mano_joints(mano, targets[start:stop], sides[start:stop], steps=steps)
        for record, mesh, error in zip(records[start:stop], vertices, errors):
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
                    meshes.setdefault(record['frame_idx'], []).append((record['side'], mesh))
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
                [mesh for _, mesh in hands], cam_t=[np.zeros(3) for _ in hands],
                render_res=np.array([image.shape[1], image.shape[0]]),
                is_right=[side == 'right' for side, _ in hands],
                mesh_base_color=[COLOR_BLUE if side == 'right' else COLOR_GREEN for side, _ in hands],
                camera_intrinsics=k,
            )
            alpha = rgba[:, :, 3:]
            image = np.clip(image * (1 - alpha) + rgba[:, :, :3][:, :, ::-1] * 255 * alpha,
                            0, 255).astype(np.uint8)
        if not cv2.imwrite(str(render_dir / f'{index:06d}.jpg'), image):
            raise OSError(f'Cannot write rendered frame {index}')
    report = {'source': str(path), 'fit_steps': steps, 'max_rmse_m': max_rmse_m,
              'present_hands': len(diagnostics),
              'rendered_hands': sum(d['rendered'] for d in diagnostics), 'hands': diagnostics}
    (out_dir / 'parquet_mesh_fit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    create_video_from_images(str(render_dir), str(out_dir / 'render.mp4'), fps=fps)
    return report
