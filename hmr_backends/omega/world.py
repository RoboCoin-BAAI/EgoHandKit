from __future__ import annotations

from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from hmr_backends.omega.preprocess import OmegaPreprocessMeta, map_original_points_to_omega
from hmr_backends.utils.process import run_mano_rotmat


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def mirror_left_geometry(points: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(points, dtype=np.float32).copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def transform_points_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    points = np.asarray(points_cam, dtype=np.float32)
    pose = np.asarray(c2w, dtype=np.float32)
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def source_focal_for_backend(backend_name: str, image_hw: tuple[int, int], model_cfg, hand_meta: dict | None) -> tuple[float, float]:
    h, w = image_hw
    if backend_name == "hawor":
        if hand_meta and "img_focal" in hand_meta:
            focal = float(hand_meta["img_focal"])
        else:
            focal = 600.0
        return focal, focal
    focal = float(model_cfg.EXTRA.FOCAL_LENGTH) / float(model_cfg.MODEL.IMAGE_SIZE) * max(w, h)
    return focal, focal


def convert_cam_trans_to_omega(
    cam_trans: np.ndarray,
    source_focal_xy: tuple[float, float],
    original_hw: tuple[int, int],
    omega_intrinsic: np.ndarray,
    preprocess_meta: OmegaPreprocessMeta,
) -> np.ndarray:
    cam_t = np.asarray(cam_trans, dtype=np.float32)
    if cam_t.shape != (3,):
        raise ValueError(f"cam_trans must have shape (3,), got {cam_t.shape}")
    if abs(float(cam_t[2])) < 1e-8:
        raise ValueError(f"cam_trans z is too close to zero: {cam_t}")
    h, w = original_hw
    fx_source, fy_source = source_focal_xy
    cx_source = w / 2.0
    cy_source = h / 2.0
    u_source = fx_source * cam_t[0] / cam_t[2] + cx_source
    v_source = fy_source * cam_t[1] / cam_t[2] + cy_source
    u_omega, v_omega = map_original_points_to_omega(
        np.array([[u_source, v_source]], dtype=np.float32),
        preprocess_meta,
    )[0]
    fx_omega = float(omega_intrinsic[0, 0])
    fy_omega = float(omega_intrinsic[1, 1])
    fx_source_pre = fx_source * preprocess_meta.scale_xy[0]
    fy_source_pre = fy_source * preprocess_meta.scale_xy[1]
    z_scale = 0.5 * ((fx_omega / fx_source_pre) + (fy_omega / fy_source_pre))
    z_omega = cam_t[2] * z_scale
    x_omega = (u_omega - float(omega_intrinsic[0, 2])) * z_omega / fx_omega
    y_omega = (v_omega - float(omega_intrinsic[1, 2])) * z_omega / fy_omega
    return np.array([x_omega, y_omega, z_omega], dtype=np.float32)


def reconstruct_root_relative_mano(mano_params: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    global_orient = np.asarray(mano_params["global_orient"], dtype=np.float32)[None]
    hand_pose = np.asarray(mano_params["hand_pose"], dtype=np.float32)[None]
    betas = np.asarray(mano_params["betas"], dtype=np.float32)[None]
    output = run_mano_rotmat(global_orient, hand_pose, betas, device=device)
    vertices = output["vertices"][0].detach().cpu().numpy().astype(np.float32)
    joints = output["joints"][0].detach().cpu().numpy().astype(np.float32)
    return vertices, joints


def preprocess_meta_from_cache(value) -> OmegaPreprocessMeta:
    if isinstance(value, np.ndarray):
        value = value.item()
    return OmegaPreprocessMeta.from_dict(value)


def derive_world_root_orient(root_orient_cam: np.ndarray, c2w: np.ndarray, is_right: bool) -> np.ndarray:
    root = np.asarray(root_orient_cam, dtype=np.float32).reshape(3, 3)
    if is_right:
        root_for_geometry = root
    else:
        root_for_geometry = MIRROR_X @ root @ MIRROR_X
    return (np.asarray(c2w, dtype=np.float32)[:3, :3] @ root_for_geometry).astype(np.float32)[None]


def _empty_world_frame(img_path: str, frame: dict | None, omega_camera: dict, frame_idx: int, world_meta: dict) -> dict:
    base = dict(frame) if frame is not None else {}
    base.setdefault("mano", [])
    base.setdefault("cam_trans", [])
    base.setdefault("tracked_ids", [])
    base.setdefault("extra_data", [])
    base.setdefault("backend_meta", [])
    base["world_vertices"] = []
    base["world_joints"] = []
    base["world_root_orient"] = []
    base["world_trans"] = []
    base["omega_cam_trans"] = []
    base["omega_intrinsic"] = omega_camera["omega_intrinsic"][frame_idx]
    base["omega_extrinsic_w2c"] = omega_camera["omega_extrinsic_w2c"][frame_idx]
    base["omega_c2w"] = omega_camera["omega_c2w"][frame_idx]
    base["world_meta"] = dict(world_meta)
    return base


def derive_omega_world_results(
    camera_space_results: dict,
    omega_camera: dict,
    backend_name: str,
    backend_model_cfg,
    source_pkl_path: str | Path,
    omega_cache_path: str | Path,
    output_path: str | Path,
    device: torch.device,
) -> dict:
    image_paths = [str(x) for x in omega_camera["image_paths"].tolist()]
    world_meta = {
        "backend": backend_name,
        "omega_checkpoint": str(omega_camera.get("omega_checkpoint", "")),
        "omega_image_resolution": int(omega_camera.get("omega_image_resolution", 0)),
        "source_pkl_path": str(source_pkl_path),
        "omega_cache_path": str(omega_cache_path),
        "left_policy": "mirror_root_relative_x",
        "coordinate_convention": "Omega OpenCV camera-from-world",
        "mano_to_omega_cam_trans": "project_source_root_pixel_then_backproject_with_omega_intrinsics",
    }
    output = {}
    for frame_idx, img_path in enumerate(image_paths):
        frame = camera_space_results.get(img_path)
        if frame is None or len(frame.get("mano", [])) == 0:
            output[img_path] = _empty_world_frame(img_path, frame, omega_camera, frame_idx, world_meta)
            continue
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image for world derivation: {img_path}")
        original_hw = img.shape[:2]
        preprocess_meta = preprocess_meta_from_cache(omega_camera["preprocess_meta"][frame_idx])
        omega_intrinsic = omega_camera["omega_intrinsic"][frame_idx]
        c2w = omega_camera["omega_c2w"][frame_idx]
        backend_meta = frame.get("backend_meta") or [{} for _ in frame.get("mano", [])]
        world_vertices = []
        world_joints = []
        world_root_orient = []
        world_trans = []
        omega_cam_trans = []
        for hand_idx, mano_params in enumerate(frame.get("mano", [])):
            is_right = int(mano_params.get("is_right", 1)) == 1
            hand_meta = backend_meta[hand_idx] if hand_idx < len(backend_meta) else {}
            source_focal_xy = source_focal_for_backend(backend_name, original_hw, backend_model_cfg, hand_meta)
            cam_t_omega = convert_cam_trans_to_omega(
                np.asarray(frame["cam_trans"][hand_idx], dtype=np.float32),
                source_focal_xy,
                original_hw,
                omega_intrinsic,
                preprocess_meta,
            )
            vertices_rel, joints_rel = reconstruct_root_relative_mano(mano_params, device=device)
            if not is_right:
                vertices_rel = mirror_left_geometry(vertices_rel)
                joints_rel = mirror_left_geometry(joints_rel)
            vertices_world = transform_points_to_world(vertices_rel + cam_t_omega[None], c2w)
            joints_world = transform_points_to_world(joints_rel + cam_t_omega[None], c2w)
            trans_world = transform_points_to_world(cam_t_omega[None], c2w)[0]
            root_world = derive_world_root_orient(mano_params["global_orient"], c2w, is_right)
            world_vertices.append(vertices_world)
            world_joints.append(joints_world)
            world_root_orient.append(root_world)
            world_trans.append(trans_world)
            omega_cam_trans.append(cam_t_omega)
        new_frame = dict(frame)
        new_frame["world_vertices"] = world_vertices
        new_frame["world_joints"] = world_joints
        new_frame["world_root_orient"] = world_root_orient
        new_frame["world_trans"] = world_trans
        new_frame["omega_cam_trans"] = omega_cam_trans
        new_frame["omega_intrinsic"] = omega_intrinsic
        new_frame["omega_extrinsic_w2c"] = omega_camera["omega_extrinsic_w2c"][frame_idx]
        new_frame["omega_c2w"] = c2w
        new_frame["world_meta"] = dict(world_meta)
        output[img_path] = new_frame
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    return output
