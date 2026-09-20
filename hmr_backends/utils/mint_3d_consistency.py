"""Optional camera-space consistency gate for MINT priors and HMR outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from observation_frontend.depth_gate import (
    depth_for_image_point,
    resolve_depth_camera_intrinsics,
    resolve_depth_frames,
    scale_camera_intrinsics,
)
from observation_frontend.schema import (
    camera_intrinsics_from_observation,
    camera_joints_from_observation,
)


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
PALM_JOINTS = (INDEX_MCP, MIDDLE_MCP, PINKY_MCP)
PALM_JOINT_NAMES = ("index_mcp", "middle_mcp", "pinky_mcp")


def _joints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return array


def _depth_anchored_joints(joints: np.ndarray, keypoints_2d: Any,
                           wrist_depth_m: float, camera_intrinsics: Any) -> np.ndarray | None:
    keypoints = np.asarray(keypoints_2d, dtype=np.float64)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if keypoints.ndim == 2 and keypoints.shape[0] >= 21 and keypoints.shape[1] >= 3:
        keypoints = keypoints[:21, :3]
    if (keypoints.shape != (21, 3) or intrinsics.shape != (3, 3)
            or not np.isfinite(intrinsics).all() or not np.isfinite(wrist_depth_m)
            or wrist_depth_m <= 0 or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.isfinite(keypoints[WRIST]).all() or keypoints[WRIST, 2] <= 0):
        return None
    relative = joints - joints[[WRIST]]
    depths = float(wrist_depth_m) + relative[:, 2]
    if not np.isfinite(depths).all() or np.any(depths <= 1e-6):
        return None
    result = np.empty((21, 3), dtype=np.float64)
    result[:, 2] = depths
    valid_projection = np.isfinite(keypoints).all(axis=1) & (keypoints[:, 2] > 0)
    result[valid_projection, 0] = (
        (keypoints[valid_projection, 0] - intrinsics[0, 2])
        * depths[valid_projection] / intrinsics[0, 0]
    )
    result[valid_projection, 1] = (
        (keypoints[valid_projection, 1] - intrinsics[1, 2])
        * depths[valid_projection] / intrinsics[1, 1]
    )
    wrist = result[WRIST].copy()
    result[~valid_projection] = wrist + relative[~valid_projection]
    return result


def convert_mint_to_camera_joints(
    reference: Mapping[str, Any] | None,
    *,
    wrist_depth_m: float | None = None,
    camera_intrinsics: Any = None,
) -> np.ndarray | None:
    """Convert MINT joints, optionally anchoring them to sensor wrist depth."""
    joints = camera_joints_from_observation(reference)
    if joints is None or wrist_depth_m is None:
        return joints
    if camera_intrinsics is None:
        camera_intrinsics = camera_intrinsics_from_observation(reference)
    keypoints = reference.get("keypoints_2d") if isinstance(reference, Mapping) else None
    return _depth_anchored_joints(joints, keypoints, wrist_depth_m, camera_intrinsics)


def convert_hmr_to_camera_joints(
    output: Any,
    *,
    wrist_depth_m: float | None = None,
    camera_intrinsics: Any = None,
) -> np.ndarray | None:
    """Back-project HMR 2D joints using sensor wrist depth and relative MANO Z."""
    joints = _joints(getattr(output, "pred_joints_3d", None))
    if joints is None:
        metadata = getattr(output, "raw_backend_meta", {})
        joints = _joints(metadata.get("pred_joints_3d")) if isinstance(metadata, Mapping) else None
    if joints is None:
        return None
    joints = joints.copy()
    if getattr(output, "hand_side", None) == "left":
        joints[:, 0] *= -1
    metadata = getattr(output, "raw_backend_meta", {})
    recovery = metadata.get('partial_hand_recovery')
    if recovery is not None:
        if recovery.get('status') != 'recovered':
            return None
        wrist = np.asarray(recovery.get('wrist_camera'), dtype=np.float64)
        if wrist.shape != (3,) or not np.isfinite(wrist).all():
            return None
        result = joints - joints[:1] + wrist
        return result if np.isfinite(result).all() and np.all(result[:, 2] > 0) else None
    anchor = metadata.get("depth_anchor", {}) if isinstance(metadata, Mapping) else {}
    if wrist_depth_m is None:
        wrist_depth_m = anchor.get("hmr_wrist_depth_m")
    if camera_intrinsics is None:
        camera_intrinsics = anchor.get("camera_intrinsics")
    if wrist_depth_m is None or camera_intrinsics is None:
        # Preserve the historical result payload when the optional sensor
        # anchoring features are disabled. Once an anchor record exists,
        # missing sensor data must remain explicit rather than falling back to
        # HaMeR's virtual rendering depth.
        if anchor:
            return None
        translation = np.asarray(getattr(output, "cam_trans", None), dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            return None
        return joints + translation
    return _depth_anchored_joints(
        joints, getattr(output, "pred_keypoints_2d", None),
        wrist_depth_m, camera_intrinsics,
    )


def _valid_wrist_pixel(keypoints_2d: Any) -> np.ndarray | None:
    keypoints = np.asarray(keypoints_2d, dtype=np.float64)
    if (keypoints.ndim != 2 or keypoints.shape[0] < 1 or keypoints.shape[1] < 3
            or not np.isfinite(keypoints[WRIST, :3]).all()
            or keypoints[WRIST, 2] <= 0):
        return None
    return keypoints[WRIST, :2]


def attach_depth_anchors(outputs: Iterable[Any], depth_dir: str | Path,
                         frame_count: int, *, unit_scale: float = 0.001,
                         expected_reference_camera: str | None = None) -> None:
    """Attach sensor-wrist anchors for both MINT and HMR projected wrists."""
    rows = list(outputs)
    if not rows:
        return
    depth_paths = resolve_depth_frames(depth_dir, frame_count)
    sensor_intrinsics, sensor_intrinsics_source = resolve_depth_camera_intrinsics(
        depth_dir, expected_reference_camera=expected_reference_camera
    )
    depth_shapes: dict[Path, tuple[int, int]] = {}
    image_shapes: dict[str, tuple[int, int]] = {}
    for output in rows:
        image_shape = image_shapes.get(output.img_path)
        if image_shape is None:
            image = cv2.imread(str(output.img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot read image for HMR depth anchoring: {output.img_path}")
            image_shape = image.shape[:2]
            image_shapes[output.img_path] = image_shape
        metadata = output.raw_backend_meta
        intrinsics = sensor_intrinsics
        intrinsics_source = sensor_intrinsics_source
        depth_path = depth_paths[int(output.frame_idx)]
        if intrinsics is not None:
            depth_shape = depth_shapes.get(depth_path)
            if depth_shape is None:
                depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    raise ValueError(f"Cannot read depth frame: {depth_path}")
                depth_shape = depth_image.shape[:2]
                depth_shapes[depth_path] = depth_shape
            intrinsics = scale_camera_intrinsics(
                intrinsics, depth_shape, image_shape
            )
        if intrinsics is None:
            intrinsics = camera_intrinsics_from_observation(metadata)
            intrinsics_source = "canonical_observation"
        mint_pixel = _valid_wrist_pixel(metadata.get("keypoints_2d"))
        hmr_pixel = _valid_wrist_pixel(output.pred_keypoints_2d)
        mint_depth = (
            depth_for_image_point(depth_path, mint_pixel, image_shape, unit_scale=unit_scale)
            if mint_pixel is not None else None
        )
        hmr_depth = (
            depth_for_image_point(depth_path, hmr_pixel, image_shape, unit_scale=unit_scale)
            if hmr_pixel is not None else None
        )
        metadata["depth_anchor"] = {
            "camera_intrinsics": intrinsics.tolist() if intrinsics is not None else None,
            "camera_intrinsics_source": intrinsics_source,
            "mint_wrist_depth_m": mint_depth,
            "hmr_wrist_depth_m": hmr_depth,
            "mint_wrist_pixel": mint_pixel.tolist() if mint_pixel is not None else None,
            "hmr_wrist_pixel": hmr_pixel.tolist() if hmr_pixel is not None else None,
            "source": "registered_depth_sensor",
        }
        output.camera_joints_3d = convert_hmr_to_camera_joints(output)


def _vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _diagnostic(output: Any, *, wrist_distance_max_m: float,
                wrist_vector_angle_max_deg: float, hand_scale_min: float,
                hand_scale_max: float) -> dict[str, Any]:
    base = {
        "frame_idx": int(output.frame_idx),
        "side": str(output.hand_side),
        "wrist_distance_m": None,
        "vector_angle_deg": None,
        "vector_angles_deg": None,
        "scale_ratio": None,
        "mint_wrist_sensor_depth_m": None,
        "hmr_wrist_sensor_depth_m": None,
        "accepted": True,
        "reject_reason": None,
        "reject_reasons": [],
    }
    metadata = getattr(output, "raw_backend_meta", {})
    recovery = metadata.get('partial_hand_recovery')
    if recovery is not None:
        accepted = convert_hmr_to_camera_joints(output) is not None
        return {**base, 'accepted': accepted,
                'status': 'partial_hand_assisted' if accepted else 'missing_hmr_geometry',
                'reject_reason': None if accepted else 'missing_hmr_geometry',
                'reject_reasons': [] if accepted else ['missing_hmr_geometry'],
                'anchor_source': recovery.get('source'),
                'not_evaluated_reason': 'assisted_wrist_not_independent'}
    anchor = metadata.get("depth_anchor", {})
    base.update({
        "mint_wrist_sensor_depth_m": anchor.get("mint_wrist_depth_m"),
        "hmr_wrist_sensor_depth_m": anchor.get("hmr_wrist_depth_m"),
    })
    mint_depth = anchor.get("mint_wrist_depth_m")
    mint = (
        convert_mint_to_camera_joints(
            metadata,
            wrist_depth_m=mint_depth,
            camera_intrinsics=anchor.get("camera_intrinsics"),
        )
        if mint_depth is not None else None
    )
    if mint is None:
        return {**base, "status": "missing_mint_reference"}
    hmr = convert_hmr_to_camera_joints(output)
    if hmr is None:
        return {
            **base,
            "accepted": False,
            "reject_reason": "missing_hmr_geometry",
            "reject_reasons": ["missing_hmr_geometry"],
            "status": "missing_hmr_geometry",
        }

    wrist_distance = float(np.linalg.norm(mint[WRIST] - hmr[WRIST]))
    vector_angles = {
        name: _vector_angle_deg(mint[index] - mint[WRIST], hmr[index] - hmr[WRIST])
        for name, index in zip(PALM_JOINT_NAMES, PALM_JOINTS)
    }
    valid_vector_angles = [angle for angle in vector_angles.values() if angle is not None]
    vector_angle = max(valid_vector_angles) if len(valid_vector_angles) == len(PALM_JOINTS) else None
    mint_scale = float(np.linalg.norm(mint[MIDDLE_MCP] - mint[WRIST]))
    hmr_scale = float(np.linalg.norm(hmr[MIDDLE_MCP] - hmr[WRIST]))
    scale_ratio = hmr_scale / mint_scale if mint_scale > 1e-8 else None

    reasons = []
    if wrist_distance > wrist_distance_max_m:
        reasons.append("wrist_distance")
    if vector_angle is None:
        reasons.append("invalid_wrist_vector")
    elif vector_angle > wrist_vector_angle_max_deg:
        reasons.append("wrist_vector_angle")
    if scale_ratio is None:
        reasons.append("invalid_hand_scale")
    elif not hand_scale_min <= scale_ratio <= hand_scale_max:
        reasons.append("hand_scale")
    return {
        **base,
        "wrist_distance_m": wrist_distance,
        "vector_angle_deg": vector_angle,
        "vector_angles_deg": vector_angles,
        "scale_ratio": scale_ratio,
        "accepted": not reasons,
        "reject_reason": reasons[0] if reasons else None,
        "reject_reasons": reasons,
        "status": "checked",
    }


def apply_mint_3d_consistency_gate(
    outputs: Iterable[Any],
    *,
    wrist_distance_max_m: float = 0.08,
    wrist_vector_angle_max_deg: float = 40.0,
    hand_scale_min: float = 0.7,
    hand_scale_max: float = 1.3,
) -> tuple[list[Any], dict[str, Any]]:
    """Reject HMR samples inconsistent with an available MINT 3D prior."""
    if wrist_distance_max_m <= 0 or not 0 < wrist_vector_angle_max_deg <= 180:
        raise ValueError("wrist distance must be positive and vector angle must lie in (0,180]")
    if hand_scale_min <= 0 or hand_scale_max < hand_scale_min:
        raise ValueError("hand scale limits must satisfy 0 < min <= max")

    kept = []
    hands = []
    output_rows = list(outputs)
    for output in output_rows:
        diagnostic = _diagnostic(
            output,
            wrist_distance_max_m=wrist_distance_max_m,
            wrist_vector_angle_max_deg=wrist_vector_angle_max_deg,
            hand_scale_min=hand_scale_min,
            hand_scale_max=hand_scale_max,
        )
        output.raw_backend_meta["mint_3d_consistency"] = dict(diagnostic)
        hands.append(diagnostic)
        if diagnostic["accepted"]:
            kept.append(output)
    wrist_debug = []
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    for output in output_rows:
        joints = convert_hmr_to_camera_joints(output)
        if joints is not None:
            by_frame.setdefault(int(output.frame_idx), {})[str(output.hand_side)] = joints[WRIST]
    for frame_idx, sides in sorted(by_frame.items()):
        if "left" in sides and "right" in sides:
            wrist_debug.append({
                "frame_idx": frame_idx,
                "left_right_wrist_distance_m": float(np.linalg.norm(sides["left"] - sides["right"])),
                "decision_effect": "none",
            })
    report = {
        "schema_version": "egohand.mint_3d_consistency.v1",
        "config": {
            "wrist_distance_max_m": float(wrist_distance_max_m),
            "wrist_vector_angle_max_deg": float(wrist_vector_angle_max_deg),
            "hand_scale_min": float(hand_scale_min),
            "hand_scale_max": float(hand_scale_max),
        },
        "hands": hands,
        "left_right_wrist_debug": wrist_debug,
        "checked_hand_count": sum(hand["status"] == "checked" for hand in hands),
        "rejected_hand_count": sum(not hand["accepted"] for hand in hands),
    }
    return kept, report
