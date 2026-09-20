"""Optional camera-space consistency gate for MINT priors and HMR outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from observation_frontend.schema import camera_joints_from_observation


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


def convert_mint_to_camera_joints(reference: Mapping[str, Any] | None) -> np.ndarray | None:
    """Read a MINT 21-joint prior from canonical observation metadata."""
    return camera_joints_from_observation(reference)


def convert_hmr_to_camera_joints(output: Any) -> np.ndarray | None:
    """Convert root-relative HMR MANO joints to OpenCV camera coordinates."""
    joints = _joints(getattr(output, "pred_joints_3d", None))
    if joints is None:
        metadata = getattr(output, "raw_backend_meta", {})
        joints = _joints(metadata.get("pred_joints_3d")) if isinstance(metadata, Mapping) else None
    if joints is None:
        return None
    joints = joints.copy()
    if getattr(output, "hand_side", None) == "left":
        joints[:, 0] *= -1
    translation = np.asarray(getattr(output, "cam_trans", None), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        return None
    return joints + translation


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
        "accepted": True,
        "reject_reason": None,
        "reject_reasons": [],
    }
    mint = convert_mint_to_camera_joints(getattr(output, "raw_backend_meta", {}))
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
