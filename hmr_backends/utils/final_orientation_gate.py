"""Final MINT/HMR coarse orientation bucket gate.

The gate compares the wrist-to-middle-MCP vector (OpenPose21 joints 0 -> 9)
between MINT camera joints and HMR joints.  It is intentionally coarse: MINT
acts as a reliable position/orientation reference, while HMR supplies shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


WRIST = 0
MIDDLE_MCP = 9
VERTICAL_SHARED_HALF_ANGLE_DEG = 30.0
PROJECTED_2D_ANGLE_MATCH_DEG = 15.0


def _joints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return array


def _keypoints_2d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape[0] != 21 or array.shape[1] < 2 or not np.isfinite(array[:, :2]).all():
        return None
    return array[:, :2]


def _mint_joints(metadata: Mapping[str, Any]) -> np.ndarray | None:
    meta = metadata.get("meta", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(meta, Mapping):
        return None
    return _joints(meta.get("joints_3d_camera"))


def _mint_keypoints_2d(metadata: Mapping[str, Any]) -> np.ndarray | None:
    if not isinstance(metadata, Mapping):
        return None
    return _keypoints_2d(metadata.get("keypoints_2d"))


def _hmr_joints(output: Any) -> np.ndarray | None:
    joints = _joints(getattr(output, "camera_joints_3d", None))
    if joints is not None:
        return joints
    # The 0 -> 9 direction is translation invariant, so root-relative HMR
    # joints are sufficient for this coarse orientation check.
    return _joints(getattr(output, "pred_joints_3d", None))


def _hmr_keypoints_2d(output: Any) -> np.ndarray | None:
    return _keypoints_2d(getattr(output, "pred_keypoints_2d", None))


def orientation_bucket(vector: Any, *, eps: float = 1e-8) -> str | None:
    """Quantize an OpenCV-camera vector into an up/down + front/back/left/right bucket."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= eps:
        return None
    vertical = "up" if v[1] < 0 else "down"
    if abs(v[0]) >= abs(v[2]):
        horizontal = "right" if v[0] >= 0 else "left"
    else:
        horizontal = "front" if v[2] >= 0 else "back"
    return f"{vertical}_{horizontal}"


def projected_2d_direction_bucket(vector: Any, *, eps: float = 1e-8) -> str | None:
    """Quantize an image-plane vector into up/down/left/right."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (2,) or not np.isfinite(v).all() or np.linalg.norm(v) <= eps:
        return None
    if abs(v[0]) >= abs(v[1]):
        return "right" if v[0] >= 0 else "left"
    return "down" if v[1] >= 0 else "up"


def projected_2d_angle_deg(a: Any, b: Any, *, eps: float = 1e-8) -> float | None:
    """Return the angle between two image-plane vectors in degrees."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != (2,) or vb.shape != (2,) or not np.isfinite(va).all() or not np.isfinite(vb).all():
        return None
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a <= eps or norm_b <= eps:
        return None
    cosine = float(np.dot(va, vb) / (norm_a * norm_b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def elevation_deg(vector: Any, *, eps: float = 1e-8) -> float | None:
    """Return elevation above the X-Z plane; positive is up in OpenCV camera coordinates."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= eps:
        return None
    horizontal = float(np.hypot(v[0], v[2]))
    return float(np.degrees(np.arctan2(-v[1], horizontal)))


def front_axis_angle_deg(vector: Any, *, eps: float = 1e-8) -> float | None:
    """Return the X-Z plane angle from +Z toward +/-X in degrees."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= eps:
        return None
    if np.hypot(v[0], v[2]) <= eps:
        return None
    return float(np.degrees(np.arctan2(v[0], v[2])))


def _split_bucket(bucket: str | None) -> tuple[str | None, str | None]:
    if bucket is None or "_" not in bucket:
        return None, None
    vertical, horizontal = bucket.split("_", 1)
    return vertical, horizontal


def _accepted_by_shared_vertical_zone(
    mint_bucket: str,
    hmr_bucket: str,
    mint_elevation: float | None,
    hmr_elevation: float | None,
    half_angle_deg: float,
) -> bool:
    mint_vertical, mint_horizontal = _split_bucket(mint_bucket)
    hmr_vertical, hmr_horizontal = _split_bucket(hmr_bucket)
    if mint_horizontal != hmr_horizontal or mint_vertical == hmr_vertical:
        return False
    if mint_elevation is None or hmr_elevation is None:
        return False
    return abs(mint_elevation) <= half_angle_deg and abs(hmr_elevation) <= half_angle_deg


def _bucket_from_joints(joints: np.ndarray | None) -> tuple[str | None, list[float] | None]:
    joints = _joints(joints)
    if joints is None:
        return None, None
    vector = joints[MIDDLE_MCP] - joints[WRIST]
    bucket = orientation_bucket(vector)
    return bucket, vector.astype(float).tolist() if bucket is not None else None


def _bucket_from_keypoints_2d(keypoints: np.ndarray | None) -> tuple[str | None, list[float] | None]:
    keypoints = _keypoints_2d(keypoints)
    if keypoints is None:
        return None, None
    vector = keypoints[MIDDLE_MCP] - keypoints[WRIST]
    bucket = projected_2d_direction_bucket(vector)
    return bucket, vector.astype(float).tolist() if bucket is not None else None


def _accepted_by_projected_2d_direction(
    mint_projected_bucket: str | None,
    hmr_projected_bucket: str | None,
) -> bool:
    return mint_projected_bucket is not None and mint_projected_bucket == hmr_projected_bucket


def _accepted_by_projected_2d_angle(angle_deg: float | None, max_angle_deg: float) -> bool:
    return angle_deg is not None and angle_deg <= max_angle_deg


def apply_orientation_bucket_gate(outputs: Iterable[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Keep HMR outputs only when their coarse hand direction matches MINT."""
    kept = []
    report = {
        "schema_version": "egohand.orientation_bucket_gate.v1",
        "config": {
            "joint_vector": "openpose21_wrist_to_middle_mcp_0_to_9",
            "camera_convention": "opencv_x_right_y_down_z_forward",
            "vertical_shared_half_angle_deg": VERTICAL_SHARED_HALF_ANGLE_DEG,
            "projected_2d_joint_vector": "openpose21_wrist_to_middle_mcp_0_to_9",
            "projected_2d_buckets": ["up", "down", "left", "right"],
            "projected_2d_angle_match_deg": PROJECTED_2D_ANGLE_MATCH_DEG,
            "buckets": [
                "up_front", "up_back", "up_left", "up_right",
                "down_front", "down_back", "down_left", "down_right",
            ],
        },
        "counts": {"evaluated": 0, "accepted": 0, "rejected": 0, "not_evaluable": 0},
        "frames": [],
    }
    for output in outputs:
        metadata = getattr(output, "raw_backend_meta", {})
        mint_bucket, mint_vector = _bucket_from_joints(_mint_joints(metadata))
        hmr_bucket, hmr_vector = _bucket_from_joints(_hmr_joints(output))
        mint_projected_bucket, mint_projected_vector = _bucket_from_keypoints_2d(_mint_keypoints_2d(metadata))
        hmr_projected_bucket, hmr_projected_vector = _bucket_from_keypoints_2d(_hmr_keypoints_2d(output))
        projected_angle = (
            projected_2d_angle_deg(mint_projected_vector, hmr_projected_vector)
            if mint_projected_vector is not None and hmr_projected_vector is not None
            else None
        )
        mint_elevation = elevation_deg(mint_vector) if mint_vector is not None else None
        hmr_elevation = elevation_deg(hmr_vector) if hmr_vector is not None else None
        mint_front_angle = front_axis_angle_deg(mint_vector) if mint_vector is not None else None
        hmr_front_angle = front_axis_angle_deg(hmr_vector) if hmr_vector is not None else None
        row = {
            "frame_idx": int(getattr(output, "frame_idx")),
            "side": getattr(output, "hand_side", None),
            "physical_track_id": metadata.get("physical_track_id") if isinstance(metadata, Mapping) else None,
            "physical_track_fragment_id": metadata.get("physical_track_fragment_id") if isinstance(metadata, Mapping) else None,
            "mint_bucket": mint_bucket,
            "hmr_bucket": hmr_bucket,
            "mint_vector": mint_vector,
            "hmr_vector": hmr_vector,
            "mint_elevation_deg": mint_elevation,
            "hmr_elevation_deg": hmr_elevation,
            "mint_front_axis_angle_deg": mint_front_angle,
            "hmr_front_axis_angle_deg": hmr_front_angle,
            "mint_projected_2d_bucket": mint_projected_bucket,
            "hmr_projected_2d_bucket": hmr_projected_bucket,
            "mint_projected_2d_vector": mint_projected_vector,
            "hmr_projected_2d_vector": hmr_projected_vector,
            "projected_2d_angle_deg": projected_angle,
        }
        if mint_bucket is None:
            row.update(classification="accepted", not_evaluable_reason="missing_mint_orientation")
            report["counts"]["accepted"] += 1
            report["counts"]["not_evaluable"] += 1
            kept.append(output)
        elif hmr_bucket is None:
            row.update(classification="reject", reject_reason="missing_hmr_orientation")
            report["counts"]["evaluated"] += 1
            report["counts"]["rejected"] += 1
        elif mint_bucket != hmr_bucket and _accepted_by_shared_vertical_zone(
            mint_bucket,
            hmr_bucket,
            mint_elevation,
            hmr_elevation,
            VERTICAL_SHARED_HALF_ANGLE_DEG,
        ):
            row.update(classification="accepted", accept_reason="vertical_shared_zone")
            report["counts"]["evaluated"] += 1
            report["counts"]["accepted"] += 1
            if isinstance(metadata, dict):
                metadata["orientation_bucket_gate"] = {
                    "status": "accepted",
                    "mint_bucket": mint_bucket,
                    "hmr_bucket": hmr_bucket,
                    "accept_reason": "vertical_shared_zone",
                }
            kept.append(output)
        elif mint_bucket != hmr_bucket and _accepted_by_projected_2d_direction(
            mint_projected_bucket,
            hmr_projected_bucket,
        ):
            row.update(classification="accepted", accept_reason="projected_2d_direction_match")
            report["counts"]["evaluated"] += 1
            report["counts"]["accepted"] += 1
            if isinstance(metadata, dict):
                metadata["orientation_bucket_gate"] = {
                    "status": "accepted",
                    "mint_bucket": mint_bucket,
                    "hmr_bucket": hmr_bucket,
                    "accept_reason": "projected_2d_direction_match",
                    "mint_projected_2d_bucket": mint_projected_bucket,
                    "hmr_projected_2d_bucket": hmr_projected_bucket,
                }
            kept.append(output)
        elif mint_bucket != hmr_bucket and _accepted_by_projected_2d_angle(
            projected_angle,
            PROJECTED_2D_ANGLE_MATCH_DEG,
        ):
            row.update(classification="accepted", accept_reason="projected_2d_angle_match")
            report["counts"]["evaluated"] += 1
            report["counts"]["accepted"] += 1
            if isinstance(metadata, dict):
                metadata["orientation_bucket_gate"] = {
                    "status": "accepted",
                    "mint_bucket": mint_bucket,
                    "hmr_bucket": hmr_bucket,
                    "accept_reason": "projected_2d_angle_match",
                    "projected_2d_angle_deg": projected_angle,
                    "projected_2d_angle_match_deg": PROJECTED_2D_ANGLE_MATCH_DEG,
                }
            kept.append(output)
        elif mint_bucket != hmr_bucket:
            row.update(classification="reject", reject_reason="orientation_bucket_mismatch")
            report["counts"]["evaluated"] += 1
            report["counts"]["rejected"] += 1
        else:
            row.update(classification="accepted", reject_reason=None)
            report["counts"]["evaluated"] += 1
            report["counts"]["accepted"] += 1
            if isinstance(metadata, dict):
                metadata["orientation_bucket_gate"] = {
                    "status": "accepted",
                    "mint_bucket": mint_bucket,
                    "hmr_bucket": hmr_bucket,
                }
            kept.append(output)
        report["frames"].append(row)
    return kept, report
