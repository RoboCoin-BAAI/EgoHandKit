"""Columnar export for camera-space left/right hand tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hmr_backends.utils.mint_3d_consistency import convert_mint_to_camera_joints
from observation_frontend.depth_gate import (
    depth_for_image_point,
    resolve_depth_camera_intrinsics,
    resolve_depth_frames,
    scale_camera_intrinsics,
)


JOINTS_TYPE = pa.list_(pa.list_(pa.float32(), 3), 21)
MISSING_JOINTS = np.full((21, 3), np.nan, dtype=np.float32)
HAND_TRACKING_SCHEMA = pa.schema([
    pa.field("frame_idx", pa.int64(), nullable=False),
    pa.field("timestamp_ns", pa.int64()),
    pa.field("left_present", pa.bool_(), nullable=False),
    pa.field("right_present", pa.bool_(), nullable=False),
    pa.field("left_joints_3d", JOINTS_TYPE),
    pa.field("right_joints_3d", JOINTS_TYPE),
    pa.field("left_confidence", pa.float32()),
    pa.field("right_confidence", pa.float32()),
    pa.field("source", pa.string(), nullable=False),
], metadata={b"coordinate_system": b"opencv_x_right_y_down_z_forward",
             b"joint_order": b"openpose21"})


def _valid_joints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return array


def _hmr_hands(result: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, float | None, str]]:
    hands = {}
    joints_values = result.get("joints_3d", [])
    metadata_values = result.get("backend_meta", [])
    for joints_value, metadata in zip(joints_values, metadata_values):
        side = metadata.get("handedness", metadata.get("backend_handedness"))
        joints = _valid_joints(joints_value)
        if side not in {"left", "right"} or joints is None:
            continue
        confidence = metadata.get("confidence")
        confidence = float(confidence) if confidence is not None else None
        hands[side] = (joints, confidence, str(metadata.get("backend", "hmr")))
    return hands


def _mint_hands(
    frame: Mapping[str, Any],
    depth_path: Path | None = None,
    camera_intrinsics: Any = None,
) -> dict[str, tuple[np.ndarray, float | None, str]]:
    hands = {}
    image_shape = None
    if depth_path is not None:
        image = cv2.imread(str(frame["img_path"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image for Parquet depth anchoring: {frame['img_path']}")
        image_shape = image.shape[:2]
    for observation in frame.get("hands", []):
        side = observation.get("handedness")
        wrist_depth = None
        if depth_path is not None:
            keypoints = np.asarray(observation.get("keypoints_2d"), dtype=np.float64)
            if (keypoints.shape == (21, 3) and np.isfinite(keypoints[0]).all()
                    and keypoints[0, 2] > 0):
                wrist_depth = depth_for_image_point(
                    depth_path, keypoints[0, :2], image_shape
                )
        if wrist_depth is not None:
            joints = convert_mint_to_camera_joints(
                observation,
                wrist_depth_m=wrist_depth,
                camera_intrinsics=camera_intrinsics,
            )
        elif depth_path is not None:
            joints = None
        else:
            joints = convert_mint_to_camera_joints(observation)
        if side not in {"left", "right"} or joints is None:
            continue
        confidence = observation.get("confidence")
        confidence = float(confidence) if confidence is not None else None
        hands[side] = (
            joints.astype(np.float32), confidence,
            str(observation.get("source", "mint")),
        )
    return hands


def export_hand_tracking_parquet(
    frames: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    *,
    depth_dir: str | Path | None = None,
    expected_reference_camera: str | None = None,
) -> Path:
    """Export accepted HMR joints, falling back to an available MINT prior."""
    rows = []
    depth_paths = resolve_depth_frames(depth_dir, len(frames)) if depth_dir is not None else None
    sensor_intrinsics = (
        resolve_depth_camera_intrinsics(
            depth_dir, expected_reference_camera=expected_reference_camera
        )[0]
        if depth_dir is not None else None
    )
    for frame in frames:
        depth_path = depth_paths[int(frame["frame_idx"])] if depth_paths is not None else None
        frame_intrinsics = sensor_intrinsics
        if frame_intrinsics is not None:
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            image = cv2.imread(str(frame["img_path"]), cv2.IMREAD_UNCHANGED)
            if depth_image is None or image is None:
                raise ValueError(f"Cannot read RGB/depth frame {frame['frame_idx']} for Parquet export")
            frame_intrinsics = scale_camera_intrinsics(
                frame_intrinsics, depth_image.shape[:2], image.shape[:2]
            )
        mint = _mint_hands(frame, depth_path, frame_intrinsics)
        selected = dict(mint)
        selected.update(_hmr_hands(results.get(frame["img_path"], {})))
        sources = {selected[side][2] for side in ("left", "right") if side in selected}
        source = next(iter(sources)) if len(sources) == 1 else "mixed" if sources else "none"
        row: dict[str, Any] = {
            "frame_idx": int(frame["frame_idx"]),
            "timestamp_ns": int(frame["timestamp_ns"]) if frame.get("timestamp_ns") is not None else None,
            "source": source,
        }
        for side in ("left", "right"):
            value = selected.get(side)
            row[f"{side}_present"] = value is not None
            # Parquet cannot encode a null fixed-size list reliably. Presence
            # is authoritative; NaNs keep the physical shape without
            # masquerading as valid camera coordinates.
            row[f"{side}_joints_3d"] = (
                value[0].tolist() if value is not None else MISSING_JOINTS.tolist()
            )
            row[f"{side}_confidence"] = value[1] if value is not None else None
        rows.append(row)

    columns = [pa.array([row[field.name] for row in rows], type=field.type) for field in HAND_TRACKING_SCHEMA]
    table = pa.Table.from_arrays(columns, schema=HAND_TRACKING_SCHEMA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return path
