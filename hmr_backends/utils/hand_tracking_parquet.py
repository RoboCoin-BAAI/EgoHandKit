"""Columnar export for camera-space left/right hand tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hmr_backends.utils.mint_3d_consistency import convert_mint_to_camera_joints


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


def _mint_hands(frame: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, float | None, str]]:
    hands = {}
    for observation in frame.get("hands", []):
        side = observation.get("handedness")
        joints = convert_mint_to_camera_joints(observation)
        if side not in {"left", "right"} or joints is None:
            continue
        confidence = observation.get("confidence")
        confidence = float(confidence) if confidence is not None else None
        hands[side] = (joints.astype(np.float32), confidence, str(observation.get("source", "mint")))
    return hands


def export_hand_tracking_parquet(
    frames: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Export accepted HMR joints, falling back to an available MINT prior."""
    rows = []
    for frame in frames:
        mint = _mint_hands(frame)
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
