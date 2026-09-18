"""Diagnostic comparison between canonical observations and legacy YOLO boxes."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def compare_observations_yolo(
    observation_frames: Sequence[Mapping[str, Any]],
    yolo_frames: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare any canonical observation source to YOLO without changing it."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in [0,1]")
    if len(observation_frames) != len(yolo_frames):
        raise ValueError("observation and YOLO validation frame counts differ")

    rows = []
    counts = {"match": 0, "low_iou": 0, "observation_missing": 0,
              "yolo_missing": 0, "both_missing": 0}
    for observation_frame, yolo_frame in zip(observation_frames, yolo_frames):
        hands = observation_frame.get("hands", observation_frame.get("selected_for_hamer", []))
        by_side = {item["handedness"]: item for item in hands}
        frame_rows = {"frame_idx": int(observation_frame["frame_idx"]), "sides": {}}
        for side in ("left", "right"):
            observation = by_side.get(side)
            observation_bbox = observation.get("bbox_xyxy") if observation else None
            yolo_bbox = yolo_frame.get(f"{side}_bbox")
            if observation_bbox is None and yolo_bbox is None:
                status, iou = "both_missing", None
            elif observation_bbox is None:
                status, iou = "observation_missing", 0.0
            elif yolo_bbox is None:
                status, iou = "yolo_missing", 0.0
            else:
                iou = _bbox_iou(np.asarray(observation_bbox), np.asarray(yolo_bbox))
                status = "match" if iou >= iou_threshold else "low_iou"
            counts[status] += 1
            frame_rows["sides"][side] = {
                "status": status,
                "iou": iou,
                "observation_bbox_xyxy": None if observation_bbox is None else [float(v) for v in observation_bbox],
                "yolo_bbox_xyxy": None if yolo_bbox is None else [float(v) for v in yolo_bbox],
                "yolo_confidence": None if yolo_frame.get(f"{side}_conf") is None else float(yolo_frame[f"{side}_conf"]),
            }
        rows.append(frame_rows)
    return {
        "schema_version": "egohand.yolo_check.v1",
        "source": "observation_yolo_diagnostic",
        "decision_effect": "none",
        "iou_threshold": float(iou_threshold),
        "frame_count": len(observation_frames),
        "side_count": sum(counts.values()),
        "counts": counts,
        "frames": rows,
    }
