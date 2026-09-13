from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def bbox_metrics(bbox_xyxy: Iterable[float]) -> dict[str, Any]:
    bbox = np.asarray(list(bbox_xyxy), dtype=np.float64).reshape(4)
    x1, y1, x2, y2 = bbox
    width = max(0.0, float(x2 - x1))
    height = max(0.0, float(y2 - y1))
    area = width * height
    return {
        "bbox_xyxy": bbox.astype(float).tolist(),
        "bbox_xywh": [float(x1), float(y1), width, height],
        "bbox_center_xy": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
        "bbox_area_px2": float(area),
        "bbox_aspect_ratio": float(width / height) if height > 0 else None,
    }


def bbox_iou(a_xyxy: Iterable[float], b_xyxy: Iterable[float]) -> float:
    a = np.asarray(list(a_xyxy), dtype=np.float64).reshape(4)
    b = np.asarray(list(b_xyxy), dtype=np.float64).reshape(4)
    intersection_width = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    intersection_height = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def best_previous_bbox_match(
    current: Mapping[str, Any],
    previous: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    bbox = current.get("bbox_xyxy")
    handedness = current.get("handedness")
    if bbox is None:
        return None
    eligible = [
        item
        for item in previous
        if item.get("bbox_xyxy") is not None
        and item.get("handedness") == handedness
        and bool(item.get("official_gate_passed"))
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: bbox_iou(bbox, item["bbox_xyxy"]))


def transition_metrics(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if previous is None or current.get("bbox_xyxy") is None or previous.get("bbox_xyxy") is None:
        return {
            "previous_match_candidate_id": None,
            "bbox_iou_with_prev": None,
            "bbox_center_delta_px": None,
            "bbox_scale_ratio": None,
            "valid_keypoint_mask_change_count": None,
            "keypoints_crossed_threshold_count": None,
        }

    current_geometry = bbox_metrics(current["bbox_xyxy"])
    previous_geometry = bbox_metrics(previous["bbox_xyxy"])
    current_center = np.asarray(current_geometry["bbox_center_xy"], dtype=np.float64)
    previous_center = np.asarray(previous_geometry["bbox_center_xy"], dtype=np.float64)
    previous_area = float(previous_geometry["bbox_area_px2"])
    current_area = float(current_geometry["bbox_area_px2"])
    current_mask = np.asarray(current.get("valid_keypoint_mask", []), dtype=np.bool_)
    previous_mask = np.asarray(previous.get("valid_keypoint_mask", []), dtype=np.bool_)
    mask_change = None
    if current_mask.shape == (21,) and previous_mask.shape == (21,):
        mask_change = int(np.count_nonzero(current_mask != previous_mask))

    threshold = float(current.get("keypoint_threshold", 0.5))
    current_scores = np.asarray(current.get("keypoint_score", []), dtype=np.float64)
    previous_scores = np.asarray(previous.get("keypoint_score", []), dtype=np.float64)
    crossed = None
    if current_scores.shape == (21,) and previous_scores.shape == (21,):
        crossed = int(
            np.count_nonzero((current_scores > threshold) != (previous_scores > threshold))
        )

    return {
        "previous_match_candidate_id": previous.get("candidate_id"),
        "bbox_iou_with_prev": bbox_iou(current["bbox_xyxy"], previous["bbox_xyxy"]),
        "bbox_center_delta_px": float(np.linalg.norm(current_center - previous_center)),
        "bbox_scale_ratio": float(np.sqrt(current_area / previous_area)) if previous_area > 0 else None,
        "valid_keypoint_mask_change_count": mask_change,
        "keypoints_crossed_threshold_count": crossed,
    }


def actual_crop_geometry(
    *,
    box_center: Iterable[float],
    box_size: float,
    image_width: int,
    image_height: int,
    model_input_size: int,
    flipped_for_model: bool,
) -> dict[str, Any]:
    center = np.asarray(list(box_center), dtype=np.float64).reshape(2)
    half = float(box_size) * 0.5
    crop = [
        float(center[0] - half),
        float(center[1] - half),
        float(center[0] + half),
        float(center[1] + half),
    ]
    clipped = crop[0] < 0 or crop[1] < 0 or crop[2] > image_width or crop[3] > image_height
    return {
        "crop_xyxy_original_image": crop,
        "crop_center_xy": center.astype(float).tolist(),
        "crop_box_size_px": float(box_size),
        "crop_model_input_size": [int(model_input_size), int(model_input_size)],
        "crop_clipped_by_image_boundary": bool(clipped),
        "crop_flipped_for_model": bool(flipped_for_model),
    }


def keypoint_statistics(scores: Iterable[float], threshold: float) -> dict[str, Any]:
    values = np.asarray(list(scores), dtype=np.float64).reshape(21)
    valid = values > float(threshold)
    return {
        "keypoint_score": values.astype(float).tolist(),
        "valid_keypoint_mask": valid.astype(bool).tolist(),
        "valid_keypoints": int(valid.sum()),
        "mean_keypoint_score": float(values.mean()),
        "median_keypoint_score": float(np.median(values)),
        "keypoints_within_0p03_of_threshold": int(
            np.count_nonzero(np.abs(values - float(threshold)) <= 0.03)
        ),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
