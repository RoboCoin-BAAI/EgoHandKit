"""ROI-local YOLO verification for MINT observations.

This module deliberately treats YOLO as a veto/verification signal only.  The
MINT bbox remains the bbox sent to HaMeR; detector boxes are used solely to
decide whether that observation is trusted and to maintain a coarse trusted
anchor for each physical hand slot.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _value(value: Any) -> Any:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _detector_candidates(detector: Any, crop_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Run a detector and normalize common Ultralytics/test-double outputs."""
    if hasattr(detector, "predict"):
        try:
            result = detector.predict(crop_rgb, verbose=False)
        except TypeError:
            result = detector.predict(crop_rgb)
    elif callable(detector):
        result = detector(crop_rgb)
    else:
        raise TypeError("YOLO detector must expose predict() or be callable")
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        # A test double may return candidate dictionaries directly, while
        # Ultralytics returns a one-element list of Results.
        if result and isinstance(result[0], dict) and any(
                key in result[0] for key in ("bbox", "bbox_xyxy", "xyxy")):
            return list(result)
        result = result[0] if result else None
    if result is None:
        return []
    if isinstance(result, dict):
        if any(key in result for key in ("bbox", "bbox_xyxy", "xyxy")):
            return [result]
        boxes = result.get("boxes", result.get("candidates", result))
        if isinstance(boxes, dict):
            xyxy = boxes.get("xyxy", boxes.get("bbox"))
            conf = boxes.get("conf", boxes.get("confidence", []))
            classes = boxes.get("cls", boxes.get("class", boxes.get("class_id", [])))
            if xyxy is None:
                return []
            xyxy, conf, classes = np.asarray(xyxy), np.asarray(conf), np.asarray(classes)
            xyxy = np.asarray(xyxy, dtype=np.float32)
            if xyxy.ndim == 1:
                xyxy = xyxy[None]
            return [{"bbox": box,
                     "confidence": float(conf[i]) if len(conf) > i else 0.0,
                     "class": classes[i].item() if len(classes) > i else None}
                    for i, box in enumerate(xyxy)]
        if isinstance(boxes, (list, tuple)):
            return boxes
        if boxes is not None:
            xyxy = np.asarray(boxes, dtype=np.float32)
            if xyxy.ndim == 1:
                xyxy = xyxy[None]
            conf = np.asarray(result.get("conf", result.get("confidence", [])))
            classes = np.asarray(result.get("cls", result.get("class", [])))
            return [{"bbox": box,
                     "confidence": float(conf[i]) if len(conf) > i else 0.0,
                     "class": classes[i].item() if len(classes) > i else None}
                    for i, box in enumerate(xyxy)]
    boxes = getattr(result, "boxes", result)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return []
    xyxy = _value(xyxy)
    conf = _value(getattr(boxes, "conf", np.zeros(len(xyxy))))
    classes = _value(getattr(boxes, "cls", np.full(len(xyxy), -1)))
    return [{"bbox": np.asarray(box, dtype=np.float32),
             "confidence": float(conf[i]) if len(conf) > i else 0.0,
             "class": classes[i].item() if len(classes) > i else None}
            for i, box in enumerate(xyxy)]


def _normalize_candidate(candidate: Any) -> tuple[np.ndarray, float, Any]:
    if isinstance(candidate, dict):
        bbox = candidate.get("bbox", candidate.get("bbox_xyxy", candidate.get("xyxy")))
        confidence = candidate.get("confidence", candidate.get("conf", candidate.get("score", 0.0)))
        label = candidate.get("class", candidate.get("cls", candidate.get("class_id")))
    else:
        bbox = getattr(candidate, "bbox", getattr(candidate, "xyxy", candidate))
        confidence = getattr(candidate, "confidence", getattr(candidate, "conf", 0.0))
        label = getattr(candidate, "class", getattr(candidate, "cls", None))
    bbox = np.asarray(_value(bbox), dtype=np.float32).reshape(-1)
    if bbox.size != 4 or not np.isfinite(bbox).all():
        raise ValueError("YOLO candidate bbox must contain four finite coordinates")
    return bbox, float(np.asarray(_value(confidence)).reshape(-1)[0]), label.item() if hasattr(label, "item") else label


def _roi(bbox: np.ndarray, image_shape: tuple[int, int], scale: float) -> np.ndarray:
    h, w = image_shape
    center = (bbox[:2] + bbox[2:]) / 2.0
    size = np.maximum(bbox[2:] - bbox[:2], 1.0) * scale
    result = np.array([center[0] - size[0] / 2, center[1] - size[1] / 2,
                       center[0] + size[0] / 2, center[1] + size[1] / 2], dtype=np.float32)
    result[[0, 2]] = np.clip(result[[0, 2]], 0, max(w - 1, 0))
    result[[1, 3]] = np.clip(result[[1, 3]], 0, max(h - 1, 0))
    return result


def apply_yolo_gate(
    frames: list[dict[str, Any]], image_paths: list[str], detector: Any, *,
    roi_scale: float = 1.5, hard_conf_threshold: float = 0.20,
    strong_conf_threshold: float = 0.25, strong_iou_threshold: float = 0.35,
    hard_iou_threshold: float = 0.10, anchor_center_threshold: float = 0.15,
    gray_grace_frames: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify each MINT observation in a local ROI and remove vetoed samples."""
    if len(frames) != len(image_paths):
        raise ValueError("YOLO gate frame count does not match image sequence")
    if roi_scale <= 0 or not 0 <= hard_conf_threshold <= strong_conf_threshold:
        raise ValueError("invalid YOLO confidence thresholds")
    if not 0 <= hard_iou_threshold <= strong_iou_threshold <= 1:
        raise ValueError("invalid YOLO IoU thresholds")
    if anchor_center_threshold <= 0 or gray_grace_frames < 0:
        raise ValueError("anchor threshold must be positive and gray grace non-negative")

    states = {side: {"anchor": None, "gray_run": 0, "missing_run": 0,
                      "fragment": None, "interrupted": False}
              for side in ("left", "right")}
    gated, rows = [], []
    counts = {"accepted": 0, "rejected": 0}
    for frame, image_path in zip(frames, image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image for YOLO gate: {image_path}")
        h, w = image.shape[:2]
        diagonal = float(np.hypot(h, w))
        kept = []
        frame_rows = {"frame_idx": int(frame["frame_idx"]), "hands": {}}
        for observation in frame.get("selected_for_hamer", []):
            side = observation["handedness"]
            state = states[side]
            incoming_fragment = int(observation.get("physical_track_fragment_id", 0))
            if state["fragment"] is None:
                state["fragment"] = incoming_fragment
            else:
                # Preserve fragment boundaries already created by depth (or a
                # prior frontend gate); this gate only ever increments them.
                state["fragment"] = max(int(state["fragment"]), incoming_fragment)
            mint_bbox = np.asarray(observation["bbox_xyxy"], dtype=np.float32)
            roi = _roi(mint_bbox, (h, w), roi_scale)
            # cv2 reads BGR; verification is explicitly performed on RGB crops.
            x1, y1 = int(np.floor(roi[0])), int(np.floor(roi[1]))
            x2, y2 = int(np.ceil(roi[2])) + 1, int(np.ceil(roi[3])) + 1
            crop_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            candidates = []
            for raw in _detector_candidates(detector, crop_rgb):
                bbox_local, confidence, label = _normalize_candidate(raw)
                bbox_full = bbox_local + np.array([x1, y1, x1, y1], dtype=np.float32)
                candidates.append((bbox_full, confidence, label))
            candidates.sort(key=lambda item: (_iou(mint_bbox, item[0]), item[1]), reverse=True)
            best_bbox, best_conf, best_class = (None, 0.0, None)
            best_iou = 0.0
            if candidates:
                best_bbox, best_conf, best_class = candidates[0]
                best_iou = _iou(mint_bbox, best_bbox)
            anchor = state["anchor"]
            anchor_distance = None
            if anchor is not None and best_bbox is not None:
                anchor_center = (anchor[:2] + anchor[2:]) / 2.0
                best_center = (best_bbox[:2] + best_bbox[2:]) / 2.0
                anchor_distance = float(np.linalg.norm(best_center - anchor_center) / max(diagonal, 1e-6))

            reasons = []
            classification = "strong_match"
            strong = (best_iou >= strong_iou_threshold and best_conf >= strong_conf_threshold) or (
                best_iou >= 0.65 and best_conf >= hard_conf_threshold)
            if not candidates:
                state["missing_run"] += 1
                classification = "no_yolo_candidate"
                # A single detector miss is kept as a pending observation. It
                # does not update the anchor; a consecutive miss is a veto.
                if state["missing_run"] > 1:
                    reasons.append("no_yolo_candidate")
            elif best_conf < hard_conf_threshold:
                state["missing_run"] = 0
                reasons.append("low_confidence"); classification = "low_confidence"
            elif best_iou < hard_iou_threshold:
                state["missing_run"] = 0
                reasons.append("low_iou"); classification = "low_iou"
            elif strong:
                state["missing_run"] = 0
                state["gray_run"] = 0
                if state["interrupted"]:
                    classification = "reacquired_new_fragment"
                state["anchor"] = best_bbox.copy()
                state["interrupted"] = False
            else:
                state["missing_run"] = 0
                classification = "gray_grace"
                if anchor is None or anchor_distance is None or anchor_distance > anchor_center_threshold:
                    reasons.append("anchor_drift"); classification = "anchor_drift"
                elif state["gray_run"] >= gray_grace_frames:
                    reasons.append("gray_timeout"); classification = "gray_timeout"
                else:
                    state["gray_run"] += 1

            valid = not reasons
            if not valid:
                state["gray_run"] = 0
                if not state["interrupted"]:
                    state["fragment"] += 1
                state["interrupted"] = True
                counts["rejected"] += 1
            else:
                updated = dict(observation)
                updated["physical_track_fragment_id"] = int(state["fragment"])
                meta = dict(updated.get("observation_meta", {}))
                meta.update({"yolo_gate": classification, "yolo_best_iou": float(best_iou),
                             "yolo_best_confidence": float(best_conf)})
                updated["observation_meta"] = meta
                kept.append(updated)
                counts["accepted"] += 1
            diagnostic_anchor = state["anchor"] if state["anchor"] is not None else anchor
            frame_rows["hands"][side] = {
                "valid": bool(valid), "classification": classification,
                "mint_bbox_xyxy": mint_bbox.tolist(), "roi_xyxy": roi.tolist(),
                "candidate_count": len(candidates),
                "best_yolo_bbox_xyxy": None if best_bbox is None else best_bbox.tolist(),
                "best_yolo_class": best_class, "best_iou": float(best_iou),
                "best_confidence": float(best_conf),
                "anchor_bbox_xyxy": None if diagnostic_anchor is None else diagnostic_anchor.tolist(),
                "anchor_center_distance": anchor_distance,
                "gray_run_length": int(state["gray_run"]),
                "fragment_id": int(state["fragment"]), "reject_reasons": reasons,
            }
        updated_frame = dict(frame)
        updated_frame["selected_for_hamer"] = kept
        gated.append(updated_frame)
        rows.append(frame_rows)
    return gated, {"schema_version": "mint_yolo_gate.v1", "roi_scale": float(roi_scale),
                   "frame_count": len(frames), "counts": counts,
                   "rejected_observations": counts["rejected"], "frames": rows}
