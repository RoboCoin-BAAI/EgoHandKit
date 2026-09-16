"""Pre-HaMeR spatial outlier gate for MINT observations.

Depth can be valid for the wrong hand.  This gate therefore checks image-space
motion before HaMeR and starts a new temporal fragment after a jump, so the
post-HMR smoother cannot pull the track through the bad sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bbox_metrics(previous: dict[str, Any], current: dict[str, Any], diagonal: float) -> dict[str, float]:
    prev_box = np.asarray(previous["bbox_xyxy"], dtype=np.float32)
    curr_box = np.asarray(current["bbox_xyxy"], dtype=np.float32)
    prev_center = (prev_box[:2] + prev_box[2:]) / 2.0
    curr_center = (curr_box[:2] + curr_box[2:]) / 2.0
    prev_size = np.maximum(prev_box[2:] - prev_box[:2], 1e-6)
    curr_size = np.maximum(curr_box[2:] - curr_box[:2], 1e-6)
    prev_area = float(np.prod(prev_size))
    curr_area = float(np.prod(curr_size))
    ix1, iy1 = np.maximum(prev_box[:2], curr_box[:2])
    ix2, iy2 = np.minimum(prev_box[2:], curr_box[2:])
    intersection = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
    iou = intersection / max(prev_area + curr_area - intersection, 1e-6)
    metrics = {
        "center_jump": float(np.linalg.norm(curr_center - prev_center) / max(diagonal, 1e-6)),
        "size_ratio": float(max(prev_area, curr_area) / max(min(prev_area, curr_area), 1e-6)),
        "iou": float(iou),
    }
    previous_points = np.asarray(previous["keypoints_2d"], dtype=np.float32)
    current_points = np.asarray(current["keypoints_2d"], dtype=np.float32)
    if previous_points.ndim == 2 and current_points.shape == previous_points.shape and previous_points.shape[1] >= 2:
        valid = np.isfinite(previous_points[:, :2]).all(axis=1) & np.isfinite(current_points[:, :2]).all(axis=1)
        if valid.any():
            metrics["joint_jump"] = float(
                np.median(np.linalg.norm(current_points[valid, :2] - previous_points[valid, :2], axis=1))
                / max(diagonal, 1e-6)
            )
    return metrics


def _is_stable(metrics: dict[str, float], *, center_threshold: float, size_ratio_threshold: float,
                iou_threshold: float, joint_threshold: float) -> bool:
    return (
        metrics["center_jump"] <= center_threshold
        and metrics["size_ratio"] <= size_ratio_threshold
        and metrics["iou"] >= iou_threshold
        and metrics.get("joint_jump", 0.0) <= joint_threshold
    )


def apply_motion_gate(
    frames: list[dict[str, Any]],
    image_paths: list[str | Path],
    *,
    center_jump_threshold: float = 0.25,
    size_ratio_threshold: float = 2.0,
    iou_threshold: float = 0.1,
    joint_jump_threshold: float = 0.25,
    min_bad_votes: int = 2,
    reacquire_frames: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject image-space jumps and split the hand track before HaMeR.

    A jump is a candidate outlier when at least ``min_bad_votes`` of center,
    size, IoU, and projected-joint tests fail.  The next observations are used
    as a small offline reacquisition lookahead for diagnostics; regardless of
    whether the jump is isolated or persistent, the first post-jump accepted
    sample starts a new fragment.  This conservative rule avoids smoothing a
    real track switch from the old fragment into the new one.
    """
    if len(frames) != len(image_paths):
        raise ValueError("motion gate frame count does not match image sequence")
    if not (0 < center_jump_threshold and size_ratio_threshold >= 1 and 0 <= iou_threshold <= 1):
        raise ValueError("invalid motion gate thresholds")
    if not (0 < joint_jump_threshold and 1 <= min_bad_votes <= 4 and reacquire_frames >= 1):
        raise ValueError("invalid motion gate vote/reacquisition settings")

    diagonals = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image for motion gate: {image_path}")
        diagonals.append(float(np.hypot(image.shape[0], image.shape[1])))

    candidates = {}
    for frame_position, frame in enumerate(frames):
        for observation in frame["hands"]:
            side = observation["handedness"]
            track = int(observation["physical_track_id"])
            candidates.setdefault((track, side), []).append((frame_position, observation))

    accepted: dict[tuple[int, int, str], dict[str, Any]] = {}
    diagnostics = [{"frame_idx": int(frame["frame_idx"]), "hands": {}} for frame in frames]
    summary_counts = {"accepted": 0, "rejected": 0, "isolated_or_return": 0, "reacquired": 0}

    for (track, side), side_candidates in candidates.items():
        side_candidates.sort(key=lambda item: item[0])
        if not side_candidates:
            continue
        max_fragment = max(int(obs.get("physical_track_fragment_id", 0)) for _, obs in side_candidates)
        previous: tuple[int, dict[str, Any]] | None = None
        break_pending = False
        for candidate_index, (position, observation) in enumerate(side_candidates):
            metrics = None
            classification = "accepted"
            if previous is not None and position == previous[0] + 1 and not break_pending:
                metrics = _bbox_metrics(previous[1], observation, diagonals[position])
                failed = [
                    metrics["center_jump"] > center_jump_threshold,
                    metrics["size_ratio"] > size_ratio_threshold,
                    metrics["iou"] < iou_threshold,
                    metrics.get("joint_jump", 0.0) > joint_jump_threshold,
                ]
                bad_votes = sum(failed)
                if bad_votes >= min_bad_votes:
                    classification = "spatial_jump"
                    next_candidates = side_candidates[candidate_index + 1: candidate_index + 1 + reacquire_frames]
                    consecutive = bool(next_candidates) and all(
                        item[0] == position + offset + 1
                        for offset, item in enumerate(next_candidates)
                    )
                    if consecutive:
                        stable_pairs = []
                        reference = observation
                        for next_position, next_observation in next_candidates:
                            pair_metrics = _bbox_metrics(reference, next_observation, diagonals[next_position])
                            stable_pairs.append(_is_stable(
                                pair_metrics,
                                center_threshold=center_jump_threshold * 0.5,
                                size_ratio_threshold=min(size_ratio_threshold, 1.5),
                                iou_threshold=max(iou_threshold, 0.3),
                                joint_threshold=joint_jump_threshold * 0.6,
                            ))
                            reference = next_observation
                        if all(stable_pairs):
                            classification = "reacquire_new_segment"
                            summary_counts["reacquired"] += 1
                    if next_candidates and _is_stable(
                        _bbox_metrics(previous[1], next_candidates[0][1], diagonals[next_candidates[0][0]]),
                        center_threshold=center_jump_threshold,
                        size_ratio_threshold=size_ratio_threshold,
                        iou_threshold=iou_threshold,
                        joint_threshold=joint_jump_threshold,
                    ):
                        classification = "isolated_or_return"
                        summary_counts["isolated_or_return"] += 1
                    summary_counts["rejected"] += 1
                    diagnostic_key = side if side not in diagnostics[position]["hands"] else f"{side}:{track}"
                    diagnostics[position]["hands"][diagnostic_key] = {
                        "valid": False,
                        "classification": classification,
                        "bad_votes": int(bad_votes),
                        "metrics": metrics,
                    }
                    break_pending = True
                    previous = None
                    continue

            if break_pending:
                max_fragment += 1
                break_pending = False
                classification = "new_fragment"
            updated = dict(observation)
            updated["physical_track_fragment_id"] = int(max_fragment)
            meta = dict(updated.get("meta", {}))
            meta.update({"motion_gate": "pass", "motion_gate_metrics": metrics})
            updated["meta"] = meta
            accepted[(position, track, side)] = updated
            previous = (position, updated)
            summary_counts["accepted"] += 1
            diagnostic_key = side if side not in diagnostics[position]["hands"] else f"{side}:{track}"
            diagnostics[position]["hands"][diagnostic_key] = {
                "valid": True,
                "classification": classification,
                "metrics": metrics,
                "fragment_id": int(max_fragment),
            }

    output = []
    for position, frame in enumerate(frames):
        selected = [
            accepted[(position, int(observation["physical_track_id"]), observation["handedness"])]
            for observation in frame["hands"]
            if (position, int(observation["physical_track_id"]), observation["handedness"]) in accepted
        ]
        output.append({**frame, "hands": selected, "motion_gate": diagnostics[position]["hands"]})
    report = {
        "schema_version": "motion_gate.v1",
        "frame_count": len(frames),
        "center_jump_threshold": center_jump_threshold,
        "size_ratio_threshold": size_ratio_threshold,
        "iou_threshold": iou_threshold,
        "joint_jump_threshold": joint_jump_threshold,
        "min_bad_votes": min_bad_votes,
        "reacquire_frames": reacquire_frames,
        "counts": summary_counts,
        "frames": diagnostics,
    }
    return output, report
