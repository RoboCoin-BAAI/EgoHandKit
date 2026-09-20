"""Depth hard-gate for pre-HaMeR MINT observations.

The gate is intentionally conservative: it rejects implausible observations
before HaMeR and before any temporal MANO smoother can absorb a bad sample.
Depth is used as a veto, never as the hand selector.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from observation_frontend.schema import camera_joints_from_observation


def resolve_depth_frames(depth_dir: str | Path, frame_count: int) -> list[Path]:
    root = Path(depth_dir)
    candidates = [
        root / "fast_foundation" / "depth_uint16_png",
        root / "depth_uint16_png",
        root / "fast_foundation_archive" / "depth_uint16_png",
    ]
    png_dir = next((path for path in candidates if path.is_dir()), None)
    if png_dir is None:
        raise FileNotFoundError(
            f"Depth PNG directory not found under {root}; expected fast_foundation/depth_uint16_png"
        )
    paths = [png_dir / f"frame_{index:06d}_depth_mm.png" for index in range(frame_count)]
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Depth sequence is missing frame: {missing}")
    return paths


def _depth_for_bbox(path: Path, bbox: Any, image_shape: tuple[int, int], *, unit_scale: float) -> float | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Cannot read depth frame: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be single-channel: {path}")
    h, w = image_shape
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(w, int(np.ceil(x2)) + 1), min(h, int(np.ceil(y2)) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    values = depth[y1:y2, x1:x2].astype(np.float32) * float(unit_scale)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    # Median is robust to object/background pixels at the bbox boundary.
    return float(np.median(values))


def _depth_for_point(path: Path, point: Any, image_shape: tuple[int, int], *, unit_scale: float) -> float | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Cannot read depth frame: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be single-channel: {path}")
    h, w = image_shape
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    xy = np.asarray(point, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return None
    x, y = int(round(xy[0])), int(round(xy[1]))
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    values = depth[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
    values = values.astype(np.float32) * float(unit_scale)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


def _mint_wrist_depth(observation: dict[str, Any]) -> float | None:
    joints = camera_joints_from_observation(observation)
    return float(joints[0, 2]) if joints is not None else None


def apply_depth_gate(
    frames: list[dict[str, Any]],
    image_paths: list[str | Path],
    depth_dir: str | Path,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 4.0,
    max_ratio: float = 2.5,
    min_ratio: float = 0.4,
    history_size: int = 10,
    max_bad_frames: int = 2,
    unit_scale: float = 0.001,
    wrist_only: bool = False,
    wrist_threshold_m: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject depth outliers and restart fragments after invalid observations."""
    if not (0 < min_depth_m < max_depth_m):
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
    if max_ratio <= 1 or not 0 < min_ratio < 1:
        raise ValueError("max_ratio must be > 1 and min_ratio must lie in (0,1)")
    if history_size < 1 or max_bad_frames < 1:
        raise ValueError("history_size and max_bad_frames must be positive")
    if wrist_threshold_m <= 0:
        raise ValueError("wrist_threshold_m must be positive")
    depth_paths = resolve_depth_frames(depth_dir, len(frames))
    output = []
    diagnostics = []
    state = {}
    for frame_index, (frame, image_path, depth_path) in enumerate(zip(frames, image_paths, depth_paths)):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image for depth gate: {image_path}")
        kept = []
        frame_diag = {"frame_idx": int(frame["frame_idx"]), "hands": {}}
        for observation in frame["hands"]:
            side = observation["handedness"]
            track = int(observation["physical_track_id"])
            source_fragment = int(observation["physical_track_fragment_id"])
            hand_state = state.setdefault(
                (track, side),
                {"history": deque(maxlen=history_size), "bad_run": 0,
                 "fragment": source_fragment, "source_fragment": source_fragment, "lost": False},
            )
            if source_fragment != hand_state["source_fragment"]:
                hand_state.update(fragment=source_fragment, source_fragment=source_fragment,
                                  bad_run=0, lost=False)
                hand_state["history"].clear()
            diagnostic_key = side if side not in frame_diag["hands"] else f"{side}:{track}"
            mint_wrist_depth_m = None
            if wrist_only:
                keypoints = np.asarray(observation["keypoints_2d"], dtype=np.float64)
                wrist_projection_valid = (
                    keypoints.shape == (21, 3)
                    and np.isfinite(keypoints[0]).all()
                    and keypoints[0, 2] > 0
                )
                depth_m = (
                    _depth_for_point(
                        depth_path, keypoints[0, :2], image.shape[:2], unit_scale=unit_scale
                    )
                    if wrist_projection_valid else None
                )
                mint_wrist_depth_m = _mint_wrist_depth(observation)
            else:
                depth_m = _depth_for_bbox(
                    depth_path, observation["bbox_xyxy"], image.shape[:2], unit_scale=unit_scale
                )
            reference = (
                float(np.median(hand_state["history"]))
                if hand_state["history"] and not wrist_only else None
            )
            reasons = []
            not_evaluated_reason = (
                "missing_mint_wrist_depth"
                if wrist_only and mint_wrist_depth_m is None else None
            )
            if not_evaluated_reason is not None:
                pass
            elif depth_m is None:
                reasons.append("no_valid_depth")
            elif wrist_only and abs(mint_wrist_depth_m - depth_m) >= wrist_threshold_m:
                reasons.append("wrist_depth_mismatch")
            elif not wrist_only and (depth_m < min_depth_m or depth_m > max_depth_m):
                reasons.append("absolute_depth_limit")
            elif not wrist_only and reference is not None and (depth_m / reference > max_ratio or depth_m / reference < min_ratio):
                reasons.append("relative_depth_jump")
            valid = not reasons
            if valid:
                if hand_state["lost"]:
                    hand_state["lost"] = False
                if depth_m is not None:
                    hand_state["history"].append(depth_m)
                hand_state["bad_run"] = 0
                updated = dict(observation)
                updated["physical_track_fragment_id"] = int(hand_state["fragment"])
                meta = dict(updated.get("meta", {}))
                meta.update({"depth_gate": "pass", "depth_m": depth_m, "depth_reference_m": reference})
                if wrist_only and not_evaluated_reason is None:
                    meta.update({"depth_joint_used": "wrist", "mint_wrist_depth_m": mint_wrist_depth_m,
                                 "depth_difference_m": abs(mint_wrist_depth_m - depth_m)})
                elif wrist_only:
                    meta.update({"depth_joint_used": "wrist",
                                 "depth_gate_not_evaluated": not_evaluated_reason})
                updated["meta"] = meta
                kept.append(updated)
            else:
                hand_state["bad_run"] += 1
                # Any rejected frame starts a new temporal segment. This is
                # the important anti-bridge rule: a later reappearance must
                # not be smoothed back to the pre-outlier segment.
                if not hand_state["lost"]:
                    hand_state["fragment"] += 1
                    hand_state["lost"] = True
                hand_state["history"].clear()
                frame_diag["hands"][diagnostic_key] = {
                    "valid": False, "depth_m": depth_m, "reference_depth_m": reference,
                    "reasons": reasons, "fragment_id": int(hand_state["fragment"]),
                    "depth_joint_used": "wrist" if wrist_only else "bbox",
                    "mint_wrist_depth_m": mint_wrist_depth_m,
                    "not_evaluated_reason": not_evaluated_reason,
                }
                continue
            frame_diag["hands"][diagnostic_key] = {
                "valid": True, "depth_m": depth_m, "reference_depth_m": reference,
                "reasons": [], "fragment_id": int(hand_state["fragment"]),
                "depth_joint_used": "wrist" if wrist_only else "bbox",
                "mint_wrist_depth_m": mint_wrist_depth_m,
                "not_evaluated_reason": not_evaluated_reason,
            }
        output.append({**frame, "hands": kept, "depth_gate": frame_diag["hands"]})
        diagnostics.append(frame_diag)
    summary = {
        "schema_version": "depth_gate.v1", "depth_dir": str(Path(depth_dir).resolve()),
        "frame_count": len(frames), "min_depth_m": min_depth_m, "max_depth_m": max_depth_m,
        "max_ratio": max_ratio, "min_ratio": min_ratio, "history_size": history_size,
        "max_bad_frames": max_bad_frames, "unit_scale": unit_scale,
        "wrist_only": bool(wrist_only), "wrist_threshold_m": wrist_threshold_m,
        "depth_joint_used": "wrist" if wrist_only else "bbox", "frames": diagnostics,
        "rejected_observations": sum(sum(not item["valid"] for item in row["hands"].values()) for row in diagnostics),
    }
    return output, summary
