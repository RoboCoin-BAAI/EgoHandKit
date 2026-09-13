from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from observation_frontend.failure_diagnostics import bbox_iou, bbox_metrics
from observation_frontend.hand_observation_consolidation import (
    PALM_KEYPOINT_INDICES,
)

ObservationKey = tuple[int, int]
TrackMemory = tuple[ObservationKey | None, str | None, ObservationKey | None]
StateMemory = tuple[ObservationKey | None, str | None]
AssociationState = tuple[StateMemory, StateMemory]


@dataclass(frozen=True)
class PhysicalHandTemporalAssociationConfig:
    max_tracks: int = 2
    max_gap_frames: int = 2
    keypoint_threshold: float = 0.5
    min_common_keypoints: int = 4
    min_common_palm_keypoints: int = 2
    center_distance_weight: float = 0.80
    bbox_iou_weight: float = 0.35
    bbox_scale_weight: float = 0.15
    keypoint_distance_weight: float = 0.65
    palm_distance_weight: float = 0.25
    # Kept opt-in until sequence-level tuning confirms the motion prior does
    # not overrule a genuine crossing/occlusion event.
    motion_prediction_weight: float = 0.0
    handedness_flip_penalty: float = 0.18
    observation_quality_weight: float = 0.25
    gap_connection_penalty: float = 0.15
    missing_frame_penalty: float = 0.60
    initial_track_start_penalty: float = 0.90
    track_restart_penalty: float = 0.95
    unassigned_base_penalty: float = 0.05
    unassigned_quality_weight: float = 2.80
    identity_switch_cost_threshold: float = 1.60
    isolated_connection_cost_threshold: float = 1.20
    canonical_start_order_penalty: float = 1e-6

    def __post_init__(self) -> None:
        if self.max_tracks != 2:
            raise ValueError("the baseline currently supports exactly two track slots")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative")
        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be in [0, 1]")
        if self.min_common_keypoints < 1 or self.min_common_palm_keypoints < 1:
            raise ValueError("minimum common-keypoint counts must be positive")
        numeric = asdict(self)
        for name, value in numeric.items():
            if name in {
                "max_tracks",
                "max_gap_frames",
                "min_common_keypoints",
                "min_common_palm_keypoints",
            }:
                continue
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = PhysicalHandTemporalAssociationConfig()


@dataclass
class _DpNode:
    cost: float
    previous_state: AssociationState | None
    assignments: tuple[int | None, int | None]
    track_steps: tuple[dict[str, Any], dict[str, Any]]
    unassigned_indices: tuple[int, ...]
    memories: tuple[TrackMemory, TrackMemory]


def _candidate_id(observation: Mapping[str, Any]) -> int:
    value = observation.get(
        "representative_candidate_id",
        observation.get(
            "official_candidate_id",
            observation.get("original_candidate_id", observation.get("candidate_id")),
        ),
    )
    if value is None:
        raise ValueError("observation is missing a representative candidate id")
    return int(value)


def _keypoints(observation: Mapping[str, Any]) -> np.ndarray:
    values = observation.get("vitpose_keypoints_2d")
    if values is None:
        values = observation.get("_vitpose_keypoints_2d")
    keypoints = np.asarray(values, dtype=np.float64)
    if keypoints.shape != (21, 3):
        raise ValueError("observation keypoints must have shape (21, 3)")
    return keypoints


def _quality(observation: Mapping[str, Any]) -> float:
    quality = observation.get("observation_quality", 0.0)
    if isinstance(quality, Mapping):
        quality = quality.get("score", 0.0)
    return float(np.clip(float(quality), 0.0, 1.0))


def _hand_scale(a_bbox: Sequence[float], b_bbox: Sequence[float]) -> float:
    diagonals = []
    for bbox in (a_bbox, b_bbox):
        values = np.asarray(bbox, dtype=np.float64).reshape(4)
        diagonals.append(
            float(np.linalg.norm(np.maximum(0.0, values[2:] - values[:2])))
        )
    return max(1.0, float(np.mean(diagonals)))


def temporal_association_metrics(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    gap_length: int = 0,
    config: PhysicalHandTemporalAssociationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    previous_bbox = np.asarray(previous["bbox_xyxy"], dtype=np.float64).reshape(4)
    current_bbox = np.asarray(current["bbox_xyxy"], dtype=np.float64).reshape(4)
    previous_geometry = bbox_metrics(previous_bbox)
    current_geometry = bbox_metrics(current_bbox)
    previous_center = np.asarray(previous_geometry["bbox_center_xy"], dtype=np.float64)
    current_center = np.asarray(current_geometry["bbox_center_xy"], dtype=np.float64)
    scale = _hand_scale(previous_bbox, current_bbox)
    center_distance = float(np.linalg.norm(current_center - previous_center) / scale)
    pair_iou = bbox_iou(previous_bbox, current_bbox)
    previous_area = float(previous_geometry["bbox_area_px2"])
    current_area = float(current_geometry["bbox_area_px2"])
    area_ratio = current_area / previous_area if previous_area > 0 else None
    scale_change = (
        float(abs(np.log(np.sqrt(area_ratio))))
        if area_ratio is not None and area_ratio > 0
        else 0.0
    )

    previous_keypoints = _keypoints(previous)
    current_keypoints = _keypoints(current)
    common_valid = (previous_keypoints[:, 2] > config.keypoint_threshold) & (
        current_keypoints[:, 2] > config.keypoint_threshold
    )
    common_count = int(np.count_nonzero(common_valid))
    keypoint_available = common_count >= config.min_common_keypoints
    keypoint_distance = None
    if keypoint_available:
        displacement = np.linalg.norm(
            current_keypoints[common_valid, :2] - previous_keypoints[common_valid, :2],
            axis=1,
        )
        keypoint_distance = float(np.median(displacement) / scale)

    common_palm = common_valid[PALM_KEYPOINT_INDICES]
    common_palm_count = int(np.count_nonzero(common_palm))
    palm_available = common_palm_count >= config.min_common_palm_keypoints
    palm_distance = None
    if palm_available:
        indices = PALM_KEYPOINT_INDICES[common_palm]
        previous_palm = np.median(previous_keypoints[indices, :2], axis=0)
        current_palm = np.median(current_keypoints[indices, :2], axis=0)
        palm_distance = float(np.linalg.norm(current_palm - previous_palm) / scale)

    handedness_changed = (
        str(previous.get("handedness", "")).lower()
        != str(current.get("handedness", "")).lower()
    )
    handedness_penalty = config.handedness_flip_penalty if handedness_changed else 0.0
    components = {
        "bbox_center": config.center_distance_weight * center_distance,
        "bbox_iou": config.bbox_iou_weight * (1.0 - pair_iou),
        "bbox_scale": config.bbox_scale_weight * scale_change,
        "keypoints": (
            config.keypoint_distance_weight * float(keypoint_distance)
            if keypoint_distance is not None
            else 0.0
        ),
        "palm": (
            config.palm_distance_weight * float(palm_distance)
            if palm_distance is not None
            else 0.0
        ),
        "handedness": handedness_penalty,
        "gap": config.gap_connection_penalty * max(0, int(gap_length)),
        "observation_quality": config.observation_quality_weight
        * (1.0 - _quality(current)),
    }
    return {
        "association_cost": float(sum(components.values())),
        "cost_components": components,
        "bbox_iou_prev": float(pair_iou),
        "normalized_center_distance": center_distance,
        "bbox_area_ratio": area_ratio,
        "bbox_scale_change": scale_change,
        "common_valid_keypoints": common_count,
        "keypoint_cost_available": keypoint_available,
        "normalized_keypoint_distance": keypoint_distance,
        "common_valid_palm_keypoints": common_palm_count,
        "palm_cost_available": palm_available,
        "normalized_palm_distance": palm_distance,
        "handedness_penalty": handedness_penalty,
        "handedness_changed": handedness_changed,
        "observation_quality": _quality(current),
        "gap_length": int(gap_length),
    }


def _motion_prediction_cost(
    previous_key: ObservationKey | None,
    last_key: ObservationKey | None,
    current: Mapping[str, Any],
    lookup: Mapping[ObservationKey, Mapping[str, Any]],
    *,
    gap_length: int,
    config: PhysicalHandTemporalAssociationConfig,
) -> float | None:
    """Predict the current center from the last two observations."""
    if previous_key is None or last_key is None:
        return None
    previous = lookup.get(previous_key)
    last = lookup.get(last_key)
    if previous is None or last is None:
        return None
    previous_bbox = np.asarray(previous["bbox_xyxy"], dtype=np.float64).reshape(4)
    last_bbox = np.asarray(last["bbox_xyxy"], dtype=np.float64).reshape(4)
    current_bbox = np.asarray(current["bbox_xyxy"], dtype=np.float64).reshape(4)
    previous_center = (previous_bbox[:2] + previous_bbox[2:]) / 2.0
    last_center = (last_bbox[:2] + last_bbox[2:]) / 2.0
    current_center = (current_bbox[:2] + current_bbox[2:]) / 2.0
    scale = _hand_scale(last_bbox, current_bbox)
    steps = max(1, int(gap_length) + 1)
    predicted_center = last_center + (last_center - previous_center) * steps
    center_residual = float(np.linalg.norm(current_center - predicted_center) / scale)

    previous_points = _keypoints(previous)
    last_points = _keypoints(last)
    current_points = _keypoints(current)
    valid = (
        (previous_points[:, 2] > config.keypoint_threshold)
        & (last_points[:, 2] > config.keypoint_threshold)
        & (current_points[:, 2] > config.keypoint_threshold)
    )
    keypoint_residual = 0.0
    if int(np.count_nonzero(valid)) >= config.min_common_keypoints:
        predicted_points = last_points[valid, :2] + (
            last_points[valid, :2] - previous_points[valid, :2]
        ) * steps
        keypoint_residual = float(
            np.median(np.linalg.norm(current_points[valid, :2] - predicted_points, axis=1))
            / scale
        )
    return config.motion_prediction_weight * (0.65 * center_residual + 0.35 * keypoint_residual)


def _frame_assignments(observation_count: int) -> list[tuple[int | None, int | None]]:
    choices: list[int | None] = [None, *range(observation_count)]
    return [
        (first, second)
        for first, second in product(choices, repeat=2)
        if first is None or second is None or first != second
    ]


def _unassigned_penalty(
    observation: Mapping[str, Any],
    config: PhysicalHandTemporalAssociationConfig,
) -> float:
    return config.unassigned_base_penalty + config.unassigned_quality_weight * _quality(
        observation
    )


def _observation_center_x(observation: Mapping[str, Any]) -> float:
    return float(bbox_metrics(observation["bbox_xyxy"])["bbox_center_xy"][0])


def _update_track(
    memory: TrackMemory,
    observation_index: int | None,
    *,
    frame_idx: int,
    observations: list[Mapping[str, Any]],
    lookup: Mapping[ObservationKey, Mapping[str, Any]],
    config: PhysicalHandTemporalAssociationConfig,
) -> tuple[TrackMemory, float, dict[str, Any]]:
    last_key, anchor_side, previous_key = memory
    ever_started = anchor_side is not None
    if observation_index is None:
        if last_key is None:
            return (
                memory,
                0.0,
                {
                    "state": "missing",
                    "association_cost": None,
                    "gap_length": None,
                    "continued": False,
                    "restart": False,
                },
            )
        missing_length = frame_idx - last_key[0]
        if missing_length <= config.max_gap_frames:
            return (
                memory,
                config.missing_frame_penalty,
                {
                    "state": "missing",
                    "association_cost": config.missing_frame_penalty,
                    "gap_length": missing_length,
                    "continued": True,
                    "restart": False,
                },
            )
        return (
            (None, anchor_side, previous_key),
            0.0,
            {
                "state": "missing",
                "association_cost": None,
                "gap_length": missing_length,
                "continued": False,
                "restart": False,
            },
        )

    current = observations[observation_index]
    current_key = (frame_idx, _candidate_id(current))
    if last_key is not None:
        gap_length = frame_idx - last_key[0] - 1
        if gap_length <= config.max_gap_frames:
            metrics = temporal_association_metrics(
                lookup[last_key],
                current,
                gap_length=gap_length,
                config=config,
            )
            motion_cost = _motion_prediction_cost(
                previous_key, last_key, current, lookup,
                gap_length=gap_length, config=config,
            )
            step = {
                "state": "observed",
                **metrics,
                "continued": True,
                "restart": False,
                "motion_prediction_cost": motion_cost,
            }
            total = float(metrics["association_cost"])
            if motion_cost is not None:
                total += motion_cost
            step["association_cost"] = total
            return ((current_key, anchor_side, last_key), total, step)

    start_penalty = (
        config.track_restart_penalty
        if ever_started
        else config.initial_track_start_penalty
    )
    current_side = str(current.get("handedness", "")).lower()
    restart_handedness_penalty = (
        config.handedness_flip_penalty
        if anchor_side is not None and current_side != anchor_side
        else 0.0
    )
    unary = config.observation_quality_weight * (1.0 - _quality(current))
    total = start_penalty + restart_handedness_penalty + unary
    return (
        (current_key, anchor_side or current_side, None),
        total,
        {
            "state": "observed",
            "association_cost": float(total),
            "cost_components": {
                "track_restart" if ever_started else "track_start": start_penalty,
                "restart_handedness": restart_handedness_penalty,
                "observation_quality": unary,
            },
            "bbox_iou_prev": None,
            "normalized_center_distance": None,
            "bbox_area_ratio": None,
            "bbox_scale_change": None,
            "common_valid_keypoints": 0,
            "keypoint_cost_available": False,
            "normalized_keypoint_distance": None,
            "common_valid_palm_keypoints": 0,
            "palm_cost_available": False,
            "normalized_palm_distance": None,
            "handedness_penalty": restart_handedness_penalty,
            "handedness_changed": bool(restart_handedness_penalty),
            "observation_quality": _quality(current),
            "gap_length": None,
            "continued": False,
            "restart": ever_started,
        },
    )


def _normalise_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[ObservationKey, Mapping[str, Any]]]:
    normalised: list[dict[str, Any]] = []
    lookup: dict[ObservationKey, Mapping[str, Any]] = {}
    previous_frame = None
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if previous_frame is not None and frame_idx != previous_frame + 1:
            raise ValueError("frames must be consecutive and sorted by frame_idx")
        observations = list(frame.get("observations", []))
        if len(observations) > 2:
            raise ValueError(
                "temporal association expects at most two consolidated observations"
            )
        for observation in observations:
            key = (frame_idx, _candidate_id(observation))
            if key in lookup:
                raise ValueError(f"duplicate observation key: {key}")
            _keypoints(observation)
            lookup[key] = observation
        normalised.append({"frame_idx": frame_idx, "observations": observations})
        previous_frame = frame_idx
    return normalised, lookup


def _mark_isolated_unassigned(
    unassigned: list[dict[str, Any]],
    lookup: Mapping[ObservationKey, Mapping[str, Any]],
    config: PhysicalHandTemporalAssociationConfig,
) -> None:
    for row in unassigned:
        key = (int(row["frame_idx"]), int(row["candidate_id"]))
        observation = lookup[key]
        connected = False
        best_cost = None
        best_key = None
        for other in unassigned:
            other_key = (int(other["frame_idx"]), int(other["candidate_id"]))
            if other_key == key:
                continue
            frame_distance = abs(other_key[0] - key[0])
            gap_length = frame_distance - 1
            if frame_distance == 0 or gap_length > config.max_gap_frames:
                continue
            if other_key[0] < key[0]:
                previous, current = lookup[other_key], observation
            else:
                previous, current = observation, lookup[other_key]
            metrics = temporal_association_metrics(
                previous,
                current,
                gap_length=gap_length,
                config=config,
            )
            cost = float(metrics["association_cost"])
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_key = other_key
            if cost <= config.isolated_connection_cost_threshold:
                connected = True
        row["isolated_observation"] = not connected
        row["best_unassigned_neighbor"] = list(best_key) if best_key else None
        row["best_unassigned_connection_cost"] = best_cost


def _track_statistics(
    rows: list[dict[str, Any]],
    track_id: int,
) -> dict[str, Any]:
    track_rows = [row for row in rows if row["track_id"] == track_id]
    observed = [row for row in track_rows if row["state"] == "observed"]
    left_count = sum(row["vitpose_side"] == "left" for row in observed)
    right_count = sum(row["vitpose_side"] == "right" for row in observed)
    dominant = None
    if observed:
        dominant = "left" if left_count >= right_count else "right"
    for row in track_rows:
        row["dominant_track_handedness"] = dominant

    first_observed = min((row["frame_idx"] for row in observed), default=None)
    last_observed = max((row["frame_idx"] for row in observed), default=None)
    missing_segments = 0
    one_frame_gaps = 0
    in_missing = False
    current_missing_length = 0
    if first_observed is not None and last_observed is not None:
        for row in track_rows:
            if not first_observed <= row["frame_idx"] <= last_observed:
                continue
            if row["state"] == "missing":
                if not in_missing:
                    missing_segments += 1
                    in_missing = True
                    current_missing_length = 0
                current_missing_length += 1
            elif in_missing:
                if current_missing_length == 1:
                    one_frame_gaps += 1
                in_missing = False
        if in_missing and current_missing_length == 1:
            one_frame_gaps += 1

    return {
        "track_id": track_id,
        "observed_frames": len(observed),
        "missing_frames_between_first_last": sum(
            row["state"] == "missing"
            and first_observed is not None
            and last_observed is not None
            and first_observed <= row["frame_idx"] <= last_observed
            for row in track_rows
        ),
        "first_observed_frame": first_observed,
        "last_observed_frame": last_observed,
        "fragment_count": sum(
            row["state"] == "observed" and not row["continued"] for row in track_rows
        ),
        "missing_segments": missing_segments,
        "one_frame_disappear_reappear_events": one_frame_gaps,
        "vitpose_left_count": left_count,
        "vitpose_right_count": right_count,
        "dominant_handedness": dominant,
        "handedness_flip_events": sum(bool(row["handedness_flip"]) for row in observed),
        "identity_switch_suspected_events": sum(
            bool(row["identity_switch_suspected"]) for row in observed
        ),
    }


def associate_physical_hand_tracks(
    frames: Sequence[Mapping[str, Any]],
    config: PhysicalHandTemporalAssociationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    normalised, lookup = _normalise_frames(frames)
    if not normalised:
        return {
            "frames": [],
            "track_rows": [],
            "unassigned_observations": [],
            "track_statistics": [],
            "summary": {
                "number_of_tracks": 0,
                "track_fragment_count": 0,
                "isolated_observation_count": 0,
                "handedness_flip_events": 0,
                "track_switch_events": 0,
                "one_frame_disappear_reappear_events": 0,
            },
        }

    dormant: TrackMemory = (None, None, None)
    initial_state: AssociationState = (dormant, dormant)
    previous_layer: dict[AssociationState, _DpNode] = {
        initial_state: _DpNode(0.0, None, (None, None), ({}, {}), (), (dormant, dormant))
    }
    layers: list[dict[AssociationState, _DpNode]] = []
    for frame in normalised:
        frame_idx = frame["frame_idx"]
        observations = frame["observations"]
        current_layer: dict[AssociationState, _DpNode] = {}
        for previous_state, previous_node in previous_layer.items():
            for assignments in _frame_assignments(len(observations)):
                assigned = {index for index in assignments if index is not None}
                unassigned_indices = tuple(
                    index for index in range(len(observations)) if index not in assigned
                )
                total = previous_node.cost + sum(
                    _unassigned_penalty(observations[index], config)
                    for index in unassigned_indices
                )
                new_memories: list[TrackMemory] = []
                steps: list[dict[str, Any]] = []
                for track_id in range(config.max_tracks):
                    memory, cost, step = _update_track(
                        previous_node.memories[track_id],
                        assignments[track_id],
                        frame_idx=frame_idx,
                        observations=observations,
                        lookup=lookup,
                        config=config,
                    )
                    new_memories.append(memory)
                    steps.append(step)
                    total += cost
                if (
                    assignments[0] is not None
                    and assignments[1] is not None
                    and previous_state[0][1] is None
                    and previous_state[1][1] is None
                    and _observation_center_x(observations[assignments[0]])
                    > _observation_center_x(observations[assignments[1]])
                ):
                    total += config.canonical_start_order_penalty
                new_state: AssociationState = (
                    (new_memories[0][0], new_memories[0][1]),
                    (new_memories[1][0], new_memories[1][1]),
                )
                existing = current_layer.get(new_state)
                if existing is None or total < existing.cost - 1e-12:
                    current_layer[new_state] = _DpNode(
                        cost=float(total),
                        previous_state=previous_state,
                        assignments=assignments,
                        track_steps=(steps[0], steps[1]),
                        unassigned_indices=unassigned_indices,
                        memories=(new_memories[0], new_memories[1]),
                    )
        layers.append(current_layer)
        previous_layer = current_layer

    final_state = min(previous_layer, key=lambda state: previous_layer[state].cost)
    decisions: list[_DpNode] = []
    state = final_state
    for layer in reversed(layers):
        node = layer[state]
        decisions.append(node)
        if node.previous_state is None:
            break
        state = node.previous_state
    decisions.reverse()
    if len(decisions) != len(normalised):
        raise RuntimeError("could not reconstruct full temporal association path")

    track_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    fragment_ids = [-1, -1]
    previous_sides: list[str | None] = [None, None]
    for frame, decision in zip(normalised, decisions):
        frame_idx = frame["frame_idx"]
        observations = frame["observations"]
        current_track_rows: list[dict[str, Any]] = []
        for track_id, (observation_index, step) in enumerate(
            zip(decision.assignments, decision.track_steps)
        ):
            row: dict[str, Any] = {
                "frame_idx": frame_idx,
                "track_id": track_id,
                "state": step["state"],
                "selected_cluster_id": None,
                "selected_candidate_id": None,
                "vitpose_side": None,
                "dominant_track_handedness": None,
                "association_cost": step.get("association_cost"),
                "bbox_iou_prev": step.get("bbox_iou_prev"),
                "normalized_center_distance": step.get("normalized_center_distance"),
                "bbox_area_ratio": step.get("bbox_area_ratio"),
                "bbox_scale_change": step.get("bbox_scale_change"),
                "normalized_keypoint_distance": step.get(
                    "normalized_keypoint_distance"
                ),
                "keypoint_cost_available": step.get("keypoint_cost_available", False),
                "normalized_palm_distance": step.get("normalized_palm_distance"),
                "palm_cost_available": step.get("palm_cost_available", False),
                "motion_prediction_cost": step.get("motion_prediction_cost"),
                "handedness_penalty": step.get("handedness_penalty", 0.0),
                "observation_quality": step.get("observation_quality"),
                "gap_length": step.get("gap_length"),
                "handedness_flip": False,
                "identity_switch_suspected": False,
                "continued": bool(step.get("continued", False)),
                "track_fragment_id": fragment_ids[track_id]
                if fragment_ids[track_id] >= 0
                else None,
            }
            if observation_index is not None:
                observation = observations[observation_index]
                if not step.get("continued"):
                    fragment_ids[track_id] += 1
                    previous_sides[track_id] = None
                side = str(observation.get("handedness", "")).lower()
                handedness_flip = bool(
                    step.get("continued")
                    and previous_sides[track_id] is not None
                    and side != previous_sides[track_id]
                )
                row.update(
                    {
                        "selected_cluster_id": observation.get(
                            "assigned_cluster_id", observation.get("cluster_id")
                        ),
                        "selected_candidate_id": _candidate_id(observation),
                        "source_person_id": observation.get("person_index"),
                        "vitpose_side": side,
                        "observation_quality": _quality(observation),
                        "handedness_flip": handedness_flip,
                        "identity_switch_suspected": bool(
                            step.get("continued")
                            and step.get("association_cost") is not None
                            and float(step["association_cost"])
                            > config.identity_switch_cost_threshold
                        ),
                        "track_fragment_id": fragment_ids[track_id],
                    }
                )
                previous_sides[track_id] = side
            current_track_rows.append(row)
            track_rows.append(row)
        frame_unassigned = []
        for observation_index in decision.unassigned_indices:
            observation = observations[observation_index]
            item = {
                "frame_idx": frame_idx,
                "candidate_id": _candidate_id(observation),
                "cluster_id": observation.get(
                    "assigned_cluster_id", observation.get("cluster_id")
                ),
                "vitpose_side": observation.get("handedness"),
                "observation_quality": _quality(observation),
                "isolated_observation": None,
            }
            unassigned.append(item)
            frame_unassigned.append(item)
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "tracks": current_track_rows,
                "unassigned_observations": frame_unassigned,
            }
        )

    _mark_isolated_unassigned(unassigned, lookup, config)
    unassigned_by_key = {
        (row["frame_idx"], row["candidate_id"]): row for row in unassigned
    }
    for frame in frame_rows:
        frame["unassigned_observations"] = [
            unassigned_by_key[(row["frame_idx"], row["candidate_id"])]
            for row in frame["unassigned_observations"]
        ]

    statistics = [
        _track_statistics(track_rows, track_id) for track_id in range(config.max_tracks)
    ]
    summary = {
        "number_of_tracks": sum(stat["observed_frames"] > 0 for stat in statistics),
        "track_fragment_count": sum(stat["fragment_count"] for stat in statistics),
        "missing_segments": sum(stat["missing_segments"] for stat in statistics),
        "isolated_observation_count": sum(
            bool(row["isolated_observation"]) for row in unassigned
        ),
        "unassigned_observation_count": len(unassigned),
        "handedness_flip_events": sum(
            stat["handedness_flip_events"] for stat in statistics
        ),
        "track_switch_events": sum(
            stat["identity_switch_suspected_events"] for stat in statistics
        ),
        "one_frame_disappear_reappear_events": sum(
            stat["one_frame_disappear_reappear_events"] for stat in statistics
        ),
        "total_cost": float(previous_layer[final_state].cost),
    }
    return {
        "frames": frame_rows,
        "track_rows": track_rows,
        "unassigned_observations": unassigned,
        "track_statistics": statistics,
        "summary": summary,
    }
