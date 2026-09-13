from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

from observation_frontend.failure_diagnostics import bbox_iou, bbox_metrics

PALM_KEYPOINT_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)


@dataclass(frozen=True)
class HandObservationConsolidationConfig:
    keypoint_threshold: float = 0.5
    bbox_iou_evidence_threshold: float = 0.30
    strong_bbox_iou_threshold: float = 0.65
    max_normalized_center_distance: float = 0.55
    bbox_evidence_max_center_distance: float = 0.35
    keypoint_distance_threshold: float = 0.35
    strong_keypoint_distance_threshold: float = 0.12
    palm_distance_threshold: float = 0.25
    min_common_keypoints: int = 4
    min_common_palm_keypoints: int = 2
    same_handedness_bonus: float = 0.10
    mixed_handedness_penalty: float = 0.20
    duplicate_evidence_threshold: float = 1.50
    max_physical_hand_clusters: int = 2
    quality_valid_keypoints_weight: float = 0.45
    quality_mean_keypoint_score_weight: float = 0.25
    quality_median_keypoint_score_weight: float = 0.20
    quality_person_score_weight: float = 0.05
    quality_bbox_completeness_weight: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be in [0, 1]")
        if self.min_common_keypoints < 1 or self.min_common_palm_keypoints < 1:
            raise ValueError("minimum common-keypoint counts must be positive")
        if self.max_physical_hand_clusters < 1:
            raise ValueError("max_physical_hand_clusters must be positive")
        quality_weights = [
            self.quality_valid_keypoints_weight,
            self.quality_mean_keypoint_score_weight,
            self.quality_median_keypoint_score_weight,
            self.quality_person_score_weight,
            self.quality_bbox_completeness_weight,
        ]
        if any(weight < 0 for weight in quality_weights):
            raise ValueError("quality weights must be non-negative")
        if not np.isclose(sum(quality_weights), 1.0):
            raise ValueError("quality weights must sum to 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = HandObservationConsolidationConfig()


def _candidate_id(observation: Mapping[str, Any]) -> int:
    value = observation.get(
        "original_candidate_id",
        observation.get("official_candidate_id", observation.get("candidate_id")),
    )
    if value is None:
        raise ValueError("hand observation is missing a candidate id")
    return int(value)


def _keypoints(observation: Mapping[str, Any]) -> np.ndarray:
    values = observation.get("vitpose_keypoints_2d")
    if values is None:
        values = observation.get("_vitpose_keypoints_2d")
    keypoints = np.asarray(values, dtype=np.float64)
    if keypoints.shape != (21, 3):
        raise ValueError("hand observation keypoints must have shape (21, 3)")
    return keypoints


def _hand_scale(a_bbox: Sequence[float], b_bbox: Sequence[float]) -> float:
    diagonals = []
    for bbox in (a_bbox, b_bbox):
        values = np.asarray(bbox, dtype=np.float64).reshape(4)
        diagonals.append(float(np.linalg.norm(np.maximum(0.0, values[2:] - values[:2]))))
    return max(1.0, float(np.mean(diagonals)))


def _bbox_completeness(
    bbox_xyxy: Sequence[float],
    image_size: tuple[int, int] | None,
) -> tuple[float, bool]:
    if image_size is None:
        return 1.0, False
    image_width, image_height = image_size
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    clipped_x1 = min(max(float(x1), 0.0), float(image_width))
    clipped_y1 = min(max(float(y1), 0.0), float(image_height))
    clipped_x2 = min(max(float(x2), 0.0), float(image_width))
    clipped_y2 = min(max(float(y2), 0.0), float(image_height))
    visible_area = max(0.0, clipped_x2 - clipped_x1) * max(
        0.0, clipped_y2 - clipped_y1
    )
    completeness = visible_area / area if area > 0 else 0.0
    return float(completeness), bool(completeness < 1.0 - 1e-9)


def hand_observation_pair_metrics(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    config: HandObservationConsolidationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    a_bbox = np.asarray(a["bbox_xyxy"], dtype=np.float64).reshape(4)
    b_bbox = np.asarray(b["bbox_xyxy"], dtype=np.float64).reshape(4)
    a_geometry = bbox_metrics(a_bbox)
    b_geometry = bbox_metrics(b_bbox)
    a_center = np.asarray(a_geometry["bbox_center_xy"], dtype=np.float64)
    b_center = np.asarray(b_geometry["bbox_center_xy"], dtype=np.float64)
    scale = _hand_scale(a_bbox, b_bbox)
    center_distance = float(np.linalg.norm(a_center - b_center) / scale)

    a_keypoints = _keypoints(a)
    b_keypoints = _keypoints(b)
    common_valid = (a_keypoints[:, 2] > config.keypoint_threshold) & (
        b_keypoints[:, 2] > config.keypoint_threshold
    )
    common_count = int(np.count_nonzero(common_valid))
    keypoint_available = common_count >= config.min_common_keypoints
    keypoint_distance = None
    if keypoint_available:
        distances = np.linalg.norm(
            a_keypoints[common_valid, :2] - b_keypoints[common_valid, :2], axis=1
        )
        keypoint_distance = float(np.median(distances) / scale)

    common_palm = common_valid[PALM_KEYPOINT_INDICES]
    common_palm_count = int(np.count_nonzero(common_palm))
    palm_available = common_palm_count >= config.min_common_palm_keypoints
    palm_distance = None
    if palm_available:
        indices = PALM_KEYPOINT_INDICES[common_palm]
        a_palm = np.median(a_keypoints[indices, :2], axis=0)
        b_palm = np.median(b_keypoints[indices, :2], axis=0)
        palm_distance = float(np.linalg.norm(a_palm - b_palm) / scale)

    pair_iou = bbox_iou(a_bbox, b_bbox)
    bbox_evidence = (
        pair_iou >= config.bbox_iou_evidence_threshold
        and center_distance <= config.bbox_evidence_max_center_distance
    )
    keypoint_evidence = bool(
        keypoint_available
        and keypoint_distance is not None
        and keypoint_distance <= config.keypoint_distance_threshold
    )
    palm_evidence = bool(
        palm_available
        and palm_distance is not None
        and palm_distance <= config.palm_distance_threshold
    )
    same_handedness = str(a.get("handedness", "")).lower() == str(
        b.get("handedness", "")
    ).lower()
    evidence_score = (
        float(bbox_evidence)
        + float(keypoint_evidence)
        + 0.75 * float(palm_evidence)
        + (
            config.same_handedness_bonus
            if same_handedness
            else -config.mixed_handedness_penalty
        )
    )
    strong_bbox_evidence = bool(
        pair_iou >= config.strong_bbox_iou_threshold
        and center_distance <= config.bbox_evidence_max_center_distance
    )
    strong_keypoint_evidence = bool(
        keypoint_available
        and keypoint_distance is not None
        and keypoint_distance <= config.strong_keypoint_distance_threshold
        and center_distance <= config.max_normalized_center_distance
    )
    same_physical_hand = bool(
        center_distance <= config.max_normalized_center_distance
        and (
            evidence_score >= config.duplicate_evidence_threshold
            or strong_bbox_evidence
            or strong_keypoint_evidence
        )
    )
    return {
        "candidate_a": _candidate_id(a),
        "candidate_b": _candidate_id(b),
        "bbox_iou": float(pair_iou),
        "normalized_center_distance": center_distance,
        "common_valid_keypoints": common_count,
        "keypoint_similarity_available": keypoint_available,
        "normalized_keypoint_distance": keypoint_distance,
        "common_valid_palm_keypoints": common_palm_count,
        "palm_similarity_available": palm_available,
        "normalized_palm_center_distance": palm_distance,
        "handedness_agrees": same_handedness,
        "bbox_evidence": bbox_evidence,
        "keypoint_evidence": keypoint_evidence,
        "palm_evidence": palm_evidence,
        "handedness_soft_adjustment": (
            config.same_handedness_bonus
            if same_handedness
            else -config.mixed_handedness_penalty
        ),
        "duplicate_evidence_score": float(evidence_score),
        "strong_bbox_evidence": strong_bbox_evidence,
        "strong_keypoint_evidence": strong_keypoint_evidence,
        "same_physical_hand": same_physical_hand,
    }


def observation_quality(
    observation: Mapping[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    config: HandObservationConsolidationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    keypoints = _keypoints(observation)
    scores = keypoints[:, 2]
    valid = scores > config.keypoint_threshold
    valid_count = int(np.count_nonzero(valid))
    valid_ratio = valid_count / 21.0
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    person_score = float(
        observation.get("person_score", observation.get("detector_score", 0.0))
    )
    completeness, clipped = _bbox_completeness(
        observation["bbox_xyxy"], image_size
    )
    components = {
        "valid_keypoints_ratio": valid_ratio,
        "mean_keypoint_score": mean_score,
        "median_keypoint_score": median_score,
        "source_person_score": person_score,
        "bbox_completeness": completeness,
    }
    weighted = {
        "valid_keypoints": config.quality_valid_keypoints_weight * valid_ratio,
        "mean_keypoint_score": config.quality_mean_keypoint_score_weight * mean_score,
        "median_keypoint_score": config.quality_median_keypoint_score_weight
        * median_score,
        "source_person_score": config.quality_person_score_weight * person_score,
        "bbox_completeness": config.quality_bbox_completeness_weight * completeness,
    }
    return {
        "score": float(sum(weighted.values())),
        "components": components,
        "weighted_components": weighted,
        "bbox_clipped_by_image_boundary": clipped,
    }


def _connected_components(
    count: int,
    duplicate_pairs: Iterable[tuple[int, int]],
) -> list[list[int]]:
    adjacency = [set() for _ in range(count)]
    for first, second in duplicate_pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[list[int]] = []
    remaining = set(range(count))
    while remaining:
        root = min(remaining)
        stack = [root]
        component: list[int] = []
        remaining.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = adjacency[current] & remaining
            remaining.difference_update(neighbours)
            stack.extend(sorted(neighbours, reverse=True))
        components.append(sorted(component))
    return components


def consolidate_hand_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int] | None = None,
    config: HandObservationConsolidationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    eligible = [
        observation
        for observation in observations
        if bool(observation.get("official_gate_passed", True))
        and observation.get("bbox_xyxy") is not None
    ]
    pairwise_metrics: list[dict[str, Any]] = []
    duplicate_pairs: list[tuple[int, int]] = []
    for (first_index, first), (second_index, second) in combinations(
        enumerate(eligible), 2
    ):
        metrics = hand_observation_pair_metrics(first, second, config)
        pairwise_metrics.append(metrics)
        if metrics["same_physical_hand"]:
            duplicate_pairs.append((first_index, second_index))

    components = _connected_components(len(eligible), duplicate_pairs)
    quality_by_index = {
        index: observation_quality(
            observation, image_size=image_size, config=config
        )
        for index, observation in enumerate(eligible)
    }
    cluster_work: list[dict[str, Any]] = []
    for cluster_id, member_indices in enumerate(components):
        representative_index = max(
            member_indices,
            key=lambda index: (
                quality_by_index[index]["score"],
                int(np.count_nonzero(_keypoints(eligible[index])[:, 2] > config.keypoint_threshold)),
                -_candidate_id(eligible[index]),
            ),
        )
        member_ids = [_candidate_id(eligible[index]) for index in member_indices]
        member_id_set = set(member_ids)
        cluster_pairs = [
            pair
            for pair in pairwise_metrics
            if pair["candidate_a"] in member_id_set
            and pair["candidate_b"] in member_id_set
        ]
        keypoint_distances = [
            float(pair["normalized_keypoint_distance"])
            for pair in cluster_pairs
            if pair["normalized_keypoint_distance"] is not None
        ]
        center_distances = [
            float(pair["normalized_center_distance"]) for pair in cluster_pairs
        ]
        handednesses = {
            str(eligible[index].get("handedness", "")).lower()
            for index in member_indices
        }
        cluster_work.append(
            {
                "cluster_id": cluster_id,
                "member_indices": member_indices,
                "member_candidate_ids": member_ids,
                "representative_index": representative_index,
                "representative_candidate_id": _candidate_id(
                    eligible[representative_index]
                ),
                "representative_quality": quality_by_index[representative_index],
                "cluster_size": len(member_indices),
                "max_pairwise_bbox_iou": max(
                    (float(pair["bbox_iou"]) for pair in cluster_pairs), default=None
                ),
                "median_pairwise_center_distance": (
                    float(np.median(center_distances)) if center_distances else None
                ),
                "median_pairwise_keypoint_distance": (
                    float(np.median(keypoint_distances)) if keypoint_distances else None
                ),
                "mixed_handedness": len(handednesses) > 1,
            }
        )

    ranked = sorted(
        cluster_work,
        key=lambda cluster: (
            cluster["representative_quality"]["score"],
            -cluster["representative_candidate_id"],
        ),
        reverse=True,
    )
    retained_ids = {
        cluster["cluster_id"]
        for cluster in ranked[: config.max_physical_hand_clusters]
    }
    retention_rank = {
        cluster["cluster_id"]: rank for rank, cluster in enumerate(ranked)
    }
    index_to_cluster: dict[int, dict[str, Any]] = {}
    for cluster in cluster_work:
        cluster["retained"] = cluster["cluster_id"] in retained_ids
        cluster["retention_rank"] = retention_rank[cluster["cluster_id"]]
        for index in cluster.pop("member_indices"):
            index_to_cluster[index] = cluster
        cluster.pop("representative_index")

    observation_diagnostics: list[dict[str, Any]] = []
    selected_observations: list[Mapping[str, Any]] = []
    for index, observation in enumerate(eligible):
        cluster = index_to_cluster[index]
        candidate_id = _candidate_id(observation)
        is_representative = candidate_id == cluster["representative_candidate_id"]
        selected = bool(cluster["retained"] and is_representative)
        if selected:
            selected_observations.append(observation)
        geometry = bbox_metrics(observation["bbox_xyxy"])
        observation_diagnostics.append(
            {
                "frame_idx": observation.get("frame_idx"),
                "original_candidate_id": candidate_id,
                "source_person_id": int(
                    observation.get("person_index", observation.get("source_person_id", -1))
                ),
                "source_person_score": float(
                    observation.get(
                        "person_score", observation.get("detector_score", 0.0)
                    )
                ),
                "vitpose_side": str(
                    observation.get("handedness", observation.get("vitpose_side", ""))
                ),
                "keypoint_score": _keypoints(observation)[:, 2].astype(float).tolist(),
                "valid_keypoints": int(
                    np.count_nonzero(
                        _keypoints(observation)[:, 2] > config.keypoint_threshold
                    )
                ),
                "mean_keypoint_score": float(
                    np.mean(_keypoints(observation)[:, 2])
                ),
                "median_keypoint_score": float(
                    np.median(_keypoints(observation)[:, 2])
                ),
                "hand_bbox": geometry["bbox_xyxy"],
                "bbox_center": geometry["bbox_center_xy"],
                "bbox_area": geometry["bbox_area_px2"],
                "official_gate_passed": True,
                "assigned_cluster_id": cluster["cluster_id"],
                "cluster_retained": cluster["retained"],
                "selected_as_cluster_representative": selected,
                "observation_quality": quality_by_index[index],
                "duplicate_of_candidate_id": (
                    None if is_representative else cluster["representative_candidate_id"]
                ),
            }
        )

    selected_observations.sort(key=_candidate_id)
    public_clusters = sorted(cluster_work, key=lambda cluster: cluster["cluster_id"])
    return {
        "selected_observations": selected_observations,
        "observation_diagnostics": observation_diagnostics,
        "clusters": public_clusters,
        "pairwise_metrics": pairwise_metrics,
        "raw_observation_count": len(eligible),
        "pre_cap_cluster_count": len(public_clusters),
        "physical_hand_cluster_count": sum(
            bool(cluster["retained"]) for cluster in public_clusters
        ),
        "selected_candidate_ids": [
            _candidate_id(observation) for observation in selected_observations
        ],
    }
