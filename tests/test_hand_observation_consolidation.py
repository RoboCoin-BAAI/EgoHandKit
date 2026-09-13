from __future__ import annotations

import numpy as np

from observation_frontend.hand_observation_consolidation import (
    consolidate_hand_observations,
)


def _observation(
    candidate_id: int,
    bbox: list[float],
    *,
    handedness: str = "left",
    offset: tuple[float, float] = (0.0, 0.0),
    valid_keypoints: int = 21,
    keypoint_score: float = 0.8,
    person_score: float = 0.8,
) -> dict:
    x1, y1, x2, y2 = bbox
    xs = np.linspace(x1, x2, 21) + offset[0]
    ys = np.linspace(y1, y2, 21) + offset[1]
    scores = np.full(21, 0.1, dtype=np.float32)
    scores[:valid_keypoints] = keypoint_score
    keypoints = np.stack([xs, ys, scores], axis=1)
    return {
        "frame_idx": 0,
        "official_candidate_id": candidate_id,
        "candidate_id": candidate_id,
        "person_index": candidate_id,
        "person_score": person_score,
        "handedness": handedness,
        "bbox_xyxy": bbox,
        "vitpose_keypoints_2d": keypoints.tolist(),
        "official_gate_passed": valid_keypoints >= 4,
    }


def test_overlapping_same_handedness_observations_form_one_cluster() -> None:
    result = consolidate_hand_observations(
        [
            _observation(0, [0, 0, 100, 100]),
            _observation(1, [4, 2, 104, 102], offset=(2, 1)),
        ]
    )

    assert result["physical_hand_cluster_count"] == 1
    assert len(result["selected_observations"]) == 1
    assert result["clusters"][0]["cluster_size"] == 2
    assert result["clusters"][0]["mixed_handedness"] is False


def test_overlapping_mixed_handedness_is_not_a_hard_split() -> None:
    result = consolidate_hand_observations(
        [
            _observation(0, [0, 0, 100, 100], handedness="left"),
            _observation(
                1,
                [3, 2, 103, 102],
                handedness="right",
                offset=(2, 1),
            ),
        ]
    )

    assert result["physical_hand_cluster_count"] == 1
    assert result["clusters"][0]["mixed_handedness"] is True


def test_spatially_separated_observations_remain_two_clusters() -> None:
    result = consolidate_hand_observations(
        [
            _observation(0, [0, 0, 100, 100]),
            _observation(1, [300, 0, 400, 100], handedness="right"),
        ]
    )

    assert result["physical_hand_cluster_count"] == 2
    assert len(result["selected_observations"]) == 2


def test_two_duplicates_plus_one_independent_form_two_clusters() -> None:
    result = consolidate_hand_observations(
        [
            _observation(0, [0, 0, 100, 100]),
            _observation(1, [5, 0, 105, 100], offset=(2, 0)),
            _observation(2, [300, 0, 400, 100], handedness="right"),
        ]
    )

    assert result["physical_hand_cluster_count"] == 2
    assert sorted(cluster["cluster_size"] for cluster in result["clusters"]) == [1, 2]


def test_cluster_representative_prefers_keypoint_quality_over_person_score() -> None:
    lower_hand_quality = _observation(
        0,
        [0, 0, 100, 100],
        valid_keypoints=8,
        keypoint_score=0.6,
        person_score=0.99,
    )
    higher_hand_quality = _observation(
        1,
        [2, 0, 102, 100],
        offset=(1, 0),
        valid_keypoints=21,
        keypoint_score=0.8,
        person_score=0.55,
    )

    result = consolidate_hand_observations(
        [lower_hand_quality, higher_hand_quality]
    )

    assert result["selected_candidate_ids"] == [1]
    qualities = {
        row["original_candidate_id"]: row["observation_quality"]["score"]
        for row in result["observation_diagnostics"]
    }
    assert qualities[1] > qualities[0]


def test_no_valid_observation_produces_no_cluster() -> None:
    result = consolidate_hand_observations(
        [_observation(0, [0, 0, 100, 100], valid_keypoints=3)]
    )

    assert result["physical_hand_cluster_count"] == 0
    assert result["selected_observations"] == []


def test_single_observation_is_preserved() -> None:
    observation = _observation(7, [0, 0, 100, 100])

    result = consolidate_hand_observations([observation])

    assert result["physical_hand_cluster_count"] == 1
    assert result["selected_observations"] == [observation]
    assert result["selected_candidate_ids"] == [7]


def test_cluster_count_is_not_padded_to_two() -> None:
    result = consolidate_hand_observations(
        [
            _observation(0, [0, 0, 100, 100]),
            _observation(1, [2, 0, 102, 100], offset=(1, 0)),
        ]
    )

    assert result["pre_cap_cluster_count"] == 1
    assert result["physical_hand_cluster_count"] == 1
