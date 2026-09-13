from __future__ import annotations

import numpy as np

from observation_frontend.physical_hand_temporal_association import (
    PhysicalHandTemporalAssociationConfig,
    associate_physical_hand_tracks,
)


def _observation(
    frame_idx: int,
    candidate_id: int,
    center: tuple[float, float],
    *,
    handedness: str = "left",
    quality: float = 0.9,
) -> dict:
    cx, cy = center
    bbox = [cx - 30, cy - 40, cx + 30, cy + 40]
    xs = np.linspace(cx - 20, cx + 20, 21)
    ys = np.linspace(cy - 30, cy + 30, 21)
    keypoints = np.stack([xs, ys, np.full(21, 0.9)], axis=1)
    return {
        "frame_idx": frame_idx,
        "official_candidate_id": candidate_id,
        "assigned_cluster_id": candidate_id,
        "person_index": 0,
        "handedness": handedness,
        "bbox_xyxy": bbox,
        "vitpose_keypoints_2d": keypoints.tolist(),
        "observation_quality": {"score": quality},
    }


def _frame(frame_idx: int, observations: list[dict]) -> dict:
    return {"frame_idx": frame_idx, "observations": observations}


def _observed(result: dict) -> list[dict]:
    return [row for row in result["track_rows"] if row["state"] == "observed"]


def _track_for_candidate(result: dict, frame_idx: int, candidate_id: int) -> int:
    return next(
        row["track_id"]
        for row in _observed(result)
        if row["frame_idx"] == frame_idx
        and row["selected_candidate_id"] == candidate_id
    )


def test_two_stable_observations_associate_to_one_track() -> None:
    frames = [_frame(i, [_observation(i, 0, (100 + 2 * i, 100))]) for i in range(3)]

    result = associate_physical_hand_tracks(frames)

    assert result["summary"]["number_of_tracks"] == 1
    assert len({_track_for_candidate(result, i, 0) for i in range(3)}) == 1


def test_two_spatially_separated_hands_form_two_tracks() -> None:
    frames = [
        _frame(
            i,
            [
                _observation(i, 0, (100 + i, 100), handedness="left"),
                _observation(i, 1, (400 - i, 100), handedness="right"),
            ],
        )
        for i in range(3)
    ]

    result = associate_physical_hand_tracks(frames)

    assert result["summary"]["number_of_tracks"] == 2
    left_tracks = {_track_for_candidate(result, i, 0) for i in range(3)}
    right_tracks = {_track_for_candidate(result, i, 1) for i in range(3)}
    assert len(left_tracks) == len(right_tracks) == 1
    assert left_tracks != right_tracks


def test_handedness_flip_can_remain_on_same_physical_track() -> None:
    sides = ["left", "right", "left"]
    frames = [
        _frame(i, [_observation(i, 0, (100 + i, 100), handedness=side)])
        for i, side in enumerate(sides)
    ]

    result = associate_physical_hand_tracks(frames)

    assert len({_track_for_candidate(result, i, 0) for i in range(3)}) == 1
    assert result["summary"]["handedness_flip_events"] == 2


def test_single_missing_frame_keeps_identity_across_gap() -> None:
    frames = [
        _frame(0, [_observation(0, 0, (100, 100))]),
        _frame(1, [_observation(1, 0, (102, 100))]),
        _frame(2, []),
        _frame(3, [_observation(3, 0, (106, 100))]),
        _frame(4, [_observation(4, 0, (108, 100))]),
    ]
    config = PhysicalHandTemporalAssociationConfig(max_gap_frames=1)

    result = associate_physical_hand_tracks(frames, config)

    track_ids = {
        _track_for_candidate(result, frame_idx, 0) for frame_idx in (0, 1, 3, 4)
    }
    assert len(track_ids) == 1
    track_id = next(iter(track_ids))
    missing = next(
        row
        for row in result["track_rows"]
        if row["frame_idx"] == 2 and row["track_id"] == track_id
    )
    assert missing["state"] == "missing"


def test_gap_longer_than_max_gap_creates_a_new_fragment() -> None:
    frames = [
        _frame(0, [_observation(0, 0, (100, 100))]),
        _frame(1, [_observation(1, 0, (102, 100))]),
        _frame(2, []),
        _frame(3, []),
        _frame(4, [_observation(4, 0, (108, 100))]),
        _frame(5, [_observation(5, 0, (110, 100))]),
    ]
    config = PhysicalHandTemporalAssociationConfig(max_gap_frames=1)

    result = associate_physical_hand_tracks(frames, config)

    assert result["summary"]["track_fragment_count"] == 2


def test_long_gap_restart_uses_prior_side_only_as_soft_identity_evidence() -> None:
    frames = [
        _frame(
            0,
            [
                _observation(0, 0, (100, 100), handedness="left"),
                _observation(0, 1, (400, 100), handedness="right"),
            ],
        ),
        _frame(1, []),
        _frame(2, []),
        _frame(
            3,
            [
                _observation(3, 0, (400, 100), handedness="left"),
                _observation(3, 1, (100, 100), handedness="right"),
            ],
        ),
    ]
    config = PhysicalHandTemporalAssociationConfig(max_gap_frames=1)

    result = associate_physical_hand_tracks(frames, config)

    assert _track_for_candidate(result, 0, 0) == _track_for_candidate(result, 3, 0)
    assert _track_for_candidate(result, 0, 1) == _track_for_candidate(result, 3, 1)


def test_isolated_low_quality_observation_does_not_steal_stable_track() -> None:
    frames = []
    for frame_idx in range(5):
        observations = [_observation(frame_idx, 0, (100 + frame_idx, 100))]
        if frame_idx == 2:
            observations.append(
                _observation(
                    frame_idx,
                    1,
                    (500, 400),
                    handedness="right",
                    quality=0.25,
                )
            )
        frames.append(_frame(frame_idx, observations))

    result = associate_physical_hand_tracks(frames)

    assert len({_track_for_candidate(result, i, 0) for i in range(5)}) == 1
    assert result["summary"]["isolated_observation_count"] == 1
    assert result["unassigned_observations"][0]["candidate_id"] == 1


def test_observation_is_exclusive_to_one_track() -> None:
    frames = [_frame(i, [_observation(i, 0, (100 + i, 100))]) for i in range(3)]

    result = associate_physical_hand_tracks(frames)

    for frame_idx in range(3):
        assigned = [
            row["selected_candidate_id"]
            for row in _observed(result)
            if row["frame_idx"] == frame_idx
        ]
        assert assigned == [0]


def test_one_observation_leaves_other_track_missing() -> None:
    frames = [_frame(i, [_observation(i, 0, (100 + i, 100))]) for i in range(3)]

    result = associate_physical_hand_tracks(frames)

    for frame in result["frames"]:
        assert sum(row["state"] == "observed" for row in frame["tracks"]) == 1
        assert sum(row["state"] == "missing" for row in frame["tracks"]) == 1


def test_empty_frame_allows_both_tracks_missing() -> None:
    result = associate_physical_hand_tracks([_frame(0, [])])

    assert all(row["state"] == "missing" for row in result["frames"][0]["tracks"])
    assert result["summary"]["number_of_tracks"] == 0


def test_single_physical_hand_does_not_force_second_track() -> None:
    frames = [_frame(i, [_observation(i, 0, (100 + i, 100))]) for i in range(4)]

    result = associate_physical_hand_tracks(frames)

    assert result["summary"]["number_of_tracks"] == 1
    assert all(
        stat["observed_frames"] == 0
        for stat in result["track_statistics"]
        if stat["track_id"] != _track_for_candidate(result, 0, 0)
    )
