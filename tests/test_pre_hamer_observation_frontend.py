from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from observation_frontend.hand_observation_consolidation import (
    consolidate_hand_observations,
)
from observation_frontend.physical_hand_temporal_association import (
    associate_physical_hand_tracks,
)
from observation_frontend.pre_hamer_observation_frontend import (
    select_pre_hamer_observations,
)


def _observation(
    frame_idx: int,
    candidate_id: int,
    center: tuple[float, float],
    *,
    handedness: str = "left",
    offset: tuple[float, float] = (0.0, 0.0),
) -> dict:
    cx, cy = center
    bbox = [cx - 30, cy - 40, cx + 30, cy + 40]
    keypoints = np.stack(
        [
            np.linspace(cx - 20, cx + 20, 21) + offset[0],
            np.linspace(cy - 30, cy + 30, 21) + offset[1],
            np.full(21, 0.9),
        ],
        axis=1,
    )
    return {
        "frame_idx": frame_idx,
        "candidate_id": candidate_id,
        "person_index": candidate_id,
        "person_score": 0.8,
        "handedness": handedness,
        "bbox_xyxy": bbox,
        "vitpose_keypoints_2d": keypoints.tolist(),
        "official_gate_passed": True,
    }


def test_frontend_deduplicates_before_hamer_and_keeps_physical_identity() -> None:
    frames = []
    for frame_idx in range(3):
        primary = _observation(frame_idx, 0, (100 + frame_idx, 100))
        duplicate = _observation(
            frame_idx,
            1,
            (103 + frame_idx, 101),
            handedness="right",
            offset=(1, 1),
        )
        frames.append(
            {
                "frame_idx": frame_idx,
                "observations": [primary, duplicate],
            }
        )

    result = select_pre_hamer_observations(frames, image_size=(640, 480))

    assert result["requires_hamer_outputs"] is False
    assert result["summary"]["input_observation_count"] == 6
    assert result["summary"]["consolidated_observation_count"] == 3
    assert result["summary"]["hamer_input_observation_count"] == 3
    track_ids = {
        frame["selected_for_hamer"][0]["physical_track_id"]
        for frame in result["frames"]
    }
    assert len(track_ids) == 1
    assert all(len(frame["selected_for_hamer"]) == 1 for frame in result["frames"])


def test_frontend_can_keep_overlapping_opposite_side_proposals() -> None:
    frames = [
        {
            "frame_idx": 0,
            "observations": [
                _observation(0, 0, (100, 100), handedness="left"),
                _observation(0, 1, (101, 100), handedness="right"),
            ],
        }
    ]
    result = select_pre_hamer_observations(
        frames, image_size=(640, 480), enable_consolidation=False
    )
    assert result["consolidation_enabled"] is False
    assert len(result["frames"][0]["selected_for_hamer"]) == 2


def test_frontend_does_not_modify_caller_observations() -> None:
    frames = [
        {
            "frame_idx": 0,
            "observations": [_observation(0, 4, (100, 100))],
        }
    ]
    original = deepcopy(frames)

    result = select_pre_hamer_observations(frames, image_size=(640, 480))

    assert frames == original
    assert result["frames"][0]["selected_for_hamer"][0]["candidate_id"] == 4
    assert "physical_track_id" not in frames[0]["observations"][0]


def test_frontend_preserves_missing_frames_without_inventing_hamer_input() -> None:
    frames = [
        {"frame_idx": 0, "observations": [_observation(0, 0, (100, 100))]},
        {"frame_idx": 1, "observations": []},
        {"frame_idx": 2, "observations": [_observation(2, 0, (104, 100))]},
    ]

    result = select_pre_hamer_observations(frames, image_size=(640, 480))

    assert result["frames"][1]["selected_for_hamer"] == []
    assert all(
        track["state"] == "missing" for track in result["frames"][1]["physical_tracks"]
    )
    first_track = result["frames"][0]["selected_for_hamer"][0]["physical_track_id"]
    last_track = result["frames"][2]["selected_for_hamer"][0]["physical_track_id"]
    assert first_track == last_track


def test_frontend_preserves_frame_metadata_and_original_crop_inputs() -> None:
    observation = _observation(10, 4, (100, 100))
    observation["crop"] = {"center": [100, 100], "size": 160}
    frames = [
        {"frame_idx": 10, "timestamp_ns": 1234567890, "observations": [observation]}
    ]

    result = select_pre_hamer_observations(frames, image_size=(640, 480))

    frame = result["frames"][0]
    assert frame["timestamp_ns"] == frames[0]["timestamp_ns"]
    selected = frame["selected_for_hamer"][0]
    for key, value in observation.items():
        assert selected[key] == value
    selected["bbox_xyxy"][0] = -999
    assert observation["bbox_xyxy"][0] == 70


def test_frontend_matches_existing_two_stage_functions() -> None:
    frames = [
        {
            "frame_idx": i,
            "observations": [
                _observation(i, 0, (100 + i, 100)),
                _observation(i, 1, (102 + i, 100)),
                _observation(i, 2, (400 - i, 100), handedness="right"),
            ],
        }
        for i in range(5)
    ]
    expected_input = []
    for frame in frames:
        consolidation = consolidate_hand_observations(
            frame["observations"], image_size=(640, 480)
        )
        representatives = []
        for cluster in consolidation["clusters"]:
            if not cluster["retained"]:
                continue
            representative = deepcopy(
                next(
                    row
                    for row in frame["observations"]
                    if row["candidate_id"] == cluster["representative_candidate_id"]
                )
            )
            representative["assigned_cluster_id"] = cluster["cluster_id"]
            representative["observation_quality"] = cluster["representative_quality"]
            representatives.append(representative)
        expected_input.append(
            {"frame_idx": frame["frame_idx"], "observations": representatives}
        )
    expected = associate_physical_hand_tracks(expected_input)

    actual = select_pre_hamer_observations(frames, image_size=(640, 480))

    assert actual["track_rows"] == expected["track_rows"]
    assert actual["track_statistics"] == expected["track_statistics"]
    for frame in actual["frames"]:
        selected = frame["selected_for_hamer"]
        assert len(selected) == 2
        assert len({row["physical_track_id"] for row in selected}) == 2


def test_frontend_keeps_handedness_hypothesis_without_breaking_track() -> None:
    frames = [
        {
            "frame_idx": i,
            "observations": [_observation(i, 0, (100 + i, 100), handedness=side)],
        }
        for i, side in enumerate(["left", "right", "left"])
    ]

    result = select_pre_hamer_observations(frames, image_size=(640, 480))

    selected = [frame["selected_for_hamer"][0] for frame in result["frames"]]
    assert [row["handedness"] for row in selected] == ["left", "right", "left"]
    assert len({row["physical_track_id"] for row in selected}) == 1
    assert all(row["backend_handedness"] == "left" for row in selected)
    assert all(row["backend_handedness_source"] == "track_dominant" for row in selected)


def test_frontend_handles_empty_sequence_and_gate_rejected_observation() -> None:
    assert select_pre_hamer_observations([], image_size=(640, 480))["frames"] == []
    observation = _observation(0, 0, (100, 100))
    observation["official_gate_passed"] = False

    result = select_pre_hamer_observations(
        [{"frame_idx": 0, "observations": [observation]}], image_size=(640, 480)
    )

    assert result["frames"][0]["selected_for_hamer"] == []
    assert result["summary"]["hamer_input_observation_count"] == 0


def test_frontend_rejects_duplicate_candidate_ids_before_clustering() -> None:
    observation = _observation(0, 0, (100, 100))
    with pytest.raises(ValueError, match="duplicate candidate id"):
        select_pre_hamer_observations(
            [{"frame_idx": 0, "observations": [observation, deepcopy(observation)]}],
            image_size=(640, 480),
        )


def test_frontend_requires_explicit_empty_frames_for_gaps() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        select_pre_hamer_observations(
            [
                {"frame_idx": 0, "observations": []},
                {"frame_idx": 2, "observations": []},
            ],
            image_size=(640, 480),
        )


def test_frontend_does_not_use_or_modify_cached_model_outputs() -> None:
    frames = [
        {"frame_idx": i, "observations": [_observation(i, 0, (100 + i, 100))]}
        for i in range(3)
    ]
    expected = select_pre_hamer_observations(frames, image_size=(640, 480))
    cached = deepcopy(frames)
    for i, frame in enumerate(cached):
        frame["observations"][0].update(
            {
                "mano_params": {
                    "global_orient": np.eye(3).tolist(),
                    "hand_pose": np.tile(np.eye(3), (15, 1, 1)).tolist(),
                    "betas": [float(i)] * 10,
                },
                "pred_cam": [1000.0 * i, -100.0 * i, 0.1],
                "pred_cam_t_full": [0.0, 0.0, 20.0 * i],
            }
        )
    original = deepcopy(cached)

    actual = select_pre_hamer_observations(cached, image_size=(640, 480))

    assert actual["track_rows"] == expected["track_rows"]
    assert cached == original
    for frame, source in zip(actual["frames"], original, strict=True):
        selected = frame["selected_for_hamer"][0]
        for field in ("mano_params", "pred_cam", "pred_cam_t_full"):
            assert selected[field] == source["observations"][0][field]
