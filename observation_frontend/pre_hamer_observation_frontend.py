from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from observation_frontend.hand_observation_consolidation import (
    DEFAULT_CONFIG as DEFAULT_CONSOLIDATION_CONFIG,
)
from observation_frontend.hand_observation_consolidation import (
    HandObservationConsolidationConfig,
    consolidate_hand_observations,
    observation_quality,
)
from observation_frontend.physical_hand_temporal_association import (
    DEFAULT_CONFIG as DEFAULT_ASSOCIATION_CONFIG,
)
from observation_frontend.physical_hand_temporal_association import (
    PhysicalHandTemporalAssociationConfig,
    associate_physical_hand_tracks,
)


def _candidate_id(observation: Mapping[str, Any]) -> int:
    value = observation.get(
        "original_candidate_id",
        observation.get("official_candidate_id", observation.get("candidate_id")),
    )
    if value is None:
        raise ValueError("hand observation is missing a candidate id")
    return int(value)


def select_pre_hamer_observations(
    frames: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    consolidation_config: HandObservationConsolidationConfig = (
        DEFAULT_CONSOLIDATION_CONFIG
    ),
    association_config: PhysicalHandTemporalAssociationConfig = (
        DEFAULT_ASSOCIATION_CONFIG
    ),
    enable_consolidation: bool = True,
) -> dict[str, Any]:
    """Select offline physical-hand observations before HaMeR inference.

    Each input frame must contain a sorted, consecutive ``frame_idx`` and an
    ``observations`` list. Each observation supplies a candidate id, ``bbox_xyxy``,
    ``vitpose_keypoints_2d`` with shape ``(21, 3)``, and optional handedness and
    person confidence. Candidate ids must be unique within each frame. Frame
    metadata (including timestamps) is preserved, but association uses frame
    distance, not elapsed time. The function does not mutate caller-owned data.
    """

    consolidated_frames: list[dict[str, Any]] = []
    input_observation_count = 0
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        observations = list(frame.get("observations", []))
        candidate_ids = [_candidate_id(observation) for observation in observations]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate id in frame {frame_idx}")
        input_observation_count += len(observations)
        if enable_consolidation:
            result = consolidate_hand_observations(
                observations,
                image_size=image_size,
                config=consolidation_config,
            )
        else:
            # The legacy cleanup and the physical association are already
            # temporal filters.  Running same-frame consolidation as well can
            # erase a genuine left/right pair when their boxes overlap.  Keep
            # every gated proposal in the default wearer-first path and leave
            # consolidation available for explicit all-person experiments.
            selected = [
                deepcopy(observation)
                for observation in observations
                if bool(observation.get("official_gate_passed", True))
                and observation.get("bbox_xyxy") is not None
            ]
            diagnostics = []
            clusters = []
            for cluster_id, observation in enumerate(selected):
                candidate_id = _candidate_id(observation)
                quality = observation_quality(
                    observation, image_size=image_size,
                    config=consolidation_config,
                )
                observation["representative_candidate_id"] = candidate_id
                observation["assigned_cluster_id"] = cluster_id
                observation["observation_quality"] = deepcopy(quality)
                clusters.append({
                    "cluster_id": cluster_id,
                    "member_candidate_ids": [candidate_id],
                    "representative_candidate_id": candidate_id,
                    "cluster_size": 1,
                    "retained": True,
                    "retention_rank": cluster_id,
                    "mixed_handedness": False,
                })
                diagnostics.append({
                    "frame_idx": frame_idx,
                    "original_candidate_id": candidate_id,
                    "assigned_cluster_id": cluster_id,
                    "cluster_retained": True,
                    "selected_as_cluster_representative": True,
                    "duplicate_of_candidate_id": None,
                    "observation_quality": deepcopy(quality),
                })
            result = {
                "selected_observations": selected,
                "observation_diagnostics": diagnostics,
                "clusters": clusters,
                "pairwise_metrics": [],
                "raw_observation_count": len(selected),
                "pre_cap_cluster_count": len(selected),
                "physical_hand_cluster_count": len(selected),
                "selected_candidate_ids": [
                    _candidate_id(observation) for observation in selected
                ],
            }
        diagnostics_by_id = {
            int(row["original_candidate_id"]): row
            for row in result["observation_diagnostics"]
        }
        representatives: list[dict[str, Any]] = []
        for observation in result["selected_observations"]:
            candidate_id = _candidate_id(observation)
            diagnostic = diagnostics_by_id[candidate_id]
            representative = deepcopy(observation)
            representative.update(
                {
                    "representative_candidate_id": candidate_id,
                    "assigned_cluster_id": int(diagnostic["assigned_cluster_id"]),
                    "observation_quality": deepcopy(diagnostic["observation_quality"]),
                }
            )
            representatives.append(representative)
        consolidated_frames.append(
            {
                **deepcopy(
                    {
                        key: value
                        for key, value in frame.items()
                        if key != "observations"
                    }
                ),
                "frame_idx": frame_idx,
                "observations": representatives,
                "raw_observation_count": result["raw_observation_count"],
                "physical_hand_cluster_count": result["physical_hand_cluster_count"],
                "clusters": result["clusters"],
                "pairwise_metrics": result["pairwise_metrics"],
                "observation_diagnostics": result["observation_diagnostics"],
            }
        )

    association = associate_physical_hand_tracks(
        consolidated_frames,
        config=association_config,
    )
    association_by_frame = {
        int(frame["frame_idx"]): frame for frame in association["frames"]
    }
    statistics_by_track = {
        int(stat["track_id"]): stat for stat in association["track_statistics"]
    }
    output_frames: list[dict[str, Any]] = []
    hamer_input_count = 0
    for frame in consolidated_frames:
        frame_idx = int(frame["frame_idx"])
        associated = association_by_frame[frame_idx]
        tracks_by_candidate = {
            int(track["selected_candidate_id"]): track
            for track in associated["tracks"]
            if track["state"] == "observed"
        }
        selected_for_hamer: list[dict[str, Any]] = []
        for observation in frame["observations"]:
            candidate_id = _candidate_id(observation)
            track = tracks_by_candidate.get(candidate_id)
            if track is None:
                continue
            selected = deepcopy(observation)
            selected.update(
                {
                    "physical_track_id": int(track["track_id"]),
                    "physical_track_fragment_id": int(track["track_fragment_id"]),
                    "dominant_track_handedness": track["dominant_track_handedness"],
                }
            )
            # The detector's anatomical-side label can flicker for a frame or
            # two.  Keep that raw hypothesis for diagnostics, but provide the
            # backend with one track-level side.  HaWoR's crop flip is a
            # segment-level decision; allowing raw ViTPose sides to drive it
            # creates a left/right flip every time the frontend changes its
            # mind.  Physical identity remains ``physical_track_id``; this is
            # only the backend crop convention.
            stat = statistics_by_track[int(track["track_id"])]
            observed_count = int(stat["observed_frames"])
            if observed_count > 0 and track["dominant_track_handedness"] in ("left", "right"):
                selected["backend_handedness"] = track["dominant_track_handedness"]
                selected["backend_handedness_source"] = "track_dominant"
            selected_for_hamer.append(selected)
        selected_for_hamer.sort(key=lambda item: int(item["physical_track_id"]))
        hamer_input_count += len(selected_for_hamer)
        output_frames.append(
            {
                **frame,
                "selected_for_hamer": selected_for_hamer,
                "selected_candidate_ids": [
                    _candidate_id(observation) for observation in selected_for_hamer
                ],
                "physical_tracks": associated["tracks"],
                "unassigned_observations": associated["unassigned_observations"],
            }
        )

    consolidated_count = sum(
        len(frame["observations"]) for frame in consolidated_frames
    )
    return {
        "schema_version": "1.0",
        "mode": "offline_pre_hamer_observation_frontend",
        "requires_hamer_outputs": False,
        "requires_full_sequence": True,
        "image_size": list(image_size),
        "consolidation_config": consolidation_config.as_dict(),
        "consolidation_enabled": bool(enable_consolidation),
        "temporal_association_config": association_config.as_dict(),
        "frames": output_frames,
        "track_rows": association["track_rows"],
        "track_statistics": association["track_statistics"],
        "unassigned_observations": association["unassigned_observations"],
        "summary": {
            **association["summary"],
            "frame_count": len(output_frames),
            "input_observation_count": input_observation_count,
            "consolidated_observation_count": consolidated_count,
            "hamer_input_observation_count": hamer_input_count,
        },
    }
