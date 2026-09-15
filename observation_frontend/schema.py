"""Canonical frontend observation contract shared by every downstream stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

OBSERVATION_SCHEMA = "egohand.observations.v1"


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_hand_observation(hand: Mapping[str, Any]) -> None:
    if hand.get("handedness") not in {"left", "right"} or hand.get("backend_handedness") not in {"left", "right"}:
        raise ValueError("handedness and backend_handedness must be left or right")
    bbox = np.asarray(hand.get("bbox_xyxy"), dtype=float)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape [4]")
    _finite(bbox, "bbox_xyxy")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox_xyxy must have positive area")
    points = np.asarray(hand.get("keypoints_2d"), dtype=float)
    if points.shape != (21, 3):
        raise ValueError("keypoints_2d must have shape [21,3]")
    _finite(points, "keypoints_2d")
    if not isinstance(hand.get("physical_track_id"), (int, np.integer)):
        raise ValueError("physical_track_id must be an int")
    if not isinstance(hand.get("physical_track_fragment_id"), (int, np.integer)) or hand["physical_track_fragment_id"] < 0:
        raise ValueError("physical_track_fragment_id must be a non-negative int")


def validate_observation_frame(frame: Mapping[str, Any]) -> None:
    if not isinstance(frame.get("frame_idx"), (int, np.integer)) or not Path(frame["img_path"]).is_file():
        raise ValueError("frame_idx must be int and img_path must exist")
    hands = frame.get("hands")
    if not isinstance(hands, list):
        raise ValueError("observation frame must contain hands list")
    seen = set()
    for hand in hands:
        validate_hand_observation(hand)
        key = (int(hand["physical_track_id"]), int(hand["physical_track_fragment_id"]))
        if key in seen:
            raise ValueError(f"duplicate hand track/fragment in frame {frame['frame_idx']}")
        seen.add(key)


def validate_observation_sequence(sequence: Mapping[str, Any]) -> None:
    if sequence.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError(f"expected {OBSERVATION_SCHEMA}")
    frames = sequence.get("frames")
    if not isinstance(frames, list):
        raise ValueError("observation sequence frames must be a list")
    indexes = [int(frame.get("frame_idx", -1)) for frame in frames]
    if indexes != list(range(len(frames))):
        raise ValueError("observation frame_idx must be unique, consecutive, and ordered")
    for frame in frames:
        validate_observation_frame(frame)
    sequence_meta = sequence.get("sequence", {})
    if int(sequence_meta.get("frame_count", -1)) != len(frames):
        raise ValueError("sequence frame_count does not match frames")


def canonical_sequence(frames: list[Mapping[str, Any]], *, sequence_name: str, fps: float,
                       frontend: str, version: str | None = None) -> dict[str, Any]:
    canonical_frames = []
    for frame in frames:
        hands = []
        for hand in frame.get("hands", frame.get("selected_for_hamer", [])):
            meta = dict(hand.get("observation_meta", hand.get("meta", {})))
            hands.append({"observation_id": f"{frame['frame_idx']}:{hand['handedness']}:{hand['physical_track_id']}",
                "handedness": hand["handedness"], "backend_handedness": hand.get("backend_handedness", hand["handedness"]),
                "bbox_xyxy": list(hand["bbox_xyxy"]), "keypoints_2d": hand.get("keypoints_2d", hand.get("vitpose_keypoints_2d")),
                "physical_track_id": int(hand["physical_track_id"]), "physical_track_fragment_id": int(hand.get("physical_track_fragment_id", 0)),
                "confidence": hand.get("confidence", meta.get("hand_presence")), "source": meta.get("source", frontend), "meta": meta})
        canonical_frames.append({"frame_idx": int(frame["frame_idx"]), "img_path": str(frame["img_path"]),
            "timestamp_ns": frame.get("timestamp_ns"), "hands": hands})
    paths = [frame["img_path"] for frame in canonical_frames]
    if canonical_frames:
        import cv2
        image = cv2.imread(paths[0]); height, width = image.shape[:2]
    else:
        height = width = 0
    result = {"schema_version": OBSERVATION_SCHEMA,
        "sequence": {"sequence_name": sequence_name, "frame_count": len(canonical_frames), "fps": float(fps), "image_width": width, "image_height": height},
        "frontend": {"name": frontend, "version": version}, "frames": canonical_frames}
    validate_observation_sequence(result)
    return result
