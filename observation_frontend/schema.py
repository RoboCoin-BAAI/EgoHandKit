"""Canonical frontend observation contract shared by every downstream stage."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle
from typing import Any, Mapping

import cv2
import numpy as np

OBSERVATION_SCHEMA = "egohand.observations.v1"
CAMERA_JOINT_FRAME = "opencv_x_right_y_down_z_forward"
CAMERA_JOINT_ORDER = "openpose21"


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def camera_joints_from_observation(reference: Mapping[str, Any] | None) -> np.ndarray | None:
    """Read supported camera-space 21-joint metadata without guessing conventions."""
    if reference is None:
        return None
    candidates: list[Mapping[str, Any]] = [reference]
    for key in ("meta", "observation_meta"):
        nested = reference.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for candidate in candidates:
        if (candidate.get("camera_frame") != CAMERA_JOINT_FRAME
                or candidate.get("joint_order") != CAMERA_JOINT_ORDER):
            continue
        for key in ("joints_3d_camera", "mint_joints_3d_camera", "joints_3d_cam", "joints_cam"):
            value = candidate.get(key)
            if value is None:
                continue
            joints = np.asarray(value, dtype=np.float64)
            if joints.shape == (21, 3) and np.isfinite(joints).all():
                return joints
    return None


def validate_hand_observation(hand: Mapping[str, Any]) -> None:
    if not isinstance(hand, Mapping):
        raise ValueError("hand observation must be a mapping")
    if not isinstance(hand.get("observation_id"), str) or not hand["observation_id"]:
        raise ValueError("observation_id must be a non-empty string")
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
    confidence = hand.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float, np.integer, np.floating)) or not np.isfinite(confidence):
            raise ValueError("confidence must be a finite number or null")
    if not isinstance(hand.get("source"), str) or not hand["source"]:
        raise ValueError("source must be a non-empty string")
    if not isinstance(hand.get("meta"), Mapping):
        raise ValueError("meta must be a mapping")


def validate_observation_frame(frame: Mapping[str, Any]) -> None:
    if not isinstance(frame, Mapping):
        raise ValueError("observation frame must be a mapping")
    if not isinstance(frame.get("frame_idx"), (int, np.integer)):
        raise ValueError("frame_idx must be an int")
    if not isinstance(frame.get("img_path"), str) or not frame["img_path"]:
        raise ValueError("img_path must be a non-empty string")
    timestamp = frame.get("timestamp_ns")
    if timestamp is not None and not isinstance(timestamp, (int, np.integer)):
        raise ValueError("timestamp_ns must be an int or null")
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
    if not isinstance(sequence, Mapping):
        raise ValueError("observation sequence must be a mapping")
    if sequence.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError(
            f"expected schema_version {OBSERVATION_SCHEMA!r}, got {sequence.get('schema_version')!r}"
        )
    sequence_meta = sequence.get("sequence")
    if not isinstance(sequence_meta, Mapping):
        raise ValueError("observation sequence metadata must be a mapping")
    if not isinstance(sequence_meta.get("sequence_name"), str) or not sequence_meta["sequence_name"]:
        raise ValueError("sequence_name must be a non-empty string")
    if not isinstance(sequence_meta.get("frame_count"), (int, np.integer)):
        raise ValueError("sequence frame_count must be an int")
    fps = sequence_meta.get("fps")
    if not isinstance(fps, (int, float, np.integer, np.floating)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("sequence fps must be a positive finite number")
    for field in ("image_width", "image_height"):
        if not isinstance(sequence_meta.get(field), (int, np.integer)) or sequence_meta[field] <= 0:
            raise ValueError(f"sequence {field} must be a positive int")
    frontend = sequence.get("frontend")
    if not isinstance(frontend, Mapping) or not isinstance(frontend.get("name"), str) or not frontend["name"]:
        raise ValueError("frontend.name must be a non-empty string")
    if frontend.get("version") is not None and not isinstance(frontend["version"], str):
        raise ValueError("frontend.version must be a string or null")
    frames = sequence.get("frames")
    if not isinstance(frames, list):
        raise ValueError("observation sequence frames must be a list")
    for frame in frames:
        validate_observation_frame(frame)
    indexes = [int(frame["frame_idx"]) for frame in frames]
    if indexes != list(range(len(frames))):
        raise ValueError("observation frame_idx must be unique, consecutive, and ordered")
    if int(sequence_meta.get("frame_count", -1)) != len(frames):
        raise ValueError("sequence frame_count does not match frames")


def _validate_serializable(value: Any, location: str = "sequence") -> None:
    """Reject runtime objects while allowing the contract's plain values."""
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError(f"{location} contains an object-dtype NumPy array")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_serializable(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{location} contains a non-string dictionary key")
        for key, item in value.items():
            _validate_serializable(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} contains unsupported runtime type {type(value).__name__}")


def save_observation_sequence(sequence: Mapping[str, Any], path: str | Path) -> None:
    """Save a validated canonical artifact containing data values only."""
    validate_observation_sequence(sequence)
    _validate_serializable(sequence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(dict(sequence), protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def load_observation_sequence(path: str | Path) -> dict[str, Any]:
    """Load and validate an ``egohand.observations.v1`` pickle artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical observation artifact does not exist: {path}")
    try:
        sequence = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Cannot load canonical observation artifact {path}: {exc}") from exc
    validate_observation_sequence(sequence)
    _validate_serializable(sequence)
    return sequence


def remap_observation_sequence(
    sequence: Mapping[str, Any],
    image_paths: list[str | Path],
    *,
    fps: float,
    sequence_name: str | None = None,
) -> dict[str, Any]:
    """Validate against the current input and remap paths using ``frame_idx``."""
    validate_observation_sequence(sequence)
    frames = sequence["frames"]
    if len(frames) != len(image_paths):
        raise ValueError(
            "Canonical observation frame count does not match current input: "
            f"artifact={len(frames)}, input={len(image_paths)}"
        )
    expected_fps = float(sequence["sequence"]["fps"])
    if not np.isclose(expected_fps, float(fps), rtol=1e-3, atol=1e-3):
        raise ValueError(
            "Canonical observation FPS does not match current input: "
            f"artifact={expected_fps}, input={float(fps)}"
        )
    expected_size = (
        int(sequence["sequence"]["image_height"]),
        int(sequence["sequence"]["image_width"]),
    )
    current_paths = [Path(path) for path in image_paths]
    for frame_idx, path in enumerate(current_paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read current input frame {frame_idx}: {path}")
        if image.shape[:2] != expected_size:
            raise ValueError(
                "Canonical observation image size does not match current input at "
                f"frame {frame_idx}: artifact={expected_size[::-1]}, "
                f"input={(image.shape[1], image.shape[0])}"
            )

    # Validation above proves that frame_idx is the complete ordered identity
    # 0..T-1. Paths are deliberately replaced rather than compared literally,
    # because extracted-frame directories commonly change between runs.
    remapped = deepcopy(dict(sequence))
    for frame, path in zip(remapped["frames"], current_paths):
        frame["img_path"] = str(path)
    if sequence_name is not None:
        remapped["sequence"]["sequence_name"] = sequence_name
    validate_observation_sequence(remapped)
    return remapped


def canonical_sequence(frames: list[Mapping[str, Any]], *, sequence_name: str, fps: float,
                       frontend: str, version: str | None = None) -> dict[str, Any]:
    canonical_frames = []
    for frame in frames:
        hands = []
        source_hands = frame.get("hands", frame.get("selected_for_hamer"))
        if source_hands is None:
            source_hands = []
            for track_id, side in enumerate(("left", "right")):
                bbox = frame.get(f"{side}_raw_bbox")
                if bbox is None:
                    bbox = frame.get(f"{side}_bbox")
                if bbox is None:
                    continue
                keypoints = frame.get(f"{side}_keypoints")
                if keypoints is None:
                    keypoints = np.zeros((21, 3), dtype=np.float32)
                source_hands.append({
                    "handedness": side,
                    "backend_handedness": side,
                    "bbox_xyxy": bbox,
                    "keypoints_2d": keypoints,
                    "physical_track_id": track_id,
                    "physical_track_fragment_id": 0,
                    "confidence": frame.get(f"{side}_conf"),
                    "meta": {"source": frontend},
                })
        for hand in source_hands:
            meta = dict(hand.get("observation_meta", hand.get("meta", {})))
            hands.append({"observation_id": hand.get("observation_id", f"{frame['frame_idx']}:{hand['handedness']}:{hand['physical_track_id']}"),
                "handedness": hand["handedness"], "backend_handedness": hand.get("backend_handedness", hand["handedness"]),
                "bbox_xyxy": list(hand["bbox_xyxy"]), "keypoints_2d": hand.get("keypoints_2d", hand.get("vitpose_keypoints_2d")),
                "physical_track_id": int(hand["physical_track_id"]), "physical_track_fragment_id": int(hand.get("physical_track_fragment_id", 0)),
                "confidence": hand.get("confidence", meta.get("hand_presence")), "source": meta.get("source", frontend), "meta": meta})
        canonical_frames.append({"frame_idx": int(frame["frame_idx"]), "img_path": str(frame["img_path"]),
            "timestamp_ns": frame.get("timestamp_ns"), "hands": hands})
    paths = [frame["img_path"] for frame in canonical_frames]
    if canonical_frames:
        image = cv2.imread(paths[0]); height, width = image.shape[:2]
    else:
        height = width = 0
    result = {"schema_version": OBSERVATION_SCHEMA,
        "sequence": {"sequence_name": sequence_name, "frame_count": len(canonical_frames), "fps": float(fps), "image_width": width, "image_height": height},
        "frontend": {"name": frontend, "version": version}, "frames": canonical_frames}
    validate_observation_sequence(result)
    return result
