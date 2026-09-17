#!/usr/bin/env python3
"""Convert an ACE-Ego-Hand prediction pickle to EgoHandKit's canonical format.

The output is an ``egohand.observations.v1`` artifact and can be passed directly
to EgoHandKit with ``run.py --frontend canonical --observations ...``.  ACE
predictions contain camera-frame direct joints, so this conversion does not need
to import the model (or MANO) at all.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OBSERVATION_SCHEMA = "egohand.observations.v1"
HANDS = ("left", "right")
MIN_VALID_DEPTH = 1e-6


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="source video")
    p.add_argument("--raw", "--prediction", dest="raw", type=Path, required=True,
                   help="pickle written by infer_video.py")
    p.add_argument("--out", type=Path, required=True,
                   help="canonical observation pickle to write")
    p.add_argument("--bbox_scale", type=float, default=1.0,
                   help="square padding around projected joints")
    p.add_argument("--presence_threshold", type=float, default=0.5,
                   help="minimum exists_3d probability to emit a hand")
    return p.parse_args()


def _video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n <= 0 or w <= 0 or h <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Source video has unusable metadata: {path}")
    return n, w, h, fps


def _intrinsics(pred: dict[str, Any]) -> dict[str, float]:
    # K-free inference writes the camera fitted from its predicted ray field.
    # For K-given runs pred_intrinsics is None and the supplied K is correct.
    k = pred.get("pred_intrinsics") or pred.get("intrinsics")
    if not isinstance(k, dict):
        raise ValueError("prediction has neither intrinsics nor pred_intrinsics")
    required = ("fx", "fy", "cx", "cy", "image_width", "image_height")
    try:
        out = {name: float(k[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid prediction intrinsics: {k!r}") from exc
    if out["fx"] <= 0 or out["fy"] <= 0 or out["image_width"] <= 0 or out["image_height"] <= 0:
        raise ValueError(f"invalid prediction intrinsics: {k!r}")
    return out


def _bbox(points: np.ndarray, scale: float, width: int, height: int) -> list[float]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = max(float(np.max(hi - lo)) * scale, 1.0)
    center = (lo + hi) / 2.0
    half = size / 2.0
    x1, y1 = np.maximum(center - half, 0)
    x2, y2 = np.minimum(center + half, [width - 1, height - 1])
    return [float(x1), float(y1), float(x2), float(y2)]


def _project(joints: np.ndarray, k: dict[str, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(joints).all(axis=-1) & (joints[:, 2] > MIN_VALID_DEPTH)
    uv = np.full((len(joints), 2), np.nan, np.float32)
    sx, sy = width / k["image_width"], height / k["image_height"]
    uv[valid, 0] = (k["fx"] * joints[valid, 0] / joints[valid, 2] + k["cx"]) * sx
    uv[valid, 1] = (k["fy"] * joints[valid, 1] / joints[valid, 2] + k["cy"]) * sy
    return uv, valid


def validate_sequence(sequence: dict[str, Any]) -> None:
    """Validate the structural portion consumed by EgoHandKit's canonical frontend."""
    frames = sequence.get("frames", [])
    if [f.get("frame_idx") for f in frames] != list(range(len(frames))):
        raise ValueError("frame_idx must be consecutive zero-based indices")
    if sequence.get("sequence", {}).get("frame_count") != len(frames):
        raise ValueError("sequence frame_count does not match frames")
    for frame in frames:
        for hand in frame.get("hands", []):
            bbox = np.asarray(hand.get("bbox_xyxy"), dtype=float)
            points = np.asarray(hand.get("keypoints_2d"), dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"invalid bbox in frame {frame['frame_idx']}")
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise ValueError(f"invalid keypoints in frame {frame['frame_idx']}")
            if not np.isfinite(float(hand.get("confidence", np.nan))):
                raise ValueError(f"invalid confidence in frame {frame['frame_idx']}")


def convert(source: Path, raw: Path, out: Path, bbox_scale: float = 1.0,
            presence_threshold: float = 0.5) -> dict[str, Any]:
    if not source.is_file() or not raw.is_file():
        raise FileNotFoundError(f"missing source or prediction: {source}, {raw}")
    video_frames, width, height, fps = _video_info(source)
    with raw.open("rb") as f:
        pred = pickle.load(f)
    if not isinstance(pred, dict):
        raise ValueError("prediction pickle must contain a mapping")
    joints = np.asarray(pred.get("direct_joints_cam", pred.get("joints_cam_direct")), dtype=np.float32)
    presence = np.asarray(pred.get("exists_3d"), dtype=np.float32)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"direct_joints_cam must have shape (F,2,21,3), got {joints.shape}")
    if presence.shape != joints.shape[:2]:
        raise ValueError(f"exists_3d must have shape {joints.shape[:2]}, got {presence.shape}")
    if len(joints) > video_frames:
        raise ValueError(
            f"prediction has {len(joints)} frames but source video has only {video_frames}"
        )
    k = _intrinsics(pred)
    frames = []
    emitted = {side: 0 for side in HANDS}
    for i in range(len(joints)):
        records = []
        for slot, side in enumerate(HANDS):
            prob = float(presence[i, slot])
            uv, valid = _project(joints[i, slot], k, width, height)
            inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
            if prob < presence_threshold or not inside.any():
                continue
            points = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
            confidence = np.where(valid, prob, 0.0).astype(np.float32)
            records.append({
                "observation_id": f"{i}:{side}", "handedness": side,
                "backend_handedness": side, "bbox_xyxy": _bbox(uv[inside], bbox_scale, width, height),
                # Store plain lists rather than numpy arrays.  EgoHandKit is
                # often run in a separate conda environment with a different
                # numpy major version; numpy arrays in a pickle can otherwise
                # reference version-specific modules (for example
                # ``numpy._core.numeric``).
                "keypoints_2d": np.column_stack((points, confidence)).astype(np.float32).tolist(),
                "physical_track_id": slot, "physical_track_fragment_id": 0,
                "confidence": prob, "source": "ace-ego-hand",
                "meta": {"source": "ace-ego-hand", "hand_presence": prob,
                         "confidence_type": "ace_exists_3d_projected", "joint_order": "openpose21",
                         "camera_frame": "opencv_x_right_y_down_z_forward"},
            })
            emitted[side] += 1
        frames.append({"frame_idx": i, "img_path": f"{source.resolve()}#frame={i}",
                       "timestamp_ns": None, "hands": records})
    result = {"schema_version": OBSERVATION_SCHEMA,
              "sequence": {"sequence_name": source.stem, "frame_count": len(frames),
                            "fps": fps, "image_width": width, "image_height": height},
              "frontend": {"name": "ace-ego-hand", "version": "1.0"}, "frames": frames}
    validate_sequence(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(out)
    print(f"Wrote {out}: {len(frames)} frames, left={emitted['left']}, right={emitted['right']}")
    return result


if __name__ == "__main__":
    a = _args()
    if a.bbox_scale <= 0 or not 0 <= a.presence_threshold <= 1:
        raise SystemExit("bbox_scale must be positive and presence_threshold must lie in [0, 1]")
    convert(a.source, a.raw, a.out, a.bbox_scale, a.presence_threshold)