#!/usr/bin/env python3
"""Convert external MANO arrays into EgoHandKit canonical observations.

The input format is the ``front_output`` package format: one structured
``left.npy`` or ``right.npy`` file per camera, with two hands per frame.  The
converter reconstructs MANO joints, projects them with the stored intrinsics,
and writes ``egohand.observations.v1``.  EgoHandKit will then run its selected
HMR backend on the generated boxes/keypoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow direct execution as ``python tools/<script>.py`` from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_frontend.schema import save_observation_sequence


def _load_external_model(front_output: Path):
    module_path = front_output / "mano_model.py"
    spec = importlib.util.spec_from_file_location("external_front_mano_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load external MANO model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANOModel(front_output / "models")


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError(f"Expected finite joints with shape (21,3), got {points.shape}")
    depth = points[:, 2]
    if np.any(depth <= 1e-6):
        raise ValueError("MANO joints contain non-positive camera depth")
    uv = points[:, :2] / depth[:, None]
    uv = uv @ np.asarray(intrinsic, dtype=np.float64)[:2, :2].T
    uv += np.asarray(intrinsic, dtype=np.float64)[:2, 2]
    return np.concatenate([uv, np.ones((21, 1), dtype=np.float64)], axis=1)


def _padded_bbox(keypoints: np.ndarray, width: int, height: int, scale: float) -> list[float]:
    xy = keypoints[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo + hi) / 2.0
    side = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 2.0) * scale
    half = side / 2.0
    bbox = np.array([center[0] - half, center[1] - half, center[0] + half, center[1] + half])
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"Projected hand bbox has no positive area: {bbox.tolist()}")
    return bbox.astype(float).tolist()


def _frame_paths(images: Path, expected_count: int) -> list[Path]:
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != expected_count:
        raise ValueError(f"Image count {len(paths)} does not match MANO frames {expected_count}: {images}")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Cannot read image: {paths[0]}")
    shape = first.shape[:2]
    for path in paths[1:]:
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != shape:
            raise ValueError(f"Images must be readable and have consistent dimensions: {path}")
    return paths


def _extract_video(video: Path, frames_dir: Path) -> tuple[list[Path], float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if existing:
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        return existing, fps
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"Cannot read FPS from video: {video}")
    paths = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        path = frames_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), image):
            capture.release()
            raise IOError(f"Cannot write extracted frame: {path}")
        paths.append(path)
        index += 1
    capture.release()
    if not paths:
        raise ValueError(f"Video contains no frames: {video}")
    return paths, fps


def convert(
    parameter_path: Path,
    image_dir: Path | None,
    output_path: Path,
    *,
    front_output: Path,
    sequence_name: str,
    fps: float,
    bbox_scale: float,
    image_size: tuple[int, int] | None = None,
    chunk_size: int = 64,
) -> dict[str, Any]:
    parameters = np.load(parameter_path, allow_pickle=False)
    required = {"global_orient", "hand_pose", "betas", "translation", "valid", "intrinsics", "frame_index"}
    missing = required.difference(parameters.dtype.names or ())
    if missing:
        raise ValueError(f"Missing fields in {parameter_path}: {sorted(missing)}")
    if parameters.ndim != 2 or parameters.shape[1:] != (2,):
        raise ValueError(f"Expected parameter shape (frames, 2), got {parameters.shape}")
    frame_count = len(parameters)
    if image_dir is None:
        if image_size is None:
            raise ValueError("image_size is required when no image directory is supplied")
        width, height = image_size
        paths = [Path(f"frame_{index:06d}.jpg") for index in range(frame_count)]
    else:
        paths = _frame_paths(image_dir, frame_count)
        first = cv2.imread(str(paths[0]))
        height, width = first.shape[:2]
    if not np.array_equal(parameters["frame_index"][:, 0], np.arange(frame_count)):
        raise ValueError("frame_index must be consecutive and zero-based")
    if not np.array_equal(parameters["frame_index"][:, 1], np.arange(frame_count)):
        raise ValueError("both hands must use the same consecutive frame_index")

    model = _load_external_model(front_output)
    joints = np.empty((frame_count, 2, 21, 3), dtype=np.float64)
    for start in range(0, frame_count, chunk_size):
        stop = min(frame_count, start + chunk_size)
        _, joints[start:stop] = model.reconstruct(parameters[start:stop])

    side_names = ("left", "right")
    fragments = [0, 0]
    was_valid = [False, False]
    frames = []
    for frame_idx, path in enumerate(paths):
        hands = []
        for hand_idx, side in enumerate(side_names):
            valid = bool(parameters["valid"][frame_idx, hand_idx])
            if not valid:
                was_valid[hand_idx] = False
                continue
            if not was_valid[hand_idx] and frame_idx > 0:
                fragments[hand_idx] += 1
            was_valid[hand_idx] = True
            try:
                keypoints = _project(joints[frame_idx, hand_idx], parameters["intrinsics"][frame_idx, hand_idx])
                bbox = _padded_bbox(keypoints, width, height, bbox_scale)
            except ValueError:
                was_valid[hand_idx] = False
                continue
            hands.append({
                "observation_id": f"{side}-{frame_idx}-{fragments[hand_idx]}",
                "handedness": side,
                "backend_handedness": side,
                "bbox_xyxy": bbox,
                "keypoints_2d": keypoints.astype(np.float32),
                "physical_track_id": hand_idx,
                "physical_track_fragment_id": fragments[hand_idx],
                "confidence": 1.0,
                "source": "external_mano",
                "meta": {
                    "parameter_path": str(parameter_path),
                    "camera_intrinsics": np.asarray(parameters["intrinsics"][frame_idx, hand_idx]).tolist(),
                    "joints_3d_camera": joints[frame_idx, hand_idx].astype(np.float32).tolist(),
                    "camera_frame": "opencv_x_right_y_down_z_forward",
                    "joint_order": "openpose21",
                    "external_valid": True,
                },
            })
        timestamp = None
        if "timestamp" in (parameters.dtype.names or ()):
            # The external array stores one timestamp per hand; both entries
            # should refer to the same video frame. Use the first and reject
            # only non-finite values rather than requiring identical dtypes.
            timestamp_value = float(parameters["timestamp"][frame_idx, 0])
            timestamp = timestamp_value if np.isfinite(timestamp_value) else None
        timestamp_ns = int(round(timestamp * 1_000_000_000)) if timestamp is not None and np.isfinite(timestamp) else None
        frames.append({"frame_idx": frame_idx, "img_path": str(path), "timestamp_ns": timestamp_ns, "hands": hands})

    sequence = {
        "schema_version": "egohand.observations.v1",
        "sequence": {"sequence_name": sequence_name, "frame_count": frame_count, "fps": float(fps),
                      "image_width": width, "image_height": height},
        "frontend": {"name": "external_mano", "version": "1.0"},
        "frames": frames,
    }
    save_observation_sequence(sequence, output_path)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-output", type=Path, required=True,
                        help="Directory containing mano_model.py, models/, and sequences/")
    parser.add_argument("--parameter", type=Path, required=True,
                        help="External structured array, e.g. sequences/<session>/right.npy")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--images", type=Path, help="Extracted image directory in frame order")
    source.add_argument("--video", type=Path, help="Video corresponding to the parameter array")
    parser.add_argument("--frames-output", type=Path,
                        help="Directory for frames extracted from --video (default: <output>.frames)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output egohand.observations.v1 pickle")
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS for image folders; videos use native FPS unless overridden")
    parser.add_argument("--bbox-scale", type=float, default=1.2)
    parser.add_argument("--image-width", type=int,
                        help="Original image width when converting without --images/--video")
    parser.add_argument("--image-height", type=int,
                        help="Original image height when converting without --images/--video")
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0 or args.bbox_scale <= 0:
        parser.error("--fps (when provided) and --bbox-scale must be positive")
    if not args.images and not args.video:
        if args.image_width is None or args.image_height is None or args.fps is None:
            parser.error("without --images/--video, provide --image-width, --image-height and --fps")
        if args.image_width <= 0 or args.image_height <= 0:
            parser.error("--image-width and --image-height must be positive")
        image_dir = None
        fps = args.fps
        image_size = (args.image_width, args.image_height)
    elif args.video:
        frames_dir = args.frames_output or args.output.with_suffix("").with_name(args.output.stem + "_frames")
        extracted, native_fps = _extract_video(args.video, frames_dir)
        if len(extracted) != len(np.load(args.parameter, allow_pickle=False)):
            raise ValueError("Video frame count does not match parameter frame count")
        image_dir = frames_dir
        fps = native_fps if args.fps is None else args.fps
        image_size = None
    else:
        image_dir = args.images
        fps = 30.0 if args.fps is None else args.fps
        image_size = None
    convert(args.parameter, image_dir, args.output, front_output=args.front_output,
            sequence_name=args.sequence_name, fps=fps, bbox_scale=args.bbox_scale,
            image_size=image_size)
    print(f"Saved canonical observations to {args.output}")


if __name__ == "__main__":
    main()
