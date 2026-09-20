
"""Export a MINT prediction as an EgoHandKit canonical observation artifact.

The output is an ``egohand.observations.v1`` pickle consumable by::

    python run.py --input video.mp4 --frontend canonical \
        --observations video.observations.pkl

The raw MINT prediction must contain:

- ``pose_enc`` with shape ``[T, 9]``; columns 7/8 are vertical/horizontal FOV
- ``hand`` with shape ``[T, 218]``
- ``hand_presence_logits`` with shape ``[T, 2]``

Projection uses the FOV stored in ``pose_enc`` directly in normalized image
coordinates. Therefore no MINT YAML/config or model input resolution is needed.
This assumes the original MINT image geometry is resize-only; crop/pad/letterbox
preprocessing would require its explicit inverse transform.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

# Keep this script under <wuji-ego-mint>/scripts/ so the MINT source tree can be
# imported without installing the package into the EgoHandKit environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAIN_ROOT = REPO_ROOT / "model_train"
for path in (REPO_ROOT, MODEL_TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import numpy as np

from mint.visualization import mano
from mint.visualization.render import prediction_to_hands

OBSERVATION_SCHEMA = "egohand.observations.v1"
HANDS = ("left", "right")
MIN_VALID_DEPTH = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source video")
    parser.add_argument("--raw", type=Path, required=True, help="Raw MINT prediction.npz")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output canonical observation artifact (.pkl)",
    )
    parser.add_argument(
        "--bbox_scale",
        type=float,
        default=1.0,
        help="Square padding applied to projected joints before HMR",
    )
    parser.add_argument(
        "--presence_threshold",
        type=float,
        default=0.5,
        help="Minimum sigmoid hand-presence probability to emit an observation",
    )
    return parser.parse_args()


def read_video_metadata(source: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read source video metadata: {source}")
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Source video reports no usable frame rate: {source}")

    return frame_count, width, height, fps


def load_raw_prediction(
    raw: Path,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pose encoding, MANO prediction and sigmoid hand presence."""
    with raw.open("rb") as handle:
        magic = handle.read(2)
    if magic != b"PK":
        raise ValueError(f"Raw prediction is not a valid .npz archive: {raw}")

    with np.load(raw, allow_pickle=False) as archive:
        required = {"pose_enc", "hand", "hand_presence_logits"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Raw prediction is missing: {sorted(missing)}")

        pose_enc = np.asarray(archive["pose_enc"], dtype=np.float32)
        hand = np.asarray(archive["hand"], dtype=np.float32)
        presence_logits = np.asarray(
            archive["hand_presence_logits"], dtype=np.float32
        )

    if pose_enc.shape != (frame_count, 9):
        raise ValueError(
            f"pose_enc shape {pose_enc.shape} does not match "
            f"source frame count {frame_count}"
        )
    if hand.shape != (frame_count, 218):
        raise ValueError(
            f"hand shape {hand.shape} does not match "
            f"source frame count {frame_count}"
        )
    if presence_logits.shape != (frame_count, 2):
        raise ValueError(
            f"hand_presence_logits shape {presence_logits.shape} is invalid"
        )

    presence = 1.0 / (1.0 + np.exp(-np.clip(presence_logits, -80.0, 80.0)))
    return pose_enc, hand, presence


def decode_camera_frame_joints(hand: np.ndarray) -> dict[str, np.ndarray]:
    """Decode MINT camera-frame MANO parameters into 21 joints per hand."""
    mano.ensure_mano_weights()
    hands = prediction_to_hands(hand)

    joints_cam: dict[str, np.ndarray] = {}
    for side in HANDS:
        values = hands[side]
        decoded = mano.decode_hand_6d(
            values["transl_cam"],
            values["orient6d"],
            values["pose6d"],
            values["betas"],
            is_right=(side == "right"),
        )
        _vertices, joints = mano.run_mano(
            decoded["trans"],
            decoded["rot"],
            decoded["hand_pose"],
            decoded["betas"],
            is_right=(side == "right"),
        )
        joints_cam[side] = joints[:, :21].astype(np.float32)

    return joints_cam


def project_to_original(
    joints: np.ndarray,
    fov_h: float,
    fov_w: float,
    orig_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame joints directly into source-video pixels.

    With resize-only preprocessing, reconstructing intrinsics at an arbitrary
    intermediate resolution and then scaling to the source resolution is
    algebraically equivalent to this normalized-FOV projection. No MINT input
    resolution/config is required.
    """
    oh, ow = orig_hw

    if not (0.0 < fov_h < math.pi and 0.0 < fov_w < math.pi):
        raise ValueError(
            f"pose_enc contains invalid field-of-view values: {(fov_h, fov_w)}"
        )

    fx_norm = 1.0 / (2.0 * math.tan(fov_w / 2.0))
    fy_norm = 1.0 / (2.0 * math.tan(fov_h / 2.0))

    valid = np.isfinite(joints).all(axis=1) & (joints[:, 2] > MIN_VALID_DEPTH)
    uv = np.full((len(joints), 2), np.nan, dtype=np.float64)

    z = joints[valid, 2]
    uv[valid, 0] = (fx_norm * joints[valid, 0] / z + 0.5) * ow
    uv[valid, 1] = (fy_norm * joints[valid, 1] / z + 0.5) * oh

    return uv, valid


def square_bbox(
    points: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> list[float]:
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    size = max(max(x2 - x1, y2 - y1) * scale, 1.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = size / 2.0

    return [
        min(max(cx - half, 0.0), width - 1.0),
        min(max(cy - half, 0.0), height - 1.0),
        min(max(cx + half, 0.0), width - 1.0),
        min(max(cy + half, 0.0), height - 1.0),
    ]


def hand_record(
    index: int,
    slot: int,
    side: str,
    joints: np.ndarray,
    probability: float,
    fov_h: float,
    fov_w: float,
    orig_hw: tuple[int, int],
    bbox_scale: float,
) -> dict | None:
    height, width = orig_hw
    uv, valid = project_to_original(joints, fov_h, fov_w, orig_hw)

    inside = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    if not inside.any():
        return None

    keypoints = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.where(valid, probability, 0.0).astype(np.float32)

    camera_intrinsics = [
        [width / (2.0 * math.tan(fov_w / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * math.tan(fov_h / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        "observation_id": f"{index}:{side}",
        "handedness": side,
        "backend_handedness": side,
        "bbox_xyxy": square_bbox(uv[inside], bbox_scale, width, height),
        # Plain lists avoid numpy-pickle compatibility issues across environments.
        "keypoints_2d": np.column_stack((keypoints, confidence))
        .astype(np.float32)
        .tolist(),
        "physical_track_id": slot,
        "physical_track_fragment_id": 0,
        "confidence": probability,
        "source": "mint",
        "meta": {
            "source": "mint",
            "hand_presence": probability,
            "projection_valid": True,
            "confidence_type": "mint_presence_projected",
            "camera_frame": "opencv_x_right_y_down_z_forward",
            "joint_order": "openpose21",
            "joints_3d_camera": np.asarray(joints, dtype=np.float32).tolist(),
            "camera_intrinsics": camera_intrinsics,
            "input_transform": "resize_only",
        },
    }


def build_observation_sequence(
    pose_enc: np.ndarray,
    joints_cam: dict[str, np.ndarray],
    presence: np.ndarray,
    orig_hw: tuple[int, int],
    fps: float,
    source: Path,
    bbox_scale: float,
    presence_threshold: float,
) -> tuple[dict, dict[str, int]]:
    frame_count = len(presence)
    frames = []
    emitted = {"left": 0, "right": 0}

    for index in range(frame_count):
        fov_h = float(pose_enc[index, 7])
        fov_w = float(pose_enc[index, 8])
        hands = []

        for slot, side in enumerate(HANDS):
            probability = float(presence[index, slot])
            if probability < presence_threshold:
                continue

            record = hand_record(
                index=index,
                slot=slot,
                side=side,
                joints=joints_cam[side][index],
                probability=probability,
                fov_h=fov_h,
                fov_w=fov_w,
                orig_hw=orig_hw,
                bbox_scale=bbox_scale,
            )
            if record is not None:
                hands.append(record)
                emitted[side] += 1

        frames.append(
            {
                "frame_idx": index,
                "img_path": f"{source.resolve()}#frame={index}",
                "timestamp_ns": None,
                "hands": hands,
            }
        )

    height, width = orig_hw
    sequence = {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": {
            "sequence_name": source.stem,
            "frame_count": frame_count,
            "fps": float(fps),
            "image_width": width,
            "image_height": height,
        },
        "frontend": {"name": "mint", "version": "1.0"},
        "frames": frames,
    }

    validate_sequence(sequence)
    return sequence, emitted


def validate_sequence(sequence: dict) -> None:
    frames = sequence["frames"]

    if [frame["frame_idx"] for frame in frames] != list(range(len(frames))):
        raise ValueError("frame_idx must be consecutive zero-based indices")
    if sequence["sequence"]["frame_count"] != len(frames):
        raise ValueError("sequence frame_count does not match frames")

    for frame in frames:
        for hand in frame["hands"]:
            bbox = np.asarray(hand["bbox_xyxy"], dtype=float)
            if (
                bbox.shape != (4,)
                or not np.isfinite(bbox).all()
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError(f"Invalid bbox in frame {frame['frame_idx']}")

            points = np.asarray(hand["keypoints_2d"], dtype=float)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise ValueError(f"Invalid keypoints in frame {frame['frame_idx']}")

            if not np.isfinite(float(hand["confidence"])):
                raise ValueError(
                    f"Non-finite confidence in frame {frame['frame_idx']}"
                )


def export_cache(args: argparse.Namespace) -> None:
    if args.bbox_scale <= 0:
        raise ValueError("bbox_scale must be positive")
    if not 0.0 <= args.presence_threshold <= 1.0:
        raise ValueError("presence_threshold must lie in [0, 1]")

    for path, label in (
        (args.source, "source video"),
        (args.raw, "raw MINT prediction"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    frame_count, orig_w, orig_h, fps = read_video_metadata(args.source)
    pose_enc, hand, presence = load_raw_prediction(args.raw, frame_count)
    joints_cam = decode_camera_frame_joints(hand)

    sequence, emitted = build_observation_sequence(
        pose_enc=pose_enc,
        joints_cam=joints_cam,
        presence=presence,
        orig_hw=(orig_h, orig_w),
        fps=fps,
        source=args.source,
        bbox_scale=args.bbox_scale,
        presence_threshold=args.presence_threshold,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_bytes(
        pickle.dumps(sequence, protocol=pickle.HIGHEST_PROTOCOL)
    )
    temporary.replace(args.out)

    print(f"Wrote EgoHandKit canonical observations: {args.out}")
    print(
        f"frames={frame_count}, fps={fps:.3f}, size={orig_w}x{orig_h}, "
        f"left={emitted['left']}, right={emitted['right']}, "
        f"bbox_scale={args.bbox_scale}, "
        f"presence_threshold={args.presence_threshold}"
    )


if __name__ == "__main__":
    export_cache(parse_args())
