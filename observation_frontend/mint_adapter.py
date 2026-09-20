"""MINT prediction-cache adapter for the pre-HaMeR observation contract.

The adapter deliberately has no import-time dependency on the MINT repository.
MINT exports are treated as a small, versioned boundary: camera parameters and
camera-frame MANO predictions are decoded here, projected to the source image,
and converted into the same ``selected_for_hamer`` records used by the existing
observation frontend.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def enlarge_bbox(bbox, scale=1.2, img_shape=None):
    """Expand a bbox to a square without importing the detector stack."""
    x1, y1, x2, y2 = map(float, bbox)
    size = max(x2 - x1, y2 - y1) * float(scale)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    result = np.array([cx - size / 2, cy - size / 2,
                       cx + size / 2, cy + size / 2], dtype=np.float32)
    if img_shape is not None:
        h, w = img_shape[:2]
        result[[0, 2]] = np.clip(result[[0, 2]], 0, w - 1)
        result[[1, 3]] = np.clip(result[[1, 3]], 0, h - 1)
    return result


MINT_CACHE_SCHEMA = "mint_prediction_cache.v1"
MINT_OBSERVATION_SCHEMA = "mint_observations.v2"
_PER_HAND = 109


def _as_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"MINT cache field {name!r} must have {ndim} dimensions, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"MINT cache field {name!r} must be numeric")
    return array


def _metadata_from_npz(data: Mapping[str, Any]) -> dict[str, Any]:
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if np.asarray(raw).ndim == 0:
        raw = np.asarray(raw).item()
    try:
        metadata = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("MINT cache metadata_json must contain a JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("MINT cache metadata_json must decode to an object")
    return metadata


def _field(data: Mapping[str, Any], metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
        if name in metadata:
            return metadata[name]
    return None


def _require(data: Mapping[str, Any], metadata: Mapping[str, Any], *names: str) -> Any:
    value = _field(data, metadata, *names)
    if value is None:
        joined = ", ".join(names)
        raise ValueError(f"MINT cache is missing required field (one of: {joined})")
    return value


def _normalise_hw(value: Any, frame_count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.shape == (2,):
        array = np.repeat(array[None], frame_count, axis=0)
    if array.shape != (frame_count, 2) or np.any(array <= 0):
        raise ValueError(f"MINT cache field {name!r} must be [2] or [{frame_count},2] with positive values")
    return array


def _normalise_presence(value: Any, frame_count: int, name: str) -> tuple[np.ndarray, bool]:
    values = np.asarray(value, dtype=np.float32)
    if values.shape == (frame_count,):
        values = values[:, None]
    if values.shape == (frame_count, 1):
        values = np.repeat(values, 2, axis=1)
    if values.shape != (frame_count, 2):
        raise ValueError(f"MINT presence field {name!r} must have shape [{frame_count},2]")
    if not np.isfinite(values).all():
        raise ValueError(f"MINT presence field {name!r} contains non-finite values")
    is_logits = "logit" in name.lower()
    if is_logits:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    elif np.any((values < 0) | (values > 1)):
        raise ValueError(f"MINT presence probabilities must lie in [0,1], got {name!r}")
    return values, is_logits


def load_mint_predictions(path: str | Path) -> dict[str, Any]:
    """Load and validate a MINT ``.npz`` prediction cache.

    Required arrays are ``frame_idx``, source ``orig_hw``, ``mint_input_hw``,
    camera intrinsics or ``pose_enc``, a combined ``hand [T,218]`` (or split
    ``left_hand/right_hand [T,109]``), and
    ``hand_presence`` (or ``hand_presence_logits``).  For environments without
    MANO assets, an exporter may additionally include ``*_joints_cam``; those
    decoded points are used directly and still retain the raw MANO arrays.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MINT prediction cache does not exist: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to read MINT prediction cache {path}: {exc}") from exc
    with loaded as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = _metadata_from_npz(data)
    schema = str(_field(data, metadata, "schema_version") or MINT_CACHE_SCHEMA)
    if schema != MINT_CACHE_SCHEMA:
        raise ValueError(f"Unsupported MINT cache schema {schema!r}; expected {MINT_CACHE_SCHEMA!r}")
    frame_idx = _as_array(_require(data, metadata, "frame_idx"), "frame_idx").reshape(-1)
    if frame_idx.size == 0 or not np.issubdtype(frame_idx.dtype, np.integer):
        raise ValueError("MINT cache frame_idx must be a non-empty integer vector")
    frame_idx = frame_idx.astype(np.int64)
    if not np.array_equal(frame_idx, np.arange(len(frame_idx))):
        raise ValueError("MINT cache frame_idx must be consecutive zero-based indices")
    frames = len(frame_idx)
    orig_value = _field(data, metadata, "orig_hw", "original_hw")
    if orig_value is None:
        ow = _require(data, metadata, "original_width", "orig_width")
        oh = _require(data, metadata, "original_height", "orig_height")
        orig_value = np.column_stack((np.broadcast_to(oh, frames), np.broadcast_to(ow, frames)))
    mint_value = _field(data, metadata, "mint_input_hw", "input_hw")
    if mint_value is None:
        mw = _require(data, metadata, "mint_input_width", "input_width")
        mh = _require(data, metadata, "mint_input_height", "input_height")
        mint_value = np.column_stack((np.broadcast_to(mh, frames), np.broadcast_to(mw, frames)))
    orig_hw = _normalise_hw(orig_value, frames, "orig_hw")
    mint_hw = _normalise_hw(mint_value, frames, "mint_input_hw")

    pose_enc = _field(data, metadata, "pose_enc")
    camera_intrinsics = _field(data, metadata, "camera_intrinsics", "intrinsics", "K")
    camera_extrinsics = _field(data, metadata, "camera_extrinsics", "extrinsics")
    if pose_enc is None and camera_intrinsics is None:
        raise ValueError("MINT cache needs pose_enc or camera_intrinsics")
    if pose_enc is not None:
        pose_enc = _as_array(pose_enc, "pose_enc", ndim=2).astype(np.float32)
        if pose_enc.shape != (frames, 9):
            raise ValueError(f"pose_enc must have shape [{frames},9], got {pose_enc.shape}")
    if camera_intrinsics is not None:
        camera_intrinsics = _as_array(camera_intrinsics, "camera_intrinsics", ndim=3).astype(np.float32)
        if camera_intrinsics.shape not in {(frames, 3, 3), (1, 3, 3)}:
            raise ValueError(f"camera_intrinsics must have shape [{frames},3,3] or [1,3,3]")
        if camera_intrinsics.shape[0] == 1:
            camera_intrinsics = np.repeat(camera_intrinsics, frames, axis=0)
        if not np.isfinite(camera_intrinsics).all():
            raise ValueError("camera_intrinsics contains non-finite values")
        if np.any(camera_intrinsics[:, 0, 0] <= 0) or np.any(camera_intrinsics[:, 1, 1] <= 0):
            raise ValueError("camera_intrinsics focal lengths must be positive")
    if camera_extrinsics is not None:
        camera_extrinsics = _as_array(camera_extrinsics, "camera_extrinsics", ndim=3).astype(np.float32)
        if camera_extrinsics.shape != (frames, 3, 4):
            raise ValueError(f"camera_extrinsics must have shape [{frames},3,4]")

    hands: dict[str, np.ndarray | None] = {}
    joints: dict[str, np.ndarray | None] = {}
    combined_hand = _field(data, metadata, "hand", "mano_hand")
    if combined_hand is not None:
        combined_hand = _as_array(combined_hand, "hand", ndim=2).astype(np.float32)
        if combined_hand.shape != (frames, 218):
            raise ValueError(f"hand must have shape [{frames},218], got {combined_hand.shape}")
    for side in ("left", "right"):
        hand = _field(data, metadata, f"{side}_hand", f"hand_{side}", f"{side}_mano", f"{side}_mano_params")
        if hand is None and combined_hand is not None:
            hand = combined_hand[:, :109] if side == "left" else combined_hand[:, 109:]
        if hand is not None:
            hand = _as_array(hand, f"{side}_hand", ndim=2).astype(np.float32)
            if hand.shape != (frames, 109):
                raise ValueError(f"{side}_hand must have shape [{frames},109], got {hand.shape}")
        hands[side] = hand
        points = _field(data, metadata, f"{side}_joints_cam", f"{side}_joints_3d_cam")
        if points is not None:
            points = _as_array(points, f"{side}_joints_cam", ndim=3).astype(np.float32)
            if points.shape[:2] != (frames, 21) or points.shape[2] != 3:
                raise ValueError(f"{side}_joints_cam must have shape [{frames},21,3]")
        joints[side] = points
        if hand is None and points is None:
            raise ValueError(f"MINT cache needs {side}_hand or {side}_joints_cam")
    presence_field = next((name for name in ("hand_presence_logits", "hand_presence", "hand_confidence", "presence") if _field(data, metadata, name) is not None), None)
    if presence_field is None:
        left_presence = _field(data, metadata, "left_presence", "left_hand_presence")
        right_presence = _field(data, metadata, "right_presence", "right_hand_presence")
        if left_presence is None or right_presence is None:
            raise ValueError("MINT cache needs hand_presence/hand_presence_logits or left_presence and right_presence")
        left, left_logits = _normalise_presence(left_presence, frames, "left_presence_logits" if "logit" in str(_field(data, metadata, "left_presence_kind") or "") else "left_presence")
        right, right_logits = _normalise_presence(right_presence, frames, "right_presence_logits" if "logit" in str(_field(data, metadata, "right_presence_kind") or "") else "right_presence")
        presence = np.column_stack((left[:, 0], right[:, 0]))
        presence_was_logits = left_logits or right_logits
        presence_field = "left_presence/right_presence"
    else:
        presence, presence_was_logits = _normalise_presence(_field(data, metadata, presence_field), frames, presence_field)

    affine = _field(data, metadata, "orig_to_mint", "original_to_mint")
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape == (2, 3):
            affine = np.repeat(affine[None], frames, axis=0)
        if affine.shape != (frames, 2, 3) or not np.isfinite(affine).all():
            raise ValueError(f"orig_to_mint must have shape [2,3] or [{frames},2,3]")
    timestamps = _field(data, metadata, "timestamp_ns", "timestamps_ns", "timestamp")
    if timestamps is not None:
        timestamps = np.asarray(timestamps).reshape(-1)
        if len(timestamps) != frames:
            raise ValueError("MINT timestamps must have one value per frame")
    return {
        "path": str(path.resolve()), "schema_version": schema, "metadata": metadata,
        "frame_idx": frame_idx, "orig_hw": orig_hw, "mint_input_hw": mint_hw,
        "pose_enc": pose_enc, "camera_intrinsics": camera_intrinsics,
        "camera_extrinsics": camera_extrinsics, "hands": hands, "joints_cam": joints,
        "presence": presence, "presence_field": presence_field,
        "presence_was_logits": presence_was_logits, "orig_to_mint": affine,
        "timestamps": timestamps,
    }


def _pose_intrinsics(pose: np.ndarray, height: int, width: int) -> np.ndarray:
    if pose.shape != (9,):
        raise ValueError(f"pose_enc row must have shape [9], got {pose.shape}")
    fov_h, fov_w = float(pose[7]), float(pose[8])
    if not np.isfinite((fov_h, fov_w)).all() or not (0 < fov_h < math.pi and 0 < fov_w < math.pi):
        raise ValueError("pose_enc contains invalid field-of-view values")
    return np.array([[width / (2 * math.tan(fov_w / 2)), 0, width / 2],
                     [0, height / (2 * math.tan(fov_h / 2)), height / 2],
                     [0, 0, 1]], dtype=np.float64)


def _rotation6d_to_matrix(values: np.ndarray) -> np.ndarray:
    a0, a1 = values[..., :3], values[..., 3:]
    b0 = a0 / np.maximum(np.linalg.norm(a0, axis=-1, keepdims=True), 1e-8)
    a1 = a1 - np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack((b0, b1, b2), axis=-2)


def _decode_mano_joints(hand: np.ndarray, is_right: bool, model_dir: str | Path | None) -> np.ndarray:
    if model_dir is None:
        raise RuntimeError("MINT cache contains raw MANO parameters but no decoded joints; pass --mint_mano_model_dir or export *_joints_cam")
    try:
        from scipy.spatial.transform import Rotation
        import torch
        from hmr_backends.models.mano_wrapper import MANO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Decoding raw MINT MANO requires scipy, torch, smplx, and EgoHandKit MANO dependencies") from exc
    values = np.asarray(hand, dtype=np.float32)
    orient = _rotation6d_to_matrix(values[:, 3:9])
    pose = _rotation6d_to_matrix(values[:, 9:99].reshape(-1, 15, 6))
    orient_aa = Rotation.from_matrix(orient).as_rotvec().astype(np.float32)
    pose_aa = Rotation.from_matrix(pose.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45).astype(np.float32)
    if not is_right:
        pose_aa[:, 1::3] *= -1
        pose_aa[:, 2::3] *= -1
    model = MANO(model_path=str(model_dir), is_rhand=is_right, gender="neutral",
                 num_hand_joints=15, create_body_pose=False).eval()
    betas = torch.from_numpy(values[:, 99:109])
    with torch.no_grad():
        # MINT subtracts the canonical identity-pose MANO root before applying
        # its camera-frame translation, rather than subtracting a posed root.
        base = model(global_orient=torch.zeros((len(values), 1, 3)),
                     hand_pose=torch.zeros((len(values), 45)), betas=betas,
                     transl=torch.zeros((len(values), 3)), pose2rot=True)
        root = base.joints[:, 0].detach().cpu().numpy()
    trans = values[:, :3] - root
    with torch.no_grad():
        output = model(global_orient=torch.from_numpy(orient_aa)[:, None],
                       hand_pose=torch.from_numpy(pose_aa), betas=betas,
                       transl=torch.from_numpy(trans), pose2rot=True)
    return output.joints[:, :21].detach().cpu().numpy().astype(np.float32)


def _project_points(points: np.ndarray, K_input: np.ndarray, orig_hw: np.ndarray,
                    mint_hw: np.ndarray, affine: np.ndarray | None) -> tuple[np.ndarray, bool]:
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    if not finite.any():
        return projected.astype(np.float32), False
    xyz = points[finite]
    uv = np.column_stack((K_input[0, 0] * xyz[:, 0] / xyz[:, 2] + K_input[0, 2],
                          K_input[1, 1] * xyz[:, 1] / xyz[:, 2] + K_input[1, 2]))
    if affine is None:
        # Explicitly documented resize-only mapping; crops must provide affine.
        oh, ow = orig_hw
        mh, mw = mint_hw
        uv[:, 0] *= float(ow) / float(mw)
        uv[:, 1] *= float(oh) / float(mh)
    else:
        A = np.vstack((affine, [0, 0, 1]))
        try:
            uv_h = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(A).T
        except np.linalg.LinAlgError:
            return np.empty((0, 2), dtype=np.float32), False
        uv = uv_h[:, :2] / uv_h[:, 2:3]
    good = np.isfinite(uv).all(axis=1)
    projected[np.flatnonzero(finite)[good]] = uv[good]
    if not good.any():
        return projected.astype(np.float32), False
    h, w = orig_hw
    inside = np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0) & (projected[:, 0] <= w - 1) & (projected[:, 1] >= 0) & (projected[:, 1] <= h - 1)
    return projected.astype(np.float32), bool(inside.any())


def build_mint_observations(predictions: Mapping[str, Any], image_paths: list[str | Path], *,
                            bbox_scale: float = 1.2, presence_threshold: float = 0.5,
                            mano_model_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Project MINT predictions and return standard EgoHandKit frame records."""
    if bbox_scale <= 0 or not 0 <= presence_threshold <= 1:
        raise ValueError("bbox_scale must be positive and presence_threshold must lie in [0,1]")
    paths = [str(Path(path)) for path in image_paths]
    n = len(predictions["frame_idx"])
    if len(paths) != n:
        raise ValueError(f"MINT cache has {n} frames but input sequence has {len(paths)} images")
    frames = []
    for i, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read input image for MINT frame {i}: {path}")
        h, w = image.shape[:2]
        expected_h, expected_w = predictions["orig_hw"][i]
        if (h, w) != (int(expected_h), int(expected_w)):
            raise ValueError(f"MINT cache original size {(expected_h, expected_w)} does not match {path} size {(h, w)}")
        mint_h, mint_w = predictions["mint_input_hw"][i]
        if predictions["camera_intrinsics"] is not None:
            K = predictions["camera_intrinsics"][i].astype(np.float64)
            intrinsics_space = predictions["metadata"].get("intrinsics_space", "mint_input")
            if intrinsics_space == "original":
                # Convert original-space K to the input space only for the affine path.
                K_input = K
                direct_original_K = True
            elif intrinsics_space == "mint_input":
                K_input, direct_original_K = K, False
            else:
                raise ValueError("metadata intrinsics_space must be 'mint_input' or 'original'")
        else:
            K_input, direct_original_K = _pose_intrinsics(predictions["pose_enc"][i], int(mint_h), int(mint_w)), False
        frame = {"frame_idx": int(predictions["frame_idx"][i]), "img_path": path,
                 "selected_for_hamer": [], "hands": []}
        if predictions["timestamps"] is not None:
            frame["timestamp_ns"] = int(predictions["timestamps"][i])
        affine = predictions["orig_to_mint"][i] if predictions["orig_to_mint"] is not None else None
        if direct_original_K:
            original_intrinsics = K_input
        elif affine is not None:
            original_intrinsics = np.linalg.inv(
                np.vstack((affine, [0, 0, 1]))
            ) @ K_input
        else:
            original_intrinsics = np.array(
                [[w / float(mint_w), 0, 0], [0, h / float(mint_h), 0], [0, 0, 1]],
                dtype=np.float64,
            ) @ K_input
        for side_index, side in enumerate(("left", "right")):
            if float(predictions["presence"][i, side_index]) < presence_threshold:
                continue
            points = predictions["joints_cam"][side][i] if predictions["joints_cam"][side] is not None else _decode_mano_joints(predictions["hands"][side][i], side == "right", mano_model_dir)[0]
            if direct_original_K:
                # Reuse the projection routine by treating original pixels as input.
                uv, valid = _project_points(points, K_input, np.array([h, w]), np.array([h, w]), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))
            else:
                uv, valid = _project_points(points, K_input, np.array([h, w]), np.array([mint_h, mint_w]), affine)
            if not valid:
                continue
            inside = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)]
            if len(inside) == 0:
                continue
            raw_bbox = np.array([inside[:, 0].min(), inside[:, 1].min(), inside[:, 0].max(), inside[:, 1].max()], dtype=np.float32)
            bbox = enlarge_bbox(raw_bbox, scale=bbox_scale, img_shape=image.shape).astype(np.float32)
            valid_keypoint = np.isfinite(uv).all(axis=1)
            keypoint_xy = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
            keypoint_confidence = np.where(valid_keypoint, float(predictions["presence"][i, side_index]), 0.0).astype(np.float32)
            keypoints = np.column_stack((keypoint_xy, keypoint_confidence))
            observation_meta = {"source": "mint", "hand_presence": float(predictions["presence"][i, side_index]),
                "projection_valid": True, "confidence_type": "mint_presence_projected",
                "mint_presence_field": predictions["presence_field"],
                "camera_frame": "opencv_x_right_y_down_z_forward",
                "joint_order": "openpose21",
                "joints_3d_camera": np.asarray(points, dtype=np.float32).tolist(),
                "camera_intrinsics": original_intrinsics.tolist()}
            frame["selected_for_hamer"].append({
                "bbox_xyxy": bbox.tolist(), "vitpose_keypoints_2d": keypoints.tolist(),
                "handedness": side, "backend_handedness": side,
                "physical_track_id": side_index, "physical_track_fragment_id": 0,
                "observation_meta": observation_meta,
            })
        frame["hands"] = frame["selected_for_hamer"]
        frames.append(frame)
    return frames


def image_sequence_signature(image_paths: list[str | Path]) -> str:
    payload = []
    for value in image_paths:
        path = Path(value).resolve(); stat = path.stat()
        payload.append([str(path), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_mint_observations(frames: list[Mapping[str, Any]], *, image_paths: list[str | Path] | None = None) -> None:
    """Validate the exact pre-HaMeR shape emitted by this adapter."""
    if image_paths is not None and len(frames) != len(image_paths):
        raise ValueError("MINT observations frame count does not match image sequence")
    for expected, frame in enumerate(frames):
        if int(frame.get("frame_idx", -1)) != expected:
            raise ValueError("MINT observation frame_idx must be consecutive zero-based indices")
        if "img_path" not in frame or "selected_for_hamer" not in frame:
            raise ValueError(f"MINT observation frame {expected} is missing img_path or selected_for_hamer")
        for observation in frame["selected_for_hamer"]:
            required = {"bbox_xyxy", "vitpose_keypoints_2d", "handedness", "backend_handedness",
                        "physical_track_id", "physical_track_fragment_id", "observation_meta"}
            missing = required.difference(observation)
            if missing:
                raise ValueError(f"MINT observation frame {expected} missing fields: {sorted(missing)}")
            if observation["handedness"] not in {"left", "right"}:
                raise ValueError("MINT observation handedness must be left or right")
            bbox = np.asarray(observation["bbox_xyxy"], dtype=np.float32)
            keypoints = np.asarray(observation["vitpose_keypoints_2d"], dtype=np.float32)
            if bbox.shape != (4,) or keypoints.shape != (21, 3) or not np.isfinite(bbox).all() or not np.isfinite(keypoints).all():
                raise ValueError(f"Invalid MINT observation geometry in frame {expected}")
            meta = observation["observation_meta"]
            if meta.get("source") != "mint" or meta.get("confidence_type") != "mint_presence_projected":
                raise ValueError("MINT observation metadata must identify source and confidence type")


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def compare_mint_yolo_observations(
    mint_frames: list[Mapping[str, Any]],
    yolo_frames: list[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare MINT boxes to YOLO boxes without changing MINT observations.

    This is deliberately a diagnostic check, not a YOLO fallback or a gate on
    HaMeR input. YOLO records are the legacy ``left_bbox``/``right_bbox`` rows.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in [0,1]")
    if len(mint_frames) != len(yolo_frames):
        raise ValueError("MINT and YOLO validation frame counts differ")
    rows = []
    counts = {"match": 0, "low_iou": 0, "mint_missing": 0, "yolo_missing": 0, "both_missing": 0}
    for mint_frame, yolo_frame in zip(mint_frames, yolo_frames):
        by_side = {
            item["handedness"]: item
            for item in mint_frame.get("hands", mint_frame.get("selected_for_hamer", []))
        }
        frame_rows = {"frame_idx": int(mint_frame["frame_idx"]), "sides": {}}
        for side in ("left", "right"):
            mint_item = by_side.get(side)
            yolo_bbox = yolo_frame.get(f"{side}_bbox")
            mint_bbox = mint_item.get("bbox_xyxy") if mint_item is not None else None
            if mint_bbox is None and yolo_bbox is None:
                status = "both_missing"
                counts[status] += 1
                iou = None
            elif mint_bbox is None:
                status = "mint_missing"
                counts[status] += 1
                iou = 0.0
            elif yolo_bbox is None:
                status = "yolo_missing"
                counts[status] += 1
                iou = 0.0
            else:
                iou = _bbox_iou(np.asarray(mint_bbox), np.asarray(yolo_bbox))
                status = "match" if iou >= iou_threshold else "low_iou"
                counts[status] += 1
            frame_rows["sides"][side] = {
                "status": status,
                "iou": iou,
                "mint_bbox_xyxy": None if mint_bbox is None else [float(value) for value in mint_bbox],
                "yolo_bbox_xyxy": None if yolo_bbox is None else [float(value) for value in yolo_bbox],
                "yolo_confidence": None if yolo_frame.get(f"{side}_conf") is None else float(yolo_frame[f"{side}_conf"]),
            }
        rows.append(frame_rows)
    total = sum(counts.values())
    return {
        "schema_version": "mint_yolo_check.v1",
        "source": "mint_yolo_diagnostic",
        "iou_threshold": float(iou_threshold),
        "frame_count": len(mint_frames),
        "side_count": total,
        "counts": counts,
        "frames": rows,
    }


def save_mint_observation_cache(path: str | Path, frames: list[dict[str, Any]], *, image_paths: list[str | Path], mint_path: str | Path, config: Mapping[str, Any]) -> None:
    validate_mint_observations(frames, image_paths=image_paths)
    payload = {"schema_version": MINT_OBSERVATION_SCHEMA, "source": "mint", "source_version": 2,
               "image_signature": image_sequence_signature(image_paths), "mint_cache": str(Path(mint_path).resolve()),
               "mint_cache_signature": hashlib.sha256(Path(mint_path).read_bytes()).hexdigest(),
               "projection_config": dict(config), "frame_count": len(frames), "frames": frames}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(payload))
    temporary.replace(path)


def load_mint_observation_cache(path: str | Path, *, image_paths: list[str | Path], mint_path: str | Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = pickle.loads(path.read_bytes())
    if payload.get("schema_version") != MINT_OBSERVATION_SCHEMA:
        raise ValueError(f"Unsupported MINT observation cache schema: {payload.get('schema_version')!r}")
    expected = hashlib.sha256(Path(mint_path).read_bytes()).hexdigest()
    if payload.get("image_signature") != image_sequence_signature(image_paths) or payload.get("mint_cache_signature") != expected or payload.get("projection_config") != dict(config):
        raise ValueError("MINT observation cache does not match the input images, prediction cache, or projection configuration; use --force_detect to rebuild")
    if payload.get("frame_count") != len(image_paths):
        raise ValueError("MINT observation cache frame count does not match input sequence")
    validate_mint_observations(payload["frames"], image_paths=image_paths)
    return payload["frames"]


def write_synthetic_mint_fixture(path: str | Path, *, frame_count: int = 1, original_hw=(100, 160), mint_input_hw=(50, 80)) -> Path:
    """Write a tiny, dependency-free cache useful for adapter smoke tests."""
    h, w = original_hw; mh, mw = mint_input_hw
    joints = np.zeros((frame_count, 21, 3), np.float32); joints[..., 2] = 2.0
    joints[..., 0] = np.linspace(-0.3, 0.3, 21); joints[..., 1] = np.linspace(-0.2, 0.2, 21)
    hands = np.zeros((frame_count, 109), np.float32)
    presence = np.tile(np.array([[1.0, 0.0]], np.float32), (frame_count, 1))
    K = np.tile(np.array([[[50.0, 0.0, mw / 2], [0.0, 50.0, mh / 2], [0.0, 0.0, 1.0]]], np.float32), (frame_count, 1, 1))
    path = Path(path)
    np.savez(path, schema_version=np.array(MINT_CACHE_SCHEMA), frame_idx=np.arange(frame_count),
             orig_hw=np.tile([h, w], (frame_count, 1)), mint_input_hw=np.tile([mh, mw], (frame_count, 1)),
             camera_intrinsics=K,
             left_hand=hands, right_hand=hands,
             left_joints_cam=joints, right_joints_cam=joints, hand_presence=presence,
             metadata_json=np.array(json.dumps({"intrinsics_space": "mint_input"})))
    return path
