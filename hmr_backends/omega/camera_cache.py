from __future__ import annotations

from pathlib import Path

import numpy as np


def build_c2w(extrinsic_w2c: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic_w2c, dtype=np.float32)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"omega_extrinsic_w2c must have shape (T, 3, 4), got {extrinsic.shape}")
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], extrinsic.shape[0], axis=0)
    w2c[:, :3, :4] = extrinsic
    return np.linalg.inv(w2c).astype(np.float32)


def validate_camera_arrays(extrinsic_w2c: np.ndarray, intrinsic: np.ndarray, expected_frames: int) -> None:
    if extrinsic_w2c.shape != (expected_frames, 3, 4):
        raise ValueError(
            f"Omega extrinsic frame count or shape mismatch: expected {(expected_frames, 3, 4)}, "
            f"got {extrinsic_w2c.shape}"
        )
    if intrinsic.shape != (expected_frames, 3, 3):
        raise ValueError(
            f"Omega intrinsic frame count or shape mismatch: expected {(expected_frames, 3, 3)}, "
            f"got {intrinsic.shape}"
        )
    if not np.isfinite(extrinsic_w2c).all():
        raise ValueError("Omega extrinsic contains non-finite values")
    if not np.isfinite(intrinsic).all():
        raise ValueError("Omega intrinsic contains non-finite values")


def save_omega_camera_cache(
    cache_path: str | Path,
    image_paths: list[str],
    checkpoint_path: str,
    image_resolution: int,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    preprocess_meta: np.ndarray,
    chunk_meta: np.ndarray,
    alignment_meta: np.ndarray,
) -> None:
    validate_camera_arrays(extrinsic_w2c, intrinsic, expected_frames=len(image_paths))
    c2w = build_c2w(extrinsic_w2c)
    np.savez(
        cache_path,
        image_paths=np.array(image_paths, dtype=object),
        omega_checkpoint=str(checkpoint_path),
        omega_image_resolution=int(image_resolution),
        omega_extrinsic_w2c=extrinsic_w2c.astype(np.float32),
        omega_intrinsic=intrinsic.astype(np.float32),
        omega_c2w=c2w.astype(np.float32),
        preprocess_meta=preprocess_meta,
        chunk_meta=chunk_meta,
        alignment_meta=alignment_meta,
    )


def load_omega_camera_cache(cache_path: str | Path, expected_frames: int) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    result = {key: data[key] for key in data.files}
    validate_camera_arrays(
        result["omega_extrinsic_w2c"],
        result["omega_intrinsic"],
        expected_frames=expected_frames,
    )
    if result["omega_c2w"].shape != (expected_frames, 4, 4):
        raise ValueError(
            f"Omega c2w shape mismatch: expected {(expected_frames, 4, 4)}, "
            f"got {result['omega_c2w'].shape}"
        )
    return result
