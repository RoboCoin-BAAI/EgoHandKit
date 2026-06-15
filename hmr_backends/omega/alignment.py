from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual: float


def align_points_umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> SimilarityTransform:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"source and target must both have shape (N, 3), got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 overlap camera centers are required for Sim(3) alignment")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    rank = np.linalg.matrix_rank(src_centered)
    if rank < 2:
        raise ValueError("Overlap camera centers are degenerate for Sim(3) alignment")
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vh
    if with_scale:
        variance = np.sum(src_centered ** 2) / src.shape[0]
        if variance <= 1e-12:
            raise ValueError("Source camera centers have near-zero variance")
        scale = float(np.sum(singular_values * sign) / variance)
    else:
        scale = 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    mapped = scale * (src @ rotation.T) + translation
    residual = float(np.sqrt(np.mean(np.sum((mapped - dst) ** 2, axis=1))))
    return SimilarityTransform(
        scale=scale,
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
        residual=residual,
    )


def transform_c2w_by_similarity(c2w: np.ndarray, sim: SimilarityTransform) -> np.ndarray:
    poses = np.asarray(c2w, dtype=np.float32).copy()
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"c2w must have shape (T, 4, 4), got {poses.shape}")
    out = poses.copy()
    out[:, :3, :3] = sim.rotation @ poses[:, :3, :3]
    centers = poses[:, :3, 3]
    out[:, :3, 3] = sim.scale * (centers @ sim.rotation.T) + sim.translation
    return out.astype(np.float32)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    return inv[:, :3, :4].astype(np.float32)
