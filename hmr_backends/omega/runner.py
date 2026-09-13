from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hmr_backends.omega.alignment import align_points_umeyama, c2w_to_w2c, transform_c2w_by_similarity
from hmr_backends.omega.camera_cache import (
    build_c2w,
    load_omega_camera_cache,
    save_omega_camera_cache,
)
from hmr_backends.omega.models import VGGTOmega
from hmr_backends.omega.preprocess import serialize_preprocess_meta
from hmr_backends.omega.utils.load_fn import load_and_preprocess_images_with_meta
from hmr_backends.omega.utils.pose_enc import encoding_to_camera


DEFAULT_OMEGA_CHECKPOINT = "_DATA/vggt_omega/vggt_omega_1b_512.pt"
DEFAULT_OMEGA_IMAGE_RESOLUTION = 512
DEFAULT_AUTO_CHUNK_SIZE = 500


def resolve_chunk_size(chunk_size: str | int, num_frames: int) -> int:
    if chunk_size == "auto":
        return min(num_frames, DEFAULT_AUTO_CHUNK_SIZE)
    try:
        value = int(chunk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--omega_chunk_size must be 'auto' or a positive integer, got {chunk_size!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"--omega_chunk_size must be positive, got {value}")
    return min(num_frames, value)


def build_chunks(num_frames: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if num_frames <= 0:
        raise ValueError("Omega requires at least one image")
    if overlap < 0:
        raise ValueError(f"--omega_overlap must be non-negative, got {overlap}")
    if chunk_size <= overlap and num_frames > chunk_size:
        raise ValueError(f"--omega_chunk_size ({chunk_size}) must be greater than --omega_overlap ({overlap})")
    chunks = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunks.append((start, end))
        if end == num_frames:
            break
        start = end - overlap
    return chunks


def load_omega_model(checkpoint_path: str | Path, device: torch.device) -> VGGTOmega:
    if device.type != "cuda":
        raise RuntimeError("Omega Pass 4 requires a CUDA device. Pass --no_omega_world to skip Pass 4.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Omega checkpoint not found: {checkpoint}. "
            "Pass --no_omega_world to skip Pass 4."
        )
    model = VGGTOmega().eval()
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device)


def infer_omega_chunk(
    model: VGGTOmega,
    image_paths: list[str],
    image_resolution: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, preprocess_meta = load_and_preprocess_images_with_meta(
        image_paths,
        image_resolution=image_resolution,
    )
    images = images.to(device)
    with torch.inference_mode():
        predictions = model(images)
    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    extrinsic_np = extrinsic.detach().float().cpu().numpy()
    intrinsic_np = intrinsic.detach().float().cpu().numpy()
    if extrinsic_np.shape[0] == 1:
        extrinsic_np = extrinsic_np[0]
    if intrinsic_np.shape[0] == 1:
        intrinsic_np = intrinsic_np[0]
    return extrinsic_np.astype(np.float32), intrinsic_np.astype(np.float32), serialize_preprocess_meta(preprocess_meta)


def run_omega_camera_recovery(
    image_paths: list[str],
    out_dir: str | Path,
    checkpoint_path: str | Path = DEFAULT_OMEGA_CHECKPOINT,
    image_resolution: int = DEFAULT_OMEGA_IMAGE_RESOLUTION,
    chunk_size: str | int = "auto",
    overlap: int = 8,
    force: bool = False,
    device: torch.device | None = None,
) -> dict:
    cache_path = Path(out_dir) / "omega_camera.npz"
    if cache_path.exists() and not force:
        return load_omega_camera_cache(cache_path, expected_frames=len(image_paths))
    if not image_paths:
        raise ValueError("Omega Pass 4 received no image paths")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved_chunk_size = resolve_chunk_size(chunk_size, len(image_paths))
    chunks = build_chunks(len(image_paths), resolved_chunk_size, overlap)
    model = load_omega_model(checkpoint_path, device)

    full_extrinsic = [None] * len(image_paths)
    full_intrinsic = [None] * len(image_paths)
    full_preprocess = [None] * len(image_paths)
    chunk_meta = []
    alignment_meta = []
    global_c2w = None

    for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Omega camera chunks")):
        chunk_paths = image_paths[start:end]
        extrinsic, intrinsic, preprocess_meta = infer_omega_chunk(
            model,
            chunk_paths,
            image_resolution=image_resolution,
            device=device,
        )
        c2w = build_c2w(extrinsic)
        if chunk_idx == 0:
            transformed_c2w = c2w
            alignment_meta.append({"chunk": chunk_idx, "start": start, "end": end, "residual": 0.0})
        else:
            overlap_indices = [idx for idx in range(start, end) if global_c2w is not None and full_extrinsic[idx] is not None]
            if len(overlap_indices) < 3:
                raise ValueError(
                    f"Chunk {chunk_idx} has only {len(overlap_indices)} overlap frames; "
                    "at least 3 are required for Sim(3) alignment"
                )
            local_centers = np.stack([c2w[idx - start, :3, 3] for idx in overlap_indices], axis=0)
            global_centers = np.stack([global_c2w[idx, :3, 3] for idx in overlap_indices], axis=0)
            sim = align_points_umeyama(local_centers, global_centers)
            transformed_c2w = transform_c2w_by_similarity(c2w, sim)
            alignment_meta.append(
                {
                    "chunk": chunk_idx,
                    "start": start,
                    "end": end,
                    "scale": sim.scale,
                    "rotation": sim.rotation,
                    "translation": sim.translation,
                    "residual": sim.residual,
                }
            )
        transformed_extrinsic = c2w_to_w2c(transformed_c2w)
        if global_c2w is None:
            global_c2w = np.empty((len(image_paths), 4, 4), dtype=np.float32)
            global_c2w[:] = np.nan
        write_start = start if chunk_idx == 0 else start + overlap
        for frame_idx in range(write_start, end):
            local_idx = frame_idx - start
            full_extrinsic[frame_idx] = transformed_extrinsic[local_idx]
            full_intrinsic[frame_idx] = intrinsic[local_idx]
            full_preprocess[frame_idx] = preprocess_meta[local_idx]
            global_c2w[frame_idx] = transformed_c2w[local_idx]
        chunk_meta.append({"chunk": chunk_idx, "start": start, "end": end, "write_start": write_start})

    if any(value is None for value in full_extrinsic):
        missing = [idx for idx, value in enumerate(full_extrinsic) if value is None]
        raise ValueError(f"Omega chunking left frames without cameras: {missing[:10]}")

    extrinsic_np = np.stack(full_extrinsic, axis=0).astype(np.float32)
    intrinsic_np = np.stack(full_intrinsic, axis=0).astype(np.float32)
    preprocess_np = np.array(full_preprocess, dtype=object)
    save_omega_camera_cache(
        cache_path=cache_path,
        image_paths=image_paths,
        checkpoint_path=str(checkpoint_path),
        image_resolution=image_resolution,
        extrinsic_w2c=extrinsic_np,
        intrinsic=intrinsic_np,
        preprocess_meta=preprocess_np,
        chunk_meta=np.array(chunk_meta, dtype=object),
        alignment_meta=np.array(alignment_meta, dtype=object),
    )
    return load_omega_camera_cache(cache_path, expected_frames=len(image_paths))
