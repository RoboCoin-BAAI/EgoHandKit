from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OmegaPreprocessMeta:
    original_hw: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    resized_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    pad_ltrb: tuple[int, int, int, int]
    scale_xy: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "original_hw": self.original_hw,
            "crop_xyxy": self.crop_xyxy,
            "resized_hw": self.resized_hw,
            "padded_hw": self.padded_hw,
            "pad_ltrb": self.pad_ltrb,
            "scale_xy": self.scale_xy,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "OmegaPreprocessMeta":
        return cls(
            original_hw=tuple(int(x) for x in value["original_hw"]),
            crop_xyxy=tuple(int(x) for x in value["crop_xyxy"]),
            resized_hw=tuple(int(x) for x in value["resized_hw"]),
            padded_hw=tuple(int(x) for x in value["padded_hw"]),
            pad_ltrb=tuple(int(x) for x in value["pad_ltrb"]),
            scale_xy=tuple(float(x) for x in value["scale_xy"]),
        )


def crop_to_supported_aspect_ratio_box(
    width: int,
    height: int,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[int, int, int, int]:
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def compute_preprocess_meta(
    original_hw: tuple[int, int],
    image_resolution: int,
    patch_size: int = 16,
    mode: str = "balanced",
    padded_hw: tuple[int, int] | None = None,
) -> OmegaPreprocessMeta:
    if mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported Omega preprocess mode: {mode}")
    height, width = original_hw
    x1, y1, x2, y2 = crop_to_supported_aspect_ratio_box(width, height)
    crop_w = x2 - x1
    crop_h = y2 - y1
    aspect_ratio = crop_h / max(crop_w, 1)
    if mode == "balanced":
        resized_hw = balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        resized_hw = max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    padded_hw = padded_hw or resized_hw
    pad_h = padded_hw[0] - resized_hw[0]
    pad_w = padded_hw[1] - resized_hw[1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"padded_hw {padded_hw} is smaller than resized_hw {resized_hw}")
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    scale_x = resized_hw[1] / max(crop_w, 1)
    scale_y = resized_hw[0] / max(crop_h, 1)
    return OmegaPreprocessMeta(
        original_hw=(height, width),
        crop_xyxy=(x1, y1, x2, y2),
        resized_hw=resized_hw,
        padded_hw=padded_hw,
        pad_ltrb=(pad_left, pad_top, pad_right, pad_bottom),
        scale_xy=(float(scale_x), float(scale_y)),
    )


def map_original_points_to_omega(points_xy: np.ndarray, meta: OmegaPreprocessMeta) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    x1, y1, _, _ = meta.crop_xyxy
    pad_left, pad_top, _, _ = meta.pad_ltrb
    scale_x, scale_y = meta.scale_xy
    out = points.copy()
    out[:, 0] = (out[:, 0] - x1) * scale_x + pad_left
    out[:, 1] = (out[:, 1] - y1) * scale_y + pad_top
    return out


def serialize_preprocess_meta(meta_list: Iterable[OmegaPreprocessMeta]) -> np.ndarray:
    return np.array([meta.to_dict() for meta in meta_list], dtype=object)
