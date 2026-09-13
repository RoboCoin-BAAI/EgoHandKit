from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class HandInstance:
    frame_idx: int
    img_path: str
    hand_side: str
    bbox: np.ndarray
    bbox_square: np.ndarray
    keypoints: Any = None
    observation_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pass3Inputs:
    img_paths: list[Path]
    image_size: tuple[int, int]
    instances: list[HandInstance]
    instances_by_key: dict[tuple[int, str], HandInstance]
    segments_by_hand: dict[str, list[list[int]]]
    img_focal: float | None = None
    temporal_segments: list[list[HandInstance]] | None = None


@dataclass
class BackendOutputInstance:
    frame_idx: int
    img_path: str
    hand_side: str
    mano_params: dict[str, Any]
    cam_trans: np.ndarray
    pred_vertices: np.ndarray
    pred_keypoints_2d: np.ndarray
    raw_backend_meta: dict[str, Any] = field(default_factory=dict)
