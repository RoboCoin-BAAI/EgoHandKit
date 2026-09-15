"""Small, boring writers for reproducible pipeline stage artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.stages = self.run_dir / "stages"
        self.final = self.run_dir / "final"
        self.stages.mkdir(parents=True, exist_ok=True)
        self.final.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, name: str) -> Path:
        path = self.stages / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write_json(path: str | Path, value: Any) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")

    @staticmethod
    def write_pickle(path: str | Path, value: Any) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def serialize_backend_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    """Convert runtime BackendOutputInstance objects to stable plain records."""
    records = []
    for output in outputs:
        records.append({"frame_idx": int(output.frame_idx), "img_path": str(output.img_path),
            "hand_side": output.hand_side, "mano_params": output.mano_params,
            "cam_trans": output.cam_trans, "pred_vertices": output.pred_vertices,
            "pred_keypoints_2d": output.pred_keypoints_2d, "raw_backend_meta": output.raw_backend_meta})
    return records

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.write_json(self.run_dir / "run_manifest.json", manifest)
