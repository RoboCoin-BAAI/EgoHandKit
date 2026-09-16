from pathlib import Path

import cv2
import numpy as np

from observation_frontend.depth_gate import apply_depth_gate, resolve_depth_frames


def _write_depth_sequence(root: Path, values: list[int], shape=(40, 60)) -> Path:
    depth_dir = root / "fast_foundation" / "depth_uint16_png"
    depth_dir.mkdir(parents=True)
    for index, value in enumerate(values):
        image = np.full(shape, value, dtype=np.uint16)
        assert cv2.imwrite(str(depth_dir / f"frame_{index:06d}_depth_mm.png"), image)
    return root


def _images(root: Path, count: int, shape=(40, 60)) -> list[Path]:
    paths = []
    for index in range(count):
        path = root / f"image_{index:06d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((*shape, 3), dtype=np.uint8))
        paths.append(path)
    return paths


def _frames(count: int) -> list[dict]:
    return [
        {
            "frame_idx": index,
            "img_path": f"image_{index:06d}.jpg",
            "hands": [
                {
                    "observation_id": f"right-{index}",
                    "bbox_xyxy": [10, 10, 30, 30],
                    "keypoints_2d": np.zeros((21, 3), dtype=np.float32),
                    "handedness": "right",
                    "backend_handedness": "right",
                    "physical_track_id": 1,
                    "physical_track_fragment_id": 0,
                    "confidence": 1.0,
                    "source": "mint",
                    "meta": {},
                }
            ],
        }
        for index in range(count)
    ]


def test_depth_gate_rejects_outlier_and_starts_new_fragment(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500, 2000, 500])
    images = _images(tmp_path, 3)
    gated, report = apply_depth_gate(
        _frames(3), images, depth_root, max_depth_m=4.0, max_ratio=2.0
    )

    assert [len(frame["hands"]) for frame in gated] == [1, 0, 1]
    assert gated[0]["hands"][0]["physical_track_fragment_id"] == 0
    assert gated[2]["hands"][0]["physical_track_fragment_id"] == 1
    assert gated[2]["hands"][0]["meta"]["depth_m"] == 0.5
    assert report["rejected_observations"] == 1
    assert report["frames"][1]["hands"]["right"]["reasons"] == ["relative_depth_jump"]


def test_depth_gate_rejects_absolute_limit_and_missing_frame_is_explicit(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500, 5000])
    images = _images(tmp_path, 2)
    gated, report = apply_depth_gate(_frames(2), images, depth_root, max_depth_m=4.0)
    assert [len(frame["hands"]) for frame in gated] == [1, 0]
    assert report["frames"][1]["hands"]["right"]["reasons"] == ["absolute_depth_limit"]

    missing_root = tmp_path / "missing"
    (missing_root / "fast_foundation" / "depth_uint16_png").mkdir(parents=True)
    assert not (missing_root / "fast_foundation" / "depth_uint16_png" / "frame_000001_depth_mm.png").exists()
    try:
        resolve_depth_frames(missing_root, 2)
    except FileNotFoundError as exc:
        assert "missing frame" in str(exc)
    else:
        raise AssertionError("missing depth frame should fail explicitly")
