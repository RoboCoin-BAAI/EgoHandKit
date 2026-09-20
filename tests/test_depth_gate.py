import json
from pathlib import Path

import cv2
import numpy as np

from observation_frontend.depth_gate import (
    apply_depth_gate,
    resolve_depth_camera_intrinsics,
    resolve_depth_frames,
    scale_camera_intrinsics,
)


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


def test_wrist_only_depth_compares_mint_and_sensor_depth(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    frames[0]["hands"][0]["keypoints_2d"][0] = [20, 20, 1]
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.55
    frames[0]["hands"][0]["meta"].update({
        "joints_3d_camera": joints,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "joint_order": "openpose21",
    })

    gated, report = apply_depth_gate(
        frames, images, depth_root, wrist_only=True, wrist_threshold_m=0.08
    )

    assert len(gated[0]["hands"]) == 1
    assert gated[0]["hands"][0]["meta"]["depth_joint_used"] == "wrist"
    diagnostic = report["frames"][0]["hands"]["right"]
    assert diagnostic["depth_joint_used"] == "wrist"
    assert diagnostic["mint_wrist_depth_m"] == np.float32(0.55)


def test_wrist_only_depth_rejects_sensor_disagreement(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    frames[0]["hands"][0]["keypoints_2d"][0] = [20, 20, 1]
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.7
    frames[0]["hands"][0]["meta"].update({
        "joints_3d_camera": joints,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "joint_order": "openpose21",
    })

    gated, report = apply_depth_gate(
        frames, images, depth_root, wrist_only=True, wrist_threshold_m=0.08
    )

    assert gated[0]["hands"] == []
    assert report["frames"][0]["hands"]["right"]["reasons"] == ["wrist_depth_mismatch"]


def test_sensor_anchor_mode_keeps_mint_despite_original_depth_disagreement(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    frames[0]["hands"][0]["keypoints_2d"][0] = [20, 20, 1]
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.9
    frames[0]["hands"][0]["meta"].update({
        "joints_3d_camera": joints,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "joint_order": "openpose21",
    })

    gated, report = apply_depth_gate(
        frames, images, depth_root, wrist_only=True,
        wrist_threshold_m=0.08, sensor_anchor=True,
    )

    assert len(gated[0]["hands"]) == 1
    meta = gated[0]["hands"][0]["meta"]
    assert meta["sensor_wrist_depth_m"] == np.float32(0.5)
    assert meta["mint_wrist_depth_m"] == np.float32(0.9)
    assert np.isclose(meta["depth_difference_m"], 0.4)
    assert meta["depth_gate_mode"] == "sensor_wrist_anchor"
    assert np.isclose(
        report["frames"][0]["hands"]["right"]["depth_difference_m"], 0.4
    )
    assert report["rejected_observations"] == 0


def test_sensor_anchor_max_depth_and_missing_depth(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [999, 1000, 1001, 0])
    images = _images(tmp_path, 4)
    frames = _frames(4)
    for frame in frames:
        frame["hands"][0]["keypoints_2d"][0] = [20, 20, 1]
    gated, report = apply_depth_gate(
        frames, images, depth_root, wrist_only=True, sensor_anchor=True,
        max_depth_m=1.0,
    )
    assert [len(frame["hands"]) for frame in gated] == [1, 1, 0, 1]
    assert report["frames"][2]["hands"]["right"]["reasons"] == ["absolute_depth_limit"]
    meta = gated[3]["hands"][0]["meta"]
    assert meta["force_frontend_fallback"] is True
    assert meta["depth_gate_not_evaluated"] == "no_valid_depth"
    assert report["rejected_observations"] == 1


def test_sensor_anchor_mode_keeps_out_of_image_wrist_as_frontend_fallback(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    frames[0]["hands"][0]["keypoints_2d"][0] = [20, 45, 1]
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.9
    frames[0]["hands"][0]["meta"].update({
        "joints_3d_camera": joints,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "joint_order": "openpose21",
    })

    gated, report = apply_depth_gate(
        frames, images, depth_root, wrist_only=True, sensor_anchor=True,
    )

    assert len(gated[0]["hands"]) == 1
    meta = gated[0]["hands"][0]["meta"]
    assert meta["force_frontend_fallback"] is True
    assert meta["depth_gate_mode"] == "frontend_fallback"
    diagnostic = report["frames"][0]["hands"]["right"]
    assert diagnostic["valid"] is True
    assert diagnostic["not_evaluated_reason"] == "wrist_out_of_image"
    assert report["frontend_fallback_observations"] == 1


def test_wrist_only_depth_keeps_observation_without_mint_reference(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    frames[0]["hands"][0]["keypoints_2d"][0] = [20, 20, 1]

    gated, report = apply_depth_gate(frames, images, depth_root, wrist_only=True)

    assert len(gated[0]["hands"]) == 1
    diagnostic = report["frames"][0]["hands"]["right"]
    assert diagnostic["valid"] is True
    assert diagnostic["not_evaluated_reason"] == "missing_mint_wrist_depth"


def test_wrist_only_depth_rejects_invalid_wrist_projection(tmp_path):
    depth_root = _write_depth_sequence(tmp_path / "depth", [500])
    images = _images(tmp_path, 1)
    frames = _frames(1)
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.5
    frames[0]["hands"][0]["meta"].update({
        "joints_3d_camera": joints,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "joint_order": "openpose21",
    })
    frames[0]["hands"][0]["keypoints_2d"][0] = [0, 0, 0]

    gated, report = apply_depth_gate(frames, images, depth_root, wrist_only=True)

    assert gated[0]["hands"] == []
    assert report["frames"][0]["hands"]["right"]["reasons"] == ["no_valid_depth"]


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


def test_depth_camera_intrinsics_come_from_stereo_metadata(tmp_path):
    root = _write_depth_sequence(tmp_path / "depth", [500])
    metadata_path = root / "fast_foundation" / "fast_foundation_stereo_video_meta.json"
    metadata_path.write_text(json.dumps({
        "intrinsics": {"fx": 448.0, "fy": 449.0, "cx": 906.0, "cy": 547.0}
    }))

    intrinsics, source = resolve_depth_camera_intrinsics(root)

    assert np.allclose(intrinsics, [
        [448.0, 0.0, 906.0],
        [0.0, 449.0, 547.0],
        [0.0, 0.0, 1.0],
    ])
    assert source == str(metadata_path.resolve())


def test_depth_camera_reference_must_match_input_camera(tmp_path):
    root = _write_depth_sequence(tmp_path / "depth", [500])
    metadata_path = root / "fast_foundation" / "fast_foundation_stereo_video_meta.json"
    metadata_path.write_text(json.dumps({
        "reference_camera": "left",
        "intrinsics": {"fx": 448.0, "fy": 449.0, "cx": 906.0, "cy": 547.0},
    }))

    try:
        resolve_depth_camera_intrinsics(root, expected_reference_camera="right")
    except ValueError as exc:
        assert "registered to 'left'" in str(exc)
    else:
        raise AssertionError("right RGB must not use left-registered depth")


def test_camera_intrinsics_scale_with_resized_registered_depth():
    intrinsics = np.array([[400.0, 0.0, 200.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]])

    scaled = scale_camera_intrinsics(intrinsics, (200, 400), (100, 200))

    assert np.allclose(scaled, [[200.0, 0.0, 100.0], [0.0, 250.0, 50.0], [0.0, 0.0, 1.0]])
