import numpy as np
import pyarrow.parquet as pq
import cv2

from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet


def _observation(side: str, joints: np.ndarray, confidence: float = 0.8) -> dict:
    return {
        "observation_id": f"0:{side}",
        "handedness": side,
        "backend_handedness": side,
        "bbox_xyxy": [1.0, 1.0, 10.0, 10.0],
        "keypoints_2d": np.zeros((21, 3), dtype=np.float32),
        "physical_track_id": 0 if side == "left" else 1,
        "physical_track_fragment_id": 0,
        "confidence": confidence,
        "source": "mint",
        "meta": {"joints_3d_camera": joints.tolist(),
                 "camera_frame": "opencv_x_right_y_down_z_forward",
                 "joint_order": "openpose21"},
    }


def test_parquet_round_trip_and_missing_hmr_keeps_mint(tmp_path):
    left = np.arange(63, dtype=np.float32).reshape(21, 3) / 100.0
    right_mint = left + 1.0
    right_hmr = left + 2.0
    frames = [
        {
            "frame_idx": 0,
            "img_path": "frame.jpg",
            "timestamp_ns": 123,
            "hands": [_observation("left", left), _observation("right", right_mint, 0.7)],
        },
        {
            "frame_idx": 1,
            "img_path": "missing.jpg",
            "timestamp_ns": None,
            "hands": [],
        },
    ]
    results = {"frame.jpg": {
        "joints_3d": [right_hmr],
        "backend_meta": [{"handedness": "right", "confidence": 0.9}],
    }}

    path = tmp_path / "hand_tracking.parquet"
    export_hand_tracking_parquet(frames, results, path)
    table = pq.read_table(path)
    row = table.to_pylist()[0]

    assert row["frame_idx"] == 0
    assert row["timestamp_ns"] == 123
    assert row["left_present"] is True
    assert row["right_present"] is True
    assert np.allclose(row["left_joints_3d"], left)
    assert np.allclose(row["right_joints_3d"], right_hmr)
    assert row["left_confidence"] == np.float32(0.8)
    assert row["right_confidence"] == np.float32(0.9)
    assert row["source"] == "mixed"
    assert table.schema.field("left_joints_3d").type.list_size == 21

    missing = table.to_pylist()[1]
    assert missing["left_present"] is False
    assert missing["right_present"] is False
    assert np.isnan(missing["left_joints_3d"]).all()
    assert np.isnan(missing["right_joints_3d"]).all()
    assert missing["left_confidence"] is None
    assert missing["right_confidence"] is None
    assert missing["source"] == "none"


def test_parquet_mint_fallback_is_anchored_to_sensor_wrist_depth(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    depth_dir = tmp_path / "depth/fast_foundation/depth_uint16_png"
    depth_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(depth_dir / "frame_000000_depth_mm.png"),
        np.full((20, 20), 500, dtype=np.uint16),
    )
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 1.0
    observation = _observation("left", joints)
    observation["keypoints_2d"][:, :] = [10.0, 10.0, 1.0]
    observation["meta"]["camera_intrinsics"] = [
        [100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]
    ]
    frames = [{"frame_idx": 0, "img_path": str(image_path),
               "timestamp_ns": None, "hands": [observation]}]

    path = tmp_path / "anchored.parquet"
    export_hand_tracking_parquet(frames, {}, path, depth_dir=tmp_path / "depth")
    row = pq.read_table(path).to_pylist()[0]

    assert row["left_present"] is True
    assert np.isclose(row["left_joints_3d"][0][2], 0.5)


def test_parquet_does_not_mix_unanchored_mint_when_sensor_depth_is_missing(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    depth_dir = tmp_path / "depth/fast_foundation/depth_uint16_png"
    depth_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(depth_dir / "frame_000000_depth_mm.png"),
        np.zeros((20, 20), dtype=np.uint16),
    )
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.8
    observation = _observation("left", joints)
    observation["keypoints_2d"][:, :] = [10.0, 10.0, 1.0]
    frames = [{"frame_idx": 0, "img_path": str(image_path),
               "timestamp_ns": None, "hands": [observation]}]

    path = tmp_path / "missing_depth.parquet"
    export_hand_tracking_parquet(frames, {}, path, depth_dir=tmp_path / "depth")
    row = pq.read_table(path).to_pylist()[0]

    assert row["left_present"] is False
    assert np.isnan(row["left_joints_3d"]).all()
    assert row["source"] == "none"


def test_parquet_uses_frontend_joints_for_explicit_out_of_image_fallback(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    depth_dir = tmp_path / "depth/fast_foundation/depth_uint16_png"
    depth_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(depth_dir / "frame_000000_depth_mm.png"),
        np.full((20, 20), 500, dtype=np.uint16),
    )
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.8
    observation = _observation("left", joints)
    observation["keypoints_2d"][:, :] = [10.0, 25.0, 1.0]
    observation["meta"]["force_frontend_fallback"] = True
    frames = [{"frame_idx": 0, "img_path": str(image_path),
               "timestamp_ns": None, "hands": [observation]}]

    path = tmp_path / "frontend_fallback.parquet"
    export_hand_tracking_parquet(frames, {}, path, depth_dir=tmp_path / "depth")
    row = pq.read_table(path).to_pylist()[0]

    assert row["left_present"] is True
    assert np.allclose(row["left_joints_3d"], joints)
    assert row["source"] == "mint_frontend_fallback"
