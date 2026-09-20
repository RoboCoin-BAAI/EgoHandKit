import numpy as np
import pyarrow.parquet as pq

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
    frames = [{
        "frame_idx": 0,
        "img_path": "frame.jpg",
        "timestamp_ns": 123,
        "hands": [_observation("left", left), _observation("right", right_mint, 0.7)],
    }]
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
