from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "hmr_backends" / "utils" / "final_orientation_gate.py"
_SPEC = importlib.util.spec_from_file_location("final_orientation_gate", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

apply_orientation_bucket_gate = _MODULE.apply_orientation_bucket_gate
orientation_bucket = _MODULE.orientation_bucket
projected_2d_direction_bucket = _MODULE.projected_2d_direction_bucket
projected_2d_angle_deg = _MODULE.projected_2d_angle_deg


def _joints(vector):
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.5
    joints[9] = joints[0] + np.asarray(vector, dtype=np.float32)
    return joints


def _keypoints_2d(vector):
    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0
    keypoints[9, :2] = keypoints[0, :2] + np.asarray(vector, dtype=np.float32)
    return keypoints


def _output(mint_vector, hmr_vector, mint_2d_vector=None, hmr_2d_vector=None):
    if mint_2d_vector is not None:
        mint_keypoints_2d = _keypoints_2d(mint_2d_vector).tolist()
    else:
        mint_keypoints_2d = None
    return SimpleNamespace(
        frame_idx=7,
        img_path="frame.jpg",
        hand_side="right",
        pred_joints_3d=_joints(hmr_vector),
        pred_keypoints_2d=_keypoints_2d(hmr_2d_vector) if hmr_2d_vector is not None else None,
        camera_joints_3d=None,
        raw_backend_meta={
            "physical_track_id": 1,
            "physical_track_fragment_id": 0,
            "keypoints_2d": mint_keypoints_2d,
            "meta": {
                "joints_3d_camera": _joints(mint_vector).tolist(),
                "joint_order": "openpose21",
                "camera_frame": "opencv_x_right_y_down_z_forward",
            },
        },
    )


def test_orientation_bucket_uses_y_for_up_and_z_for_front():
    assert orientation_bucket(np.array([0.01, -0.2, 0.3])) == "up_front"
    assert orientation_bucket(np.array([0.01, 0.2, -0.3])) == "down_back"


def test_orientation_bucket_uses_dominant_x_over_z_for_left_right():
    assert orientation_bucket(np.array([0.4, -0.2, 0.1])) == "up_right"
    assert orientation_bucket(np.array([-0.4, 0.2, 0.1])) == "down_left"


def test_projected_2d_direction_bucket_uses_image_xy_axes():
    assert projected_2d_direction_bucket(np.array([0.0, -4.0])) == "up"
    assert projected_2d_direction_bucket(np.array([0.0, 4.0])) == "down"
    assert projected_2d_direction_bucket(np.array([-4.0, 1.0])) == "left"
    assert projected_2d_direction_bucket(np.array([4.0, 1.0])) == "right"


def test_gate_keeps_hamer_when_mint_and_hamer_share_bucket():
    output = _output(mint_vector=[0.05, -0.2, 0.5], hmr_vector=[0.01, -0.1, 0.2])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == [output]
    assert report["counts"] == {"evaluated": 1, "accepted": 1, "rejected": 0, "not_evaluable": 0}
    assert output.raw_backend_meta["orientation_bucket_gate"]["status"] == "accepted"


def test_gate_rejects_hamer_when_mint_and_hamer_buckets_differ():
    output = _output(mint_vector=[0.0, -0.2, 0.5], hmr_vector=[0.5, -0.2, 0.0])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["counts"] == {"evaluated": 1, "accepted": 0, "rejected": 1, "not_evaluable": 0}
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "up_right"
    assert row["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_restores_3d_bucket_mismatch_when_projected_2d_direction_matches():
    output = _output(
        mint_vector=[0.0, -0.2, 0.5],
        hmr_vector=[0.5, -0.2, 0.0],
        mint_2d_vector=[20.0, 3.0],
        hmr_2d_vector=[12.0, 2.0],
    )

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == [output]
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "up_right"
    assert row["mint_projected_2d_bucket"] == "right"
    assert row["hmr_projected_2d_bucket"] == "right"
    assert row["classification"] == "accepted"
    assert row["accept_reason"] == "projected_2d_direction_match"


def test_gate_keeps_rejecting_3d_bucket_mismatch_when_projected_2d_direction_differs():
    output = _output(
        mint_vector=[0.0, -0.2, 0.5],
        hmr_vector=[0.5, -0.2, 0.0],
        mint_2d_vector=[20.0, 3.0],
        hmr_2d_vector=[2.0, -12.0],
    )

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    row = report["frames"][0]
    assert row["mint_projected_2d_bucket"] == "right"
    assert row["hmr_projected_2d_bucket"] == "up"
    assert row["classification"] == "reject"
    assert row["reject_reason"] == "orientation_bucket_mismatch"


def test_projected_2d_angle_reports_small_difference_across_bucket_boundary():
    angle = projected_2d_angle_deg(
        [106.62738037109375, -91.9813232421875],
        [90.49412643459323, -92.40380220134591],
    )

    assert angle == np.float64(angle)
    assert angle < 5.0


def test_gate_restores_3d_bucket_mismatch_when_projected_2d_angle_is_small():
    output = _output(
        mint_vector=[0.0, -0.2, 0.5],
        hmr_vector=[0.5, -0.2, 0.0],
        mint_2d_vector=[106.62738037109375, -91.9813232421875],
        hmr_2d_vector=[90.49412643459323, -92.40380220134591],
    )

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == [output]
    row = report["frames"][0]
    assert row["mint_projected_2d_bucket"] == "right"
    assert row["hmr_projected_2d_bucket"] == "up"
    assert row["projected_2d_angle_deg"] < 5.0
    assert row["classification"] == "accepted"
    assert row["accept_reason"] == "projected_2d_angle_match"


def test_gate_keeps_rejecting_3d_bucket_mismatch_when_projected_2d_angle_is_large():
    output = _output(
        mint_vector=[0.0, -0.2, 0.5],
        hmr_vector=[0.5, -0.2, 0.0],
        mint_2d_vector=[20.0, 3.0],
        hmr_2d_vector=[-3.0, -20.0],
    )

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["frames"][0]["projected_2d_angle_deg"] > 15.0
    assert report["frames"][0]["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_accepts_up_down_mismatch_inside_shared_horizontal_zone():
    output = _output(mint_vector=[0.0, -0.1, 0.5], hmr_vector=[0.0, 0.1, 0.5])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == [output]
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "down_front"
    assert row["classification"] == "accepted"
    assert row["accept_reason"] == "vertical_shared_zone"


def test_gate_rejects_up_down_mismatch_outside_shared_horizontal_zone():
    output = _output(mint_vector=[0.0, -0.5, 0.5], hmr_vector=[0.0, 0.5, 0.5])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["frames"][0]["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_rejects_front_right_mismatch_even_near_bucket_boundary():
    output = _output(mint_vector=[0.75, -0.1, 1.0], hmr_vector=[1.0, -0.1, 0.75])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "up_right"
    assert row["classification"] == "reject"
    assert row["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_rejects_front_right_mismatch_outside_horizontal_shared_zone():
    output = _output(mint_vector=[0.2, -0.1, 1.0], hmr_vector=[1.0, -0.1, 0.2])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["frames"][0]["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_does_not_share_front_with_back():
    output = _output(mint_vector=[0.0, -0.1, 1.0], hmr_vector=[0.0, -0.1, -1.0])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["frames"][0]["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_rejects_same_vertical_when_horizontal_buckets_differ_even_if_steep():
    output = _output(mint_vector=[-0.013, -0.067, 0.066], hmr_vector=[0.012, -0.075, 0.009])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "up_right"
    assert row["classification"] == "reject"
    assert row["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_rejects_frame_694_like_horizontal_bucket_mismatch():
    output = _output(
        mint_vector=[0.0419867560, -0.0618945956, 0.0550921559],
        hmr_vector=[0.0485539214, -0.0891623984, -0.0112041198],
    )

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    row = report["frames"][0]
    assert row["mint_bucket"] == "up_front"
    assert row["hmr_bucket"] == "up_right"
    assert row["classification"] == "reject"
    assert row["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_rejects_steep_vectors_with_opposite_vertical_direction():
    output = _output(mint_vector=[0.01, -0.08, 0.01], hmr_vector=[0.01, 0.08, 0.01])

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == []
    assert report["frames"][0]["reject_reason"] == "orientation_bucket_mismatch"


def test_gate_does_not_reject_when_mint_orientation_is_missing():
    output = _output(mint_vector=[0.0, -0.2, 0.5], hmr_vector=[0.5, -0.2, 0.0])
    output.raw_backend_meta["meta"].pop("joints_3d_camera")

    kept, report = apply_orientation_bucket_gate([output])

    assert kept == [output]
    assert report["counts"] == {"evaluated": 0, "accepted": 1, "rejected": 0, "not_evaluable": 1}
    assert report["frames"][0]["not_evaluable_reason"] == "missing_mint_orientation"
