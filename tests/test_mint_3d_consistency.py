from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from hmr_backends.runners.schema import BackendOutputInstance
from hmr_backends.utils.mint_3d_consistency import (
    apply_mint_3d_consistency_gate,
    convert_hmr_to_camera_joints,
    convert_mint_to_camera_joints,
)


def _joints() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = [0.0, 0.0, 0.6]
    joints[5] = [0.04, 0.02, 0.6]
    joints[9] = [0.0, 0.06, 0.6]
    joints[17] = [-0.04, 0.02, 0.6]
    return joints


def _output(mint_joints=None, hmr_joints=None):
    meta = {
        "physical_track_id": 1,
        "physical_track_fragment_id": 0,
        "confidence": 0.9,
        "meta": {},
    }
    if mint_joints is not None:
        meta["meta"].update({
            "joints_3d_camera": np.asarray(mint_joints).tolist(),
            "camera_frame": "opencv_x_right_y_down_z_forward",
            "joint_order": "openpose21",
        })
    return SimpleNamespace(
        frame_idx=3,
        img_path="frame.jpg",
        hand_side="right",
        pred_joints_3d=None if hmr_joints is None else np.asarray(hmr_joints),
        cam_trans=np.zeros(3, dtype=np.float32),
        raw_backend_meta=meta,
    )


def test_identical_mint_and_hmr_joints_pass():
    joints = _joints()
    output = _output(joints, joints)

    kept, report = apply_mint_3d_consistency_gate([output])

    assert kept == [output]
    assert report["hands"][0]["accepted"] is True
    assert report["hands"][0]["wrist_distance_m"] == 0.0
    assert report["hands"][0]["vector_angle_deg"] == 0.0
    assert report["hands"][0]["scale_ratio"] == 1.0
    assert output.raw_backend_meta["mint_3d_consistency"]["accepted"] is True


def test_large_wrist_translation_fails():
    mint = _joints()
    hmr = mint + np.array([0.09, 0.0, 0.0], dtype=np.float32)

    kept, report = apply_mint_3d_consistency_gate([_output(mint, hmr)])

    assert kept == []
    assert "wrist_distance" in report["hands"][0]["reject_reason"]


def test_large_wrist_to_palm_rotation_fails():
    mint = _joints()
    hmr = mint.copy()
    wrist = hmr[0].copy()
    for index in (5, 9, 17):
        vector = hmr[index] - wrist
        hmr[index] = wrist + np.array([-vector[1], vector[0], vector[2]])

    kept, report = apply_mint_3d_consistency_gate([_output(mint, hmr)])

    assert kept == []
    assert report["hands"][0]["vector_angle_deg"] > 40.0
    assert "wrist_vector_angle" in report["hands"][0]["reject_reason"]


def test_opposing_mcp_errors_cannot_cancel_in_palm_metric():
    mint = _joints()
    hmr = mint.copy()
    hmr[5], hmr[17] = mint[17].copy(), mint[5].copy()

    kept, report = apply_mint_3d_consistency_gate([_output(mint, hmr)])

    assert kept == []
    angles = report["hands"][0]["vector_angles_deg"]
    assert angles["index_mcp"] > 40.0
    assert angles["pinky_mcp"] > 40.0


def test_missing_mint_reference_passes():
    output = _output(None, _joints())

    kept, report = apply_mint_3d_consistency_gate([output])

    assert kept == [output]
    assert report["hands"][0]["accepted"] is True
    assert report["hands"][0]["reject_reason"] is None
    assert report["hands"][0]["status"] == "missing_mint_reference"


def test_missing_hmr_geometry_is_removed_so_mint_can_be_kept():
    output = _output(_joints(), None)

    kept, report = apply_mint_3d_consistency_gate([output])

    assert kept == []
    assert report["hands"][0]["accepted"] is False
    assert report["hands"][0]["reject_reason"] == "missing_hmr_geometry"


def test_joint_converters_return_camera_coordinates():
    joints = _joints()
    output = _output(joints, joints - np.array([0.1, 0.2, 0.3], dtype=np.float32))
    output.cam_trans = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    assert np.allclose(convert_mint_to_camera_joints(output.raw_backend_meta), joints)
    assert np.allclose(convert_hmr_to_camera_joints(output), joints)


def test_mint_converter_does_not_guess_unsupported_conventions():
    output = _output(_joints(), _joints())
    output.raw_backend_meta["meta"]["joint_order"] = "unknown21"

    assert convert_mint_to_camera_joints(output.raw_backend_meta) is None


def test_gate_disabled_keeps_backend_output_in_pipeline(tmp_path, monkeypatch):
    import mesh_recovery

    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    mint = _joints()
    hmr = mint + np.array([0.2, 0.0, 0.0], dtype=np.float32)
    observation = {
        "observation_id": "right-0",
        "handedness": "right",
        "backend_handedness": "right",
        "bbox_xyxy": [2, 2, 30, 30],
        "keypoints_2d": np.zeros((21, 3), dtype=np.float32),
        "physical_track_id": 1,
        "physical_track_fragment_id": 0,
        "confidence": 1.0,
        "source": "mint",
        "meta": {"joints_3d_camera": mint.tolist(),
                 "camera_frame": "opencv_x_right_y_down_z_forward",
                 "joint_order": "openpose21"},
    }
    frames = [{"frame_idx": 0, "img_path": str(image_path), "hands": [observation]}]

    def run(enabled: bool, output_dir):
        output = BackendOutputInstance(
            frame_idx=0,
            img_path=str(image_path),
            hand_side="right",
            mano_params={"global_orient": np.eye(3), "hand_pose": np.tile(np.eye(3), (15, 1, 1)),
                         "betas": np.zeros(10), "is_right": 1},
            cam_trans=np.zeros(3),
            pred_vertices=np.zeros((778, 3)),
            pred_keypoints_2d=np.zeros((21, 3)),
            pred_joints_3d=hmr,
            raw_backend_meta={**observation, "backend": "hamer"},
        )
        monkeypatch.setattr(mesh_recovery, "build_runner", lambda *_args: SimpleNamespace(infer=lambda _inputs: [output]))
        args = SimpleNamespace(
            img_focal=None, render=False, mint_3d_consistency_gate=enabled,
            mint_wrist_distance_max_m=0.08, mint_wrist_vector_angle_max_deg=40.0,
            mint_hand_scale_min=0.7, mint_hand_scale_max=1.3,
            endpoint_wrist_gate=False, temporal_smoother=False,
        )
        return mesh_recovery.run_mesh_recovery(
            frames, SimpleNamespace(backend_name="hamer"), Mock(), args,
            output_dir, device=torch.device("cpu"),
        )

    disabled = run(False, tmp_path / "disabled")
    enabled = run(True, tmp_path / "enabled")

    assert len(disabled[str(image_path)]["mano"]) == 1
    assert enabled[str(image_path)]["mano"] == []
    assert (tmp_path / "enabled/stages/45_mint_3d_consistency/mint_3d_consistency.json").is_file()
