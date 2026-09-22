from types import SimpleNamespace
from unittest.mock import Mock
import json

import cv2
import numpy as np
import pytest
import torch
import mesh_recovery

from hmr_backends.runners.schema import BackendOutputInstance
from hmr_backends.runners.batch_runner import restore_keypoints_from_flipped_crops
from hmr_backends.runners.hawor_runner import project_hawor_joints
from hmr_backends.utils.mint_3d_consistency import (
    apply_mint_3d_consistency_gate,
    attach_depth_anchors,
    convert_hmr_to_camera_joints,
    convert_mint_to_camera_joints,
)


def _joints() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.6
    joints[5] = [0.04, 0.02, 0.6]
    joints[9] = [0.0, 0.06, 0.6]
    joints[17] = [-0.04, 0.02, 0.6]
    return joints


def _output(mint_joints=None, hmr_joints=None):
    intrinsics = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    def project(joints):
        joints = np.asarray(joints)
        return np.column_stack((
            intrinsics[0, 0] * joints[:, 0] / joints[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * joints[:, 1] / joints[:, 2] + intrinsics[1, 2],
            np.ones(21),
        ))
    meta = {
        "physical_track_id": 1,
        "physical_track_fragment_id": 0,
        "confidence": 0.9,
        "meta": {},
        "depth_anchor": {
            "camera_intrinsics": intrinsics.tolist(),
            "mint_wrist_depth_m": float(mint_joints[0, 2]) if mint_joints is not None else None,
            "hmr_wrist_depth_m": float(hmr_joints[0, 2]) if hmr_joints is not None else None,
        },
    }
    if mint_joints is not None:
        meta["meta"].update({
            "joints_3d_camera": np.asarray(mint_joints).tolist(),
            "camera_frame": "opencv_x_right_y_down_z_forward",
            "joint_order": "openpose21",
            "camera_intrinsics": intrinsics.tolist(),
        })
        meta["keypoints_2d"] = project(mint_joints)
    return SimpleNamespace(
        frame_idx=3,
        img_path="frame.jpg",
        hand_side="right",
        pred_joints_3d=None if hmr_joints is None else np.asarray(hmr_joints),
        pred_keypoints_2d=None if hmr_joints is None else project(hmr_joints),
        camera_joints_3d=None,
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
    output = _output(joints, joints)
    output.cam_trans = np.array([0.0, 0.0, 50.0], dtype=np.float32)
    intrinsics = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    projected = np.column_stack((
        intrinsics[0, 0] * joints[:, 0] / joints[:, 2] + intrinsics[0, 2],
        intrinsics[1, 1] * joints[:, 1] / joints[:, 2] + intrinsics[1, 2],
        np.ones(21),
    ))
    output.pred_keypoints_2d = projected

    assert np.allclose(convert_mint_to_camera_joints(output.raw_backend_meta), joints)
    assert np.allclose(
        convert_hmr_to_camera_joints(
            output,
            wrist_depth_m=float(joints[0, 2]),
            camera_intrinsics=intrinsics,
        ),
        joints,
    )


def test_depth_anchors_can_compensate_surface_depth_to_wrist_center(tmp_path):
    depth_dir = tmp_path / "depth" / "fast_foundation" / "depth_uint16_png"
    depth_dir.mkdir(parents=True)
    depth = np.full((40, 60), 500, dtype=np.uint16)
    assert cv2.imwrite(str(depth_dir / "frame_000000_depth_mm.png"), depth)
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((40, 60, 3), dtype=np.uint8))
    joints = _joints()
    output = _output(joints, joints)
    output.frame_idx = 0
    output.img_path = str(image_path)
    output.raw_backend_meta["keypoints_2d"][:, :2] = [20.0, 20.0]
    output.pred_keypoints_2d[:, :2] = [20.0, 20.0]

    attach_depth_anchors(
        [output], tmp_path / "depth", 1, wrist_surface_compensation=True,
        wrist_surface_offset_min_m=0.02, wrist_surface_offset_max_m=0.02,
    )

    anchor = output.raw_backend_meta["depth_anchor"]
    assert anchor["mint_wrist_surface_depth_m"] == pytest.approx(0.5)
    assert anchor["hmr_wrist_surface_depth_m"] == pytest.approx(0.5)
    assert anchor["mint_wrist_depth_m"] == pytest.approx(0.52)
    assert anchor["hmr_wrist_depth_m"] == pytest.approx(0.52)
    assert anchor["mint_wrist_surface_offset_m"] == pytest.approx(0.02)
    assert anchor["hmr_wrist_surface_offset_m"] == pytest.approx(0.02)
    assert output.camera_joints_3d[0, 2] == pytest.approx(0.52)


def test_legacy_hmr_conversion_is_preserved_without_sensor_anchor():
    joints = _joints()
    output = _output(None, joints)
    output.raw_backend_meta.pop("depth_anchor")
    output.cam_trans = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    converted = convert_hmr_to_camera_joints(output)

    assert np.allclose(converted, joints + output.cam_trans)


def test_left_crop_keypoints_are_unflipped_before_image_mapping():
    keypoints = np.array([[[20.0, 10.0], [80.0, 10.0]],
                          [[20.0, 10.0], [80.0, 10.0]]])

    restored = restore_keypoints_from_flipped_crops(
        keypoints, np.array([0.0, 1.0]), 100
    )

    assert np.allclose(restored[0, :, 0], [80.0, 20.0])
    assert np.allclose(restored[1, :, 0], [20.0, 80.0])


def test_hawor_missing_model_2d_is_projected_from_recovered_geometry():
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 2] = 0.5
    joints[0, 0] = 0.1

    right = project_hawor_joints(
        joints, [0.0, 0.0, 0.5], 100.0, [50.0, 40.0], is_left=False
    )
    left = project_hawor_joints(
        joints, [0.0, 0.0, 0.5], 100.0, [50.0, 40.0], is_left=True
    )

    assert np.allclose(right[0], [60.0, 40.0, 1.0])
    assert np.allclose(left[0], [40.0, 40.0, 1.0])


def test_smoothed_geometry_reprojects_before_final_depth_anchor(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    output = _output(None, _joints())
    output.img_path = str(image_path)
    output.cam_trans = np.array([0.0, 0.0, 1.0])
    output.pred_keypoints_2d[:] = 0.0
    bundle = SimpleNamespace(
        backend_name="hamer",
        model_cfg=SimpleNamespace(
            EXTRA=SimpleNamespace(FOCAL_LENGTH=100.0),
            MODEL=SimpleNamespace(IMAGE_SIZE=200.0),
        ),
    )

    mesh_recovery._reproject_smoothed_outputs([output], bundle)

    assert np.allclose(output.pred_keypoints_2d[0], [100.0, 50.0, 1.0])
    assert output.pred_keypoints_2d[9, 1] > 50.0


def test_depth_anchor_prefers_sensor_calibration_over_mint_fov(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    depth_root = tmp_path / "depth"
    depth_dir = depth_root / "fast_foundation" / "depth_uint16_png"
    depth_dir.mkdir(parents=True)
    assert cv2.imwrite(
        str(depth_dir / "frame_000000_depth_mm.png"),
        np.full((32, 32), 600, dtype=np.uint16),
    )
    (depth_root / "fast_foundation" / "fast_foundation_stereo_video_meta.json").write_text(
        json.dumps({"intrinsics": {"fx": 80.0, "fy": 81.0, "cx": 15.0, "cy": 16.0}})
    )
    joints = _joints()
    output = _output(joints, joints)
    output.frame_idx = 0
    output.img_path = str(image_path)
    output.raw_backend_meta["meta"]["camera_intrinsics"] = [
        [700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]
    ]
    output.raw_backend_meta["keypoints_2d"] = np.tile([16.0, 16.0, 1.0], (21, 1))
    output.pred_keypoints_2d = np.tile([16.0, 16.0, 1.0], (21, 1))

    attach_depth_anchors([output], depth_root, 1)

    anchor = output.raw_backend_meta["depth_anchor"]
    assert np.allclose(anchor["camera_intrinsics"], [
        [80.0, 0.0, 15.0], [0.0, 81.0, 16.0], [0.0, 0.0, 1.0]
    ])
    assert anchor["camera_intrinsics_source"].endswith("fast_foundation_stereo_video_meta.json")


def test_mint_converter_does_not_guess_unsupported_conventions():
    output = _output(_joints(), _joints())
    output.raw_backend_meta["meta"]["joint_order"] = "unknown21"

    assert convert_mint_to_camera_joints(output.raw_backend_meta) is None


def test_gate_disabled_keeps_backend_output_in_pipeline(tmp_path, monkeypatch):
    import mesh_recovery

    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    mint = _joints()
    hmr = mint + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    intrinsics = np.array([[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]])
    def project(joints):
        return np.column_stack((
            intrinsics[0, 0] * joints[:, 0] / joints[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * joints[:, 1] / joints[:, 2] + intrinsics[1, 2],
            np.ones(21),
        ))
    depth_dir = tmp_path / "depth/fast_foundation/depth_uint16_png"
    depth_dir.mkdir(parents=True)
    assert cv2.imwrite(str(depth_dir / "frame_000000_depth_mm.png"),
                       np.full((32, 32), 600, dtype=np.uint16))
    observation = {
        "observation_id": "right-0",
        "handedness": "right",
        "backend_handedness": "right",
        "bbox_xyxy": [2, 2, 30, 30],
        "keypoints_2d": project(mint),
        "physical_track_id": 1,
        "physical_track_fragment_id": 0,
        "confidence": 1.0,
        "source": "mint",
        "meta": {"joints_3d_camera": mint.tolist(),
                 "camera_frame": "opencv_x_right_y_down_z_forward",
                 "joint_order": "openpose21",
                 "camera_intrinsics": intrinsics.tolist()},
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
            pred_keypoints_2d=project(hmr),
            pred_joints_3d=hmr,
            raw_backend_meta={**observation, "backend": "hamer"},
        )
        monkeypatch.setattr(mesh_recovery, "build_runner", lambda *_args: SimpleNamespace(infer=lambda _inputs: [output]))
        args = SimpleNamespace(
            img_focal=None, render=False, mint_3d_consistency_gate=enabled,
            mint_wrist_distance_max_m=0.08, mint_wrist_vector_angle_max_deg=40.0,
            mint_hand_scale_min=0.7, mint_hand_scale_max=1.3,
            endpoint_wrist_gate=False, temporal_smoother=False,
            hand_tracking_parquet=False, depth_dir=str(tmp_path / "depth"),
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
