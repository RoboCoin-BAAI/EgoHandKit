from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from mesh_recovery import _assemble_results, _collect_inputs, _recover_partial_outputs
from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet
from hmr_backends.utils.mint_3d_consistency import convert_hmr_to_camera_joints, apply_mint_3d_consistency_gate
from hmr_backends.utils.partial_hand_recovery import attach_partial_hand_anchors, visible_hand


def sample(tmp_path, side='right'):
    path = tmp_path / 'image.jpg'
    cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))
    joints = np.zeros((21, 3))
    joints[:, 0] = np.linspace(0, 0.08, 21)
    joints[:, 1] = np.linspace(0, 0.06, 21)
    points = np.tile([30., 40., 1.], (21, 1))
    points[0] = [-10, 40, 1]
    observation = {'handedness': side, 'physical_track_id': 0, 'physical_track_fragment_id': 0,
                   'bbox_xyxy': [-20, 20, 60, 80], 'keypoints_2d': points.copy(), 'source': 'mint',
                   'meta': {'force_frontend_fallback': True, 'joints_3d_camera': joints + [0.1, 0.2, 0.7],
                            'camera_frame': 'opencv_x_right_y_down_z_forward', 'joint_order': 'openpose21'}}
    metadata = {**observation, 'backend': 'hamer', 'depth_anchor': {
        'camera_intrinsics': [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        'mint_wrist_depth_m': None, 'hmr_wrist_depth_m': None}}
    output = SimpleNamespace(frame_idx=0, img_path=str(path), hand_side=side,
                             pred_joints_3d=joints, pred_keypoints_2d=points,
                             raw_backend_meta=metadata, camera_joints_3d=None,
                             cam_trans=np.zeros(3), mano_params={})
    frames = [{'frame_idx': 0, 'img_path': str(path), 'hands': [observation]}]
    return output, frames


def mock_depth(monkeypatch, samples):
    monkeypatch.setattr('hmr_backends.utils.partial_hand_recovery.resolve_depth_frames', lambda *a: ['unused'])
    sequence = iter(samples)
    monkeypatch.setattr('hmr_backends.utils.partial_hand_recovery.depth_for_image_point', lambda *a: next(sequence))


@pytest.mark.parametrize('side', ['left', 'right'])
def test_unavailable_sensor_uses_mint_wrist_but_keeps_hmr_shape(tmp_path, monkeypatch, side):
    output, _ = sample(tmp_path, side)
    mock_depth(monkeypatch, [None] * 4)
    report = attach_partial_hand_anchors([output], tmp_path, 1)
    result = convert_hmr_to_camera_joints(output)
    expected = output.pred_joints_3d.copy()
    if side == 'left':
        expected[:, 0] *= -1
    np.testing.assert_allclose(result - result[:1], expected, atol=1e-8)
    np.testing.assert_allclose(result[0], [0.1, 0.2, 0.7])
    assert report[0]['source'] == 'mint_frontend_wrist'
    assert not report[0]['sensor_calibrated']
    kept, diagnostic = apply_mint_3d_consistency_gate([output], wrist_distance_max_m=0.00001)
    assert kept == [output]
    assert diagnostic['hands'][0]['status'] == 'partial_hand_assisted'


def test_visible_mcp_depth_estimates_wrist(tmp_path, monkeypatch):
    output, _ = sample(tmp_path)
    mock_depth(monkeypatch, [0.49, 0.5, 0.51, None])
    report = attach_partial_hand_anchors([output], tmp_path, 1)
    assert report[0]['source'] == 'visible_mcp_sensor'
    np.testing.assert_allclose(convert_hmr_to_camera_joints(output)[0], [-0.3, -0.05, 0.5])


def test_inconsistent_palm_depth_does_not_anchor_to_background(tmp_path, monkeypatch):
    output, _ = sample(tmp_path)
    mock_depth(monkeypatch, [0.4, 0.8, 0.6, 0.4])
    report = attach_partial_hand_anchors([output], tmp_path, 1)
    assert report[0]['source'] == 'mint_frontend_wrist'


def test_depth_limit_excludes_both_backend_and_frontend(tmp_path):
    output, frames = sample(tmp_path)
    output.raw_backend_meta['depth_anchor']['hmr_wrist_depth_m'] = 1.01
    args = SimpleNamespace(hmr_partial_hand_recovery=True, depth_dir=tmp_path,
                           hmr_partial_depth_spread_max_m=0.08, depth_gate=True, depth_max_m=1.)
    outputs, selected = _recover_partial_outputs([output], frames, args, tmp_path)
    assert not outputs and not selected[0]['hands']
    assert len(frames[0]['hands']) == 1
    parquet = export_hand_tracking_parquet(selected, _assemble_results(outputs, selected), tmp_path / 'tracks.parquet')
    assert not pq.read_table(parquet).to_pylist()[0]['right_present']


@pytest.mark.parametrize('enabled,expected', [(False, 0), (True, 1)])
def test_partial_input_switch_and_visibility(tmp_path, enabled, expected):
    _, frames = sample(tmp_path)
    args = SimpleNamespace(hmr_partial_hand_recovery=enabled)
    assert len(_collect_inputs(frames, args, 'hamer').instances) == expected
    assert visible_hand(frames[0]['hands'][0], (100, 100))
    frames[0]['hands'][0]['keypoints_2d'][:, 0] = -100
    assert not _collect_inputs(frames, args, 'hamer').instances


def test_missing_geometry_still_falls_back_to_mint(tmp_path):
    output, frames = sample(tmp_path)
    output.pred_joints_3d = None
    attach_partial_hand_anchors([output], tmp_path, 1)
    kept, _ = apply_mint_3d_consistency_gate([output])
    assert kept == []
    result = pq.read_table(export_hand_tracking_parquet(
        frames, {}, tmp_path / 'tracks.parquet')).to_pylist()[0]
    assert result['right_present']
    assert result['source'] == 'mint_frontend_fallback'


def test_export_marks_hmr_shape_mint_position(tmp_path, monkeypatch):
    output, frames = sample(tmp_path)
    mock_depth(monkeypatch, [None] * 4)
    attach_partial_hand_anchors([output], tmp_path, 1)
    results = _assemble_results([output], frames)
    row = pq.read_table(export_hand_tracking_parquet(frames, results, tmp_path / 'tracks.parquet')).to_pylist()[0]
    assert row['source'] == 'hamer_mint_frontend_wrist'
    np.testing.assert_allclose(row['right_joints_3d'], convert_hmr_to_camera_joints(output))


def test_mint_measured_wrist_precedes_palm_depth_estimation(tmp_path, monkeypatch):
    output, _ = sample(tmp_path)
    output.raw_backend_meta['depth_anchor']['mint_wrist_depth_m'] = 0.6
    monkeypatch.setattr('hmr_backends.utils.partial_hand_recovery.resolve_depth_frames',
                        lambda *a: pytest.fail('Should not sample palm when a measured wrist is available'))
    report = attach_partial_hand_anchors([output], tmp_path, 1)
    assert report[0]['source'] == 'mint_sensor_wrist'
    assert report[0]['sensor_calibrated']
    assert convert_hmr_to_camera_joints(output)[0, 2] == 0.6


def test_no_mint_or_sensor_reference_does_not_invent_translation(tmp_path, monkeypatch):
    output, _ = sample(tmp_path)
    output.raw_backend_meta['meta'].pop('joints_3d_camera')
    mock_depth(monkeypatch, [None] * 4)
    attach_partial_hand_anchors([output], tmp_path, 1)
    assert convert_hmr_to_camera_joints(output) is None


def test_partial_recovery_flag_validates_dependencies():
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    args = parser.parse_args(['--input', 'unused', '--hmr_partial_hand_recovery'])
    with pytest.raises(SystemExit):
        validate_cli_args(parser, args)
    args = parser.parse_args(['--input', 'unused', '--frontend', 'canonical',
                              '--observations', 'unused.pkl', '--depth_gate',
                              '--mint_depth_sensor_anchor', '--depth_dir', 'depth',
                              '--hmr_partial_hand_recovery'])
    validate_cli_args(parser, args)
    assert args.hand_tracking_parquet
    assert not args.mint_3d_consistency_gate


def test_disabled_recovery_does_not_change_existing_outputs(tmp_path):
    output, frames = sample(tmp_path)
    outputs = [output]
    kept, selected = _recover_partial_outputs(outputs, frames, SimpleNamespace(hmr_partial_hand_recovery=False), tmp_path)
    assert kept is outputs
    assert selected is frames
    assert 'partial_hand_recovery' not in output.raw_backend_meta


def test_measured_wrist_at_one_metre_is_not_rejected(tmp_path):
    output, _ = sample(tmp_path)
    output.raw_backend_meta['depth_anchor']['hmr_wrist_depth_m'] = 1.0
    report = attach_partial_hand_anchors([output], tmp_path, 1, max_depth_m=1.0)
    assert report[0]['status'] == 'recovered'
    assert convert_hmr_to_camera_joints(output)[0, 2] == 1.0
