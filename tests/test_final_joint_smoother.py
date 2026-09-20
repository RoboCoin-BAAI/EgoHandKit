import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from hmr_backends.utils.final_joint_smoother import smooth_final_rows
from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet
from hmr_backends.utils.mint_style_smoother import smooth_camera_joint_sequence


def test_final_smoother_cli_enables_export_not_backend_smoother():
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    args = parser.parse_args(['--input', 'unused', '--depth_dir', 'depth', '--final_joints_smoother'])
    validate_cli_args(parser, args)
    assert args.hand_tracking_parquet
    assert not args.temporal_smoother
    args.smoother_q = float('nan')
    with pytest.raises(SystemExit):
        validate_cli_args(parser, args)


def test_nonconsecutive_frames_and_short_runs_are_not_bridged():
    indices = [0, 1, 2, 8, 9, 10]
    rows = [{'frame_idx': i, 'left_present': True, 'right_present': False,
             'left_joints_3d': np.full((21, 3), 0.5 + i * 0.01).tolist()} for i in indices]
    before = [row['left_joints_3d'] for row in rows]
    report = smooth_final_rows(rows, [{'left': (0, 0)} for _ in rows])
    assert report['smoothed_hands'] == 0
    assert len(report['segments']) == 2
    assert before == [row['left_joints_3d'] for row in rows]


def test_large_depth_jump_starts_new_run_without_deleting_hands():
    rows = [{'frame_idx': i, 'left_present': True, 'right_present': False,
             'left_joints_3d': np.full((21, 3), 0.5 if i < 5 else 2.0).tolist()} for i in range(10)]
    report = smooth_final_rows(rows, [{'left': (0, 0)} for _ in rows])
    assert report['jump_boundaries'] == [{'side': 'left', 'frame_idx': 5}]
    assert len(report['segments']) == 2
    np.testing.assert_allclose(rows[4]['left_joints_3d'], 0.5)
    np.testing.assert_allclose(rows[5]['left_joints_3d'], 2.0)
    assert all(row['left_present'] for row in rows)


def test_final_camera_smoothing_reduces_jitter_without_moving_constant_hand():
    points = np.zeros((40, 21, 3), np.float32)
    points[:, :, 2] = 0.6
    points[:, :, 0] = np.linspace(0, 0.1, 21)
    np.testing.assert_allclose(smooth_camera_joint_sequence(points, np.arange(40)), points, atol=1e-6)
    points[:, :, 0] += np.random.default_rng(42).normal(0, 0.01, (40, 1))
    result = smooth_camera_joint_sequence(points, np.arange(40))
    assert np.std(np.diff(result[:, 0, 0], n=2)) < np.std(np.diff(points[:, 0, 0], n=2))
    np.testing.assert_allclose(result[:, 1:] - result[:, :1], points[:, 1:] - points[:, :1], atol=1e-6)


def test_segments_stop_at_missing_hands_and_identity_changes():
    rows, keys = [], []
    for i in range(15):
        points = np.full((21, 3), 0.5 + i * 0.01)
        rows.append({'frame_idx': i, 'left_present': i != 5, 'right_present': False,
                     'left_joints_3d': points.tolist(), 'source': 'mint' if i % 2 else 'hamer'})
        keys.append({'left': (0, 0 if i < 10 else 1)})
    missing = rows[5]['left_joints_3d']
    report = smooth_final_rows(rows, keys)
    assert [(s['start_frame'], s['end_frame']) for s in report['segments']] == [(0, 4), (6, 9), (10, 14)]
    assert rows[5]['left_joints_3d'] == missing
    assert sum(row['left_present'] for row in rows) == 14
    assert all(not row['right_present'] for row in rows)


def test_final_export_smooths_mint_and_hmr_together_and_disabled_is_unchanged(tmp_path):
    frames, results, original = [], {}, []
    for i in range(12):
        points = np.ones((21, 3), np.float32) * 0.5
        points[:, 0] += 0.02 if i % 2 else -0.02
        original.append(points)
        hand = {'handedness': 'left', 'physical_track_id': 0,
                'physical_track_fragment_id': 0, 'source': 'mint',
                'meta': {'joints_3d_camera': points,
                         'camera_frame': 'opencv_x_right_y_down_z_forward', 'joint_order': 'openpose21'}}
        frames.append({'frame_idx': i, 'img_path': str(i), 'timestamp_ns': i * 33333333, 'hands': [hand]})
        if i % 2:
            results[str(i)] = {'joints_3d': [points], 'backend_meta': [{'handedness': 'left', 'backend': 'hamer'}]}
    raw = pq.read_table(export_hand_tracking_parquet(frames, results, tmp_path / 'raw.parquet'))
    filtered = pq.read_table(export_hand_tracking_parquet(frames, results, tmp_path / 'smooth.parquet', final_smoother=True))
    raw_rows, smooth_rows = raw.to_pylist(), filtered.to_pylist()
    np.testing.assert_allclose([r['left_joints_3d'] for r in raw_rows], original)
    assert not np.allclose([r['left_joints_3d'] for r in smooth_rows], original)
    for before, after in zip(raw_rows, smooth_rows):
        assert before['source'] == after['source']
        assert before['timestamp_ns'] == after['timestamp_ns']
        assert before['left_present'] == after['left_present'] is True
        assert before['right_present'] == after['right_present'] is False
        assert np.isnan(after['right_joints_3d']).all()
    report = json.loads((tmp_path / 'final_joints_smoother.json').read_text())
    assert report['smoothed_hands'] == 12
    assert len(report['segments']) == 1
    assert b'final_joints_smoother' in filtered.schema.metadata
    assert b'final_joints_smoother' not in raw.schema.metadata
