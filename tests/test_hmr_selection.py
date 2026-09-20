import numpy as np
import pytest

from test_mint_3d_consistency import _joints, _output
from test_partial_hand_recovery import sample, mock_depth
from hmr_backends.utils.mint_3d_consistency import apply_mint_3d_consistency_gate, convert_hmr_to_camera_joints
from hmr_backends.utils.partial_hand_recovery import attach_partial_hand_anchors


def trusted(mint=None, hmr=None, frame=0):
    output = _output(_joints() if mint is None else mint, _joints() if hmr is None else hmr)
    output.frame_idx = frame
    depth = output.raw_backend_meta['depth_anchor']['mint_wrist_depth_m']
    output.raw_backend_meta['depth_anchor']['mint_mcp_wrist_depths_m'] = [depth] * 4
    return output


def test_moderate_error_and_scale_difference_keep_hmr():
    mint = _joints()
    hmr = (mint - mint[0]) * 1.5 + mint[0] + [0.10, 0, 0]
    output = trusted(mint, hmr)
    assert not apply_mint_3d_consistency_gate([output])[0]
    kept, report = apply_mint_3d_consistency_gate([output], hmr_primary=True)
    assert kept == [output]
    assert report['hands'][0]['scale_ratio'] > 1.3


@pytest.mark.parametrize('error', ['translation', 'rotation'])
def test_sustained_severe_error_rejects_before_smoothing(error):
    mint, hmr = _joints(), _joints()
    if error == 'translation':
        hmr += [0.3, 0, 0]
    else:
        hmr[:, :2] *= -1
    outputs = [trusted(mint, hmr, frame=i) for i in range(3)]
    kept, report = apply_mint_3d_consistency_gate(outputs, hmr_primary=True)
    assert not kept
    assert all(row['severe_error_confirmed'] for row in report['hands'])


def test_isolated_error_and_gaps_do_not_confirm():
    outputs = [trusted(hmr=_joints() + [0.3, 0, 0], frame=i) for i in (0, 1, 3)]
    assert len(apply_mint_3d_consistency_gate(outputs, hmr_primary=True)[0]) == 3
    outputs[2].frame_idx = 2
    outputs[2].raw_backend_meta['physical_track_fragment_id'] = 1
    assert len(apply_mint_3d_consistency_gate(outputs, hmr_primary=True)[0]) == 3


def test_background_wrist_does_not_veto_hmr():
    output = trusted(hmr=_joints() + [0.3, 0, 0])
    output.raw_backend_meta['depth_anchor']['mint_mcp_wrist_depths_m'] = [0.32] * 4
    kept, report = apply_mint_3d_consistency_gate(
        [output], hmr_primary=True, error_confirm_frames=1)
    assert kept == [output]
    assert report['hands'][0]['reference_quality']['reason'] == 'wrist_palm_depth_disagreement'


def test_unknown_reference_quality_does_not_veto_hmr():
    output = _output(_joints(), _joints() + [0.3, 0, 0])
    assert apply_mint_3d_consistency_gate([output], hmr_primary=True, error_confirm_frames=1)[0]


def pair(separated=True, duplicate=True):
    left_mint = _joints() + [-0.25, 0, 0]
    right_mint = _joints() + ([0.25, 0, 0] if separated else [-0.24, 0, 0])
    left = trusted(left_mint, left_mint)
    left.hand_side = 'left'
    # The converter mirrors raw left MANO X; projected keypoints are already unflipped.
    left.pred_joints_3d = left.pred_joints_3d.copy()
    left.pred_joints_3d[:, 0] *= -1
    right = trusted(right_mint, left_mint if duplicate else right_mint)
    return [left, right]


def test_duplicate_gate_keeps_correct_hand_and_rejects_duplicate():
    outputs = pair()
    kept, report = apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_hand_gate=True)
    assert kept == [outputs[0]]
    assert report['hands'][1]['reject_reason'] == 'duplicate_hand_on_other_side'
    assert report['pair_diagnostics'][0]['duplicate_candidate']
    assert report['pair_diagnostics'][0]['nearest_reference_side'] == {'left': 'left', 'right': 'left'}


def test_duplicate_disabled_is_diagnostic_only():
    outputs = pair()
    kept, report = apply_mint_3d_consistency_gate(outputs, hmr_primary=True)
    assert len(kept) == 2
    assert report['pair_diagnostics'][0]['decision_effect'] == 'none'


@pytest.mark.parametrize('separated,duplicate', [(False, True), (True, False)])
def test_real_crossing_and_separate_hands_are_not_duplicates(separated, duplicate):
    outputs = pair(separated, duplicate)
    kept, report = apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_hand_gate=True)
    assert len(kept) == 2
    assert not report['pair_diagnostics'][0]['duplicate_candidate']


def test_unreliable_frontend_cannot_suppress_duplicate():
    outputs = pair()
    outputs[1].raw_backend_meta['depth_anchor']['mint_mcp_wrist_depths_m'] = []
    assert len(apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_hand_gate=True)[0]) == 2


def test_ambiguous_same_side_tracks_do_not_trigger_pair_rejection():
    import copy
    outputs = pair()
    extra = copy.deepcopy(outputs[0])
    extra.raw_backend_meta['physical_track_id'] = 99
    outputs.append(extra)
    kept, report = apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_hand_gate=True)
    assert len(kept) == 3
    assert report['pair_diagnostics'][0]['not_evaluated_reason'] == 'ambiguous_same_side_tracks'
    assert report['pair_diagnostics'][0]['decision_effect'] == 'none'


@pytest.mark.parametrize('side', ['left', 'right'])
def test_borrowed_depth_preserves_hmr_wrist_ray(tmp_path, monkeypatch, side):
    output, _ = sample(tmp_path, side)
    mock_depth(monkeypatch, [None] * 4)
    report = attach_partial_hand_anchors([output], tmp_path, 1, stable_wrist_anchor=True)
    wrist = convert_hmr_to_camera_joints(output)[0]
    np.testing.assert_allclose(wrist, [-0.42, -0.07, 0.7])
    np.testing.assert_allclose([100 * wrist[0] / wrist[2] + 50,
                                100 * wrist[1] / wrist[2] + 50], [-10, 40])
    assert report[0]['wrist_ray_source'] == 'hmr'


def test_stable_anchor_prefers_local_palm_over_mint_background(tmp_path, monkeypatch):
    output, _ = sample(tmp_path)
    output.raw_backend_meta['depth_anchor']['mint_wrist_depth_m'] = 0.9
    mock_depth(monkeypatch, [0.5] * 4)
    report = attach_partial_hand_anchors([output], tmp_path, 1, stable_wrist_anchor=True)
    assert report[0]['source'] == 'visible_mcp_sensor'
    assert convert_hmr_to_camera_joints(output)[0, 2] == 0.5


def test_missing_hmr_and_missing_mint_behaviour():
    missing_hmr = _output(_joints(), None)
    missing_mint = _output(None, _joints())
    kept, _ = apply_mint_3d_consistency_gate([missing_hmr, missing_mint], hmr_primary=True)
    assert kept == [missing_mint]


def test_flags_validate_dependencies_and_default_off():
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    args = parser.parse_args(['--input', 'unused'])
    assert not args.hmr_primary_policy and not args.hmr_stable_wrist_anchor
    assert not args.hmr_duplicate_hand_gate
    for flags in (['--hmr_primary_policy'], ['--hmr_duplicate_hand_gate'],
                  ['--hmr_stable_wrist_anchor'], ['--hmr_severe_wrist_distance_m', 'nan'],
                  ['--temporal_smoother', '--final_joints_smoother']):
        with pytest.raises(SystemExit):
            validate_cli_args(parser, parser.parse_args(['--input', 'unused'] + flags))


def test_rejected_duplicate_exports_mint_without_removing_other_hand(tmp_path):
    import pyarrow.parquet as pq
    from mesh_recovery import _assemble_results
    from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet
    outputs = pair()
    frames = [{'frame_idx': 0, 'img_path': 'frame.jpg',
               'hands': [{**o.raw_backend_meta, 'handedness': o.hand_side, 'source': 'mint'}
                         for o in outputs]}]
    for output in outputs:
        output.raw_backend_meta.update(handedness=output.hand_side, backend='hamer')
        output.mano_params = {}
    kept, _ = apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_hand_gate=True)
    row = pq.read_table(export_hand_tracking_parquet(
        frames, _assemble_results(kept, frames), tmp_path / 'final.parquet')).to_pylist()[0]
    assert row['left_present'] and row['right_present']
    assert row['left_source'] == 'hamer' and row['right_source'] == 'mint'
    np.testing.assert_allclose(row['right_joints_3d'], _joints() + [0.25, 0, 0])
