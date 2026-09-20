import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hmr_backends.utils.hand_identity import DuplicateImageConfig, diagnose_image_duplicates
from hmr_backends.utils.mint_3d_consistency import apply_mint_3d_consistency_gate


def captured_pair():
    """Numeric-only capture of the two predictions at frame 599 (19.967s)."""
    path = Path(__file__).parent / 'fixtures/overlapping_hmr_hands.json'
    rows = json.loads(path.read_text())
    for row in rows:
        for key in ('pred_joints_3d', 'pred_keypoints_2d', 'cam_trans'):
            row[key] = np.asarray(row[key], dtype=float)
    return [SimpleNamespace(**row, img_path='frame.jpg', mano_params={}) for row in rows]


def test_real_overlap_uses_positions_even_when_mint_depth_is_untrusted():
    outputs = captured_pair()
    kept, report = apply_mint_3d_consistency_gate(
        outputs, hmr_primary=True, duplicate_image_gate=True)
    assert kept == [outputs[1]]
    assert not report['hands'][1]['reference_quality']['trusted']
    assert report['hands'][1]['position_reference_quality']['trusted']
    assert report['hands'][0]['reject_reason'] == 'duplicate_hand_image_depth'
    pair = report['image_pair_diagnostics'][0]
    assert pair['hmr_bbox_iou'] >= 0.5
    assert pair['hmr_wrist_depth_difference_m'] < 0.06
    assert pair['nearest_reference_side'] == {'left': 'right', 'right': 'right'}
    assert pair['rejected_side'] == 'left'


def test_disabled_image_gate_keeps_previous_behavior():
    outputs = captured_pair()
    kept, report = apply_mint_3d_consistency_gate(outputs, hmr_primary=True)
    assert len(kept) == 2
    assert report['image_pair_diagnostics'] == []


def test_position_check_does_not_require_mint_sensor_depth():
    outputs = captured_pair()
    for output in outputs:
        output.raw_backend_meta['depth_anchor']['mint_wrist_depth_m'] = None
        output.raw_backend_meta['depth_anchor']['mint_mcp_wrist_depths_m'] = []
    assert diagnose_image_duplicates(outputs, DuplicateImageConfig())[0]['rejected_side'] == 'left'


@pytest.mark.parametrize('change', ['same_reference', 'correct_assignments', 'different_depth',
                                  'missing_hmr_depth', 'low_confidence', 'invalid_projection',
                                  'missing_fallback', 'ambiguous_tracks', 'missing_side'])
def test_uncertain_or_real_overlap_is_not_automatically_rejected(change):
    outputs = captured_pair()
    if change == 'same_reference':
        outputs[1].raw_backend_meta['keypoints_2d'] = copy.deepcopy(outputs[0].raw_backend_meta['keypoints_2d'])
    elif change == 'correct_assignments':
        for output in outputs:
            output.pred_keypoints_2d[0] = output.raw_backend_meta['keypoints_2d'][0]
    elif change == 'different_depth':
        outputs[0].raw_backend_meta['depth_anchor']['hmr_wrist_depth_m'] = 0.7
    elif change == 'missing_hmr_depth':
        outputs[0].raw_backend_meta['depth_anchor']['hmr_wrist_depth_m'] = None
    elif change == 'low_confidence':
        outputs[0].raw_backend_meta['confidence'] = 0.1
    elif change == 'invalid_projection':
        outputs[0].pred_keypoints_2d[0][0] = float('nan')
    elif change == 'missing_fallback':
        outputs[0].raw_backend_meta['meta'].pop('joints_3d_camera')
    elif change == 'ambiguous_tracks':
        outputs.append(copy.deepcopy(outputs[0]))
    else:
        outputs.pop()
    report = diagnose_image_duplicates(outputs, DuplicateImageConfig())
    assert all(pair['rejected_side'] is None for pair in report)


def test_input_crop_overlap_is_not_used_as_output_overlap():
    outputs = captured_pair()
    for output in outputs:
        output.raw_backend_meta['bbox_xyxy'] = [0, 0, 1920, 1050]
    points = np.asarray(outputs[0].pred_keypoints_2d)
    points[1:, 0] -= 300
    outputs[0].pred_keypoints_2d = points
    pair = diagnose_image_duplicates(outputs, DuplicateImageConfig())[0]
    assert pair['hmr_bbox_iou'] < 0.5
    assert pair['rejected_side'] is None


@pytest.mark.parametrize('ambiguous', [False, True])
def test_each_hand_retains_position_quality_when_pair_is_not_evaluable(ambiguous):
    outputs = captured_pair()
    if ambiguous:
        outputs.append(copy.deepcopy(outputs[0]))
    else:
        outputs.pop()
    _, report = apply_mint_3d_consistency_gate(outputs, hmr_primary=True, duplicate_image_gate=True)
    assert all(hand['position_reference_quality']['trusted'] for hand in report['hands'])
    assert report['image_pair_diagnostics'][0]['not_evaluated_reason'] == 'missing_or_ambiguous_side'


@pytest.mark.parametrize('key', ['handedness', 'backend_handedness'])
def test_side_disagreement_cannot_veto_into_a_different_export_slot(key):
    outputs = captured_pair()
    for output in outputs:
        output.raw_backend_meta.update(handedness=output.hand_side, backend_handedness=output.hand_side)
    outputs[0].raw_backend_meta[key] = 'right'
    kept, report = apply_mint_3d_consistency_gate(outputs, hmr_primary=True, duplicate_image_gate=True)
    assert len(kept) == 2
    pair = report['image_pair_diagnostics'][0]
    assert pair['not_evaluated_reason'] == 'frontend_backend_side_disagreement'
    assert pair['rejected_side'] is None


def test_identity_decision_is_resolution_independent():
    outputs = captured_pair()
    for output in outputs:
        for points in (output.pred_keypoints_2d, output.raw_backend_meta['keypoints_2d']):
            for point in points:
                point[0] *= 0.5
                point[1] *= 0.5
    assert diagnose_image_duplicates(outputs, DuplicateImageConfig())[0]['rejected_side'] == 'left'


@pytest.mark.parametrize('extent,duplicate', [(1.1, True), (1.15, True), (1.2, False)])
def test_bbox_extent_changes_respect_overlap_threshold(extent, duplicate):
    outputs = captured_pair()
    points = outputs[0].pred_keypoints_2d
    points[:, :2] = points[0, :2] + (points[:, :2] - points[0, :2]) * extent
    pair = diagnose_image_duplicates(outputs, DuplicateImageConfig())[0]
    assert pair['hmr_bbox_iou'] < 0.5
    assert pair['duplicate_candidate'] == duplicate
    assert pair['rejected_side'] == ('left' if duplicate else None)


def test_only_wrong_side_falls_back_in_final_parquet(tmp_path):
    import pyarrow.parquet as pq
    from mesh_recovery import _assemble_results
    from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet
    outputs = captured_pair()
    for output in outputs:
        output.raw_backend_meta.update(handedness=output.hand_side, backend='hamer', source='mint')
    frames = [{'frame_idx': 599, 'img_path': 'frame.jpg',
               'hands': [output.raw_backend_meta for output in outputs]}]
    kept, _ = apply_mint_3d_consistency_gate(outputs, hmr_primary=True, duplicate_image_gate=True)
    rows = pq.read_table(export_hand_tracking_parquet(
        frames, _assemble_results(kept, frames), tmp_path / 'tracking.parquet')).to_pylist()
    assert rows[0]['left_present'] and rows[0]['right_present']
    assert rows[0]['left_source'] == 'mint'
    assert rows[0]['right_source'] == 'hamer'


def test_cli_switch_and_threshold_validation():
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    assert not parser.parse_args(['--input', 'unused']).hmr_duplicate_image_gate
    for flags in (['--hmr_duplicate_image_gate'], ['--hmr_duplicate_iou_min', '1.1'],
                  ['--hmr_duplicate_depth_max_m', 'nan'], ['--hmr_duplicate_assignment_margin', '0']):
        with pytest.raises(SystemExit):
            validate_cli_args(parser, parser.parse_args(['--input', 'unused'] + flags))
    args = validate_cli_args(parser, parser.parse_args([
        '--input', 'unused', '--mint_3d_consistency_gate', '--depth_dir', 'depth',
        '--hmr_duplicate_image_gate']))
    assert args.hmr_duplicate_image_gate


def test_cli_and_pipeline_share_duplicate_config_defaults_and_overrides():
    from mesh_recovery import _consistency_options
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    defaults = parser.parse_args(['--input', 'unused'])
    assert DuplicateImageConfig.from_args(defaults) == DuplicateImageConfig()
    assert DuplicateImageConfig.from_args(SimpleNamespace()) == DuplicateImageConfig()
    args = validate_cli_args(parser, parser.parse_args([
        '--input', 'unused', '--hmr_duplicate_iou_min', '0.55',
        '--hmr_duplicate_depth_max_m', '0.04',
        '--hmr_duplicate_reference_separation', '0.7',
        '--hmr_duplicate_assignment_margin', '0.2']))
    assert _consistency_options(args)['duplicate_image_config'] == DuplicateImageConfig(0.55, 0.04, 0.7, 0.2)
