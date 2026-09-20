from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from observation_frontend import adapter
from mesh_recovery import _assemble_results, _collect_inputs, run_mesh_recovery
from hmr_backends.runners.hawor_runner import HaworBackendRunner
from hmr_backends.runners.batch_runner import BatchBackendRunner


def observation(track=0, side='left', fragment=0, candidate=0):
    return {
        'candidate_id': candidate, 'physical_track_id': track,
        'physical_track_fragment_id': fragment, 'handedness': side,
        'dominant_track_handedness': 'right',
        'bbox_xyxy': [10 + track * 50, 20, 40 + track * 50, 60],
        'observation_id': f'{track}:{fragment}:{candidate}',
        'keypoints_2d': [[20 + track * 50, 30, 0.9]] * 21,
        'confidence': 0.9, 'source': 'observations', 'meta': {},
    }


def make_frames(tmp_path, observations):
    frames = []
    for i, selected in enumerate(observations):
        path = tmp_path / f'{i:06d}.jpg'
        assert cv2.imwrite(str(path), np.full((100, 160, 3), 80, np.uint8))
        frames.append({'frame_idx': i, 'img_path': str(path), 'timestamp_ns': None, 'hands': selected})
    return frames


def test_all_person_proposals_and_original_gate(tmp_path):
    path = tmp_path / '000000.jpg'
    cv2.imwrite(str(path), np.zeros((100, 160, 3), np.uint8))
    instances = SimpleNamespace(
        pred_classes=torch.tensor([0, 0, 0, 1]),
        scores=torch.tensor([0.99, 0.8, 0.5, 0.95]),
        pred_boxes=SimpleNamespace(tensor=torch.tensor([[0, 0, 50, 90]] * 4)),
    )
    detector = Mock(return_value={'instances': instances})
    points = np.zeros((133, 3), np.float32)
    points[-42:, :2] = np.arange(84).reshape(42, 2)
    points[-42:-39, 2] = 0.9  # Three valid left keypoints: rejected.
    points[-39, 2] = 0.5  # Equality is not sufficient.
    points[-21:-17, 2] = 0.9  # Four right keypoints: accepted.
    pose = Mock()
    pose.predict_pose.return_value = [{'keypoints': points}]
    result = adapter.extract_observations([path], detector, pose, all_person=True)
    assert pose.predict_pose.call_count == 2
    selected = result['frames'][0]['observations']
    assert [item['candidate_id'] for item in selected] == [1, 3]
    assert [item['person_index'] for item in selected] == [0, 1]
    np.testing.assert_array_equal(selected[0]['vitpose_keypoints_2d'], points[-21:])
    reliable = points[-21:][points[-21:, 2] > 0.5, :2]
    expected_raw = np.array([*reliable.min(0), *reliable.max(0)], dtype=np.float32)
    expected = adapter.enlarge_bbox(expected_raw, scale=1.2, img_shape=(100, 160, 3))
    np.testing.assert_allclose(selected[0]['bbox_xyxy'], expected)
    np.testing.assert_array_equal(selected[0]['raw_bbox_xyxy'], expected_raw)


def test_empty_detection_keeps_timestamps(tmp_path):
    frames = make_frames(tmp_path, [[], []])
    (tmp_path / 'timestamps.txt').write_text('0 1.000000001\n1 1.033333334\n')
    instances = SimpleNamespace(
        pred_classes=torch.zeros(0, dtype=torch.int64), scores=torch.zeros(0),
        pred_boxes=SimpleNamespace(tensor=torch.zeros((0, 4))),
    )
    pose = Mock()
    result = adapter.extract_observations(
        [Path(frame['img_path']) for frame in frames], Mock(return_value={'instances': instances}), pose,
    )
    assert len(result['frames']) == 2
    assert all(not frame['observations'] for frame in result['frames'])
    assert result['frames'][0]['timestamp_ns'] == 1_000_000_001
    assert result['frames'][1]['timestamp_ns'] == 1_033_333_334
    pose.predict_pose.assert_not_called()


def test_track_side_gap_and_fragment_boundaries_preserve_observations(tmp_path):
    frames = make_frames(tmp_path, [
        [observation(), observation(1, candidate=1)],
        [observation(), observation(1, candidate=1)],
        [observation(side='right')], [],
        [observation(side='right')],
        [observation(side='right', fragment=1)],
    ])
    original = deepcopy(frames)
    inputs = _collect_inputs(frames, SimpleNamespace(img_focal=550), 'hawor')
    assert len(inputs.instances) == 7
    assert [[inst.frame_idx for inst in segment] for segment in inputs.temporal_segments] == [
        [0, 1], [0, 1], [2], [4], [5],
    ]
    for inst in inputs.instances:
        source = next(item for item in frames[inst.frame_idx]['hands']
                      if item['physical_track_id'] == inst.observation_meta['physical_track_id'])
        np.testing.assert_array_equal(inst.bbox, source['bbox_xyxy'])
        np.testing.assert_allclose(inst.keypoints, source['keypoints_2d'])
        assert inst.hand_side == source['handedness']
    assert frames == original


def test_backend_side_stabilizes_transient_frontend_flip(tmp_path):
    frames = make_frames(tmp_path, [
        [observation(side='right')],
        [dict(observation(side='left'), backend_handedness='right')],
        [observation(side='right')],
    ])
    inputs = _collect_inputs(frames, SimpleNamespace(img_focal=550), 'hawor')
    assert [[inst.frame_idx for inst in segment] for segment in inputs.temporal_segments] == [[0, 1, 2]]
    assert [inst.hand_side for inst in inputs.temporal_segments[0]] == ['right'] * 3
    assert [inst.observation_meta['handedness'] for inst in inputs.temporal_segments[0]] == ['right', 'left', 'right']


def test_hawor_inference_keeps_two_same_side_hands_and_physical_ids(tmp_path):
    frames = make_frames(tmp_path, [[observation(), observation(1, candidate=1)],
                                    [observation(side='right')], []])
    inputs = _collect_inputs(frames, SimpleNamespace(img_focal=550), 'hawor')
    model = Mock()

    def infer(paths, boxes, **kwargs):
        n = len(paths)
        return {'pred_rotmat': torch.eye(3).repeat(n, 16, 1, 1),
                'pred_shape': torch.zeros(n, 10), 'pred_trans': torch.ones(n, 1, 3)}

    model.inference.side_effect = infer
    model.mano.query.return_value = SimpleNamespace(vertices=torch.zeros(1, 778, 3))
    runner = HaworBackendRunner(model, None, None, torch.device('cpu'))
    outputs = runner.infer(inputs)
    assert [call.kwargs['do_flip'] for call in model.inference.call_args_list] == [True, True, False]
    results = _assemble_results(outputs, frames)
    first = results[frames[0]['img_path']]
    assert first['tracked_ids'] == [0, 1]
    assert [item['is_right'] for item in first['mano']] == [0, 0]
    assert [meta['candidate_id'] for meta in first['backend_meta']] == [0, 1]
    assert first['track_id_semantics'] == 'anonymous_physical_slot'
    assert not results[frames[2]['img_path']]['mano']
    assert results[frames[1]['img_path']]['mano'][0]['is_right'] == 1


def test_wilor_adapter_keeps_selection_metadata_and_crop_inputs(tmp_path, monkeypatch):
    frames = make_frames(tmp_path, [[observation(), observation(1, candidate=1)]])
    inputs = _collect_inputs(frames, SimpleNamespace(img_focal=None), 'wilor')
    entries_seen = []

    def dataset(cfg, entries, **kwargs):
        entries_seen.extend(entries)
        return [dict(img=torch.zeros(3, 8, 8), right=e['right'],
                     box_center=e['center'], box_size=np.float32(e['scale'].max() * 200),
                     img_size=np.array([160, 100], dtype=np.float32), instance_idx=e['instance_idx'])
                for e in entries]

    monkeypatch.setattr('hmr_backends.runners.batch_runner.SequenceVitDetDataset', dataset)
    model = Mock(return_value={
        'pred_cam': torch.ones(2, 3), 'pred_keypoints_2d': torch.zeros(2, 21, 2),
        'pred_vertices': torch.zeros(2, 778, 3),
        'pred_mano_params': {'global_orient': torch.eye(3).repeat(2, 1, 1, 1),
                             'hand_pose': torch.eye(3).repeat(2, 15, 1, 1), 'betas': torch.zeros(2, 10)},
    })
    cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=256))
    args = SimpleNamespace(rescale_factor=2.0, batch_size=2)
    outputs = BatchBackendRunner(model, cfg, args, torch.device('cpu'), 'wilor').infer(inputs)
    assert [entry['right'] for entry in entries_seen] == [0.0, 0.0]
    for entry, inst in zip(entries_seen, inputs.instances):
        np.testing.assert_array_equal(entry['center'], (inst.bbox[:2] + inst.bbox[2:]) / 2)
        np.testing.assert_array_equal(entry['scale'], 2 * (inst.bbox[2:] - inst.bbox[:2]) / 200)
    assert [out.raw_backend_meta['physical_track_id'] for out in outputs] == [0, 1]


def test_all_missing_sequence_still_renders_and_saves_empty_entries(tmp_path, monkeypatch):
    frames = make_frames(tmp_path, [[], []])
    render = Mock()
    build = Mock()
    monkeypatch.setattr('mesh_recovery._render_results', render)
    monkeypatch.setattr('mesh_recovery.build_runner', build)
    result = run_mesh_recovery(frames, SimpleNamespace(backend_name='hawor'), Mock(),
                               SimpleNamespace(img_focal=550), tmp_path, device=torch.device('cpu'))
    assert len(result) == 2
    assert all(not frame['mano'] for frame in result.values())
    render.assert_called_once()
    build.assert_not_called()


def test_selected_cache_reused_and_signature_change_invalidates(tmp_path, monkeypatch):
    frames = make_frames(tmp_path, [[observation()]])
    paths = [Path(frames[0]['img_path'])]
    raw = {'frames': [{'frame_idx': 0, 'img_path': str(paths[0]), 'observations': []}], 'image_size': (160, 100)}
    detect = Mock(return_value=raw)
    monkeypatch.setattr(adapter, '_detect_with_models', detect)
    signature = Mock(return_value='one')
    monkeypatch.setattr(adapter, 'cache_signature', signature)
    for _ in range(2):
        adapter.run_observation_frontend(paths, tmp_path / 'out', tmp_path, torch.device('cpu'))
    assert detect.call_count == 1
    signature.return_value = 'two'
    adapter.run_observation_frontend(paths, tmp_path / 'out', tmp_path, torch.device('cpu'))
    assert detect.call_count == 2
    adapter.run_observation_frontend(paths, tmp_path / 'out', tmp_path, torch.device('cpu'), force=True)
    assert detect.call_count == 3


def test_overlay_default_and_explicit_omega():
    from run import build_arg_parser
    parser = build_arg_parser()
    defaults = parser.parse_args(['--input', 'unused'])
    assert not defaults.omega_world
    assert not defaults.mint_depth_wrist_only
    assert not defaults.mint_3d_consistency_gate
    assert not defaults.hand_tracking_parquet
    assert not parser.parse_args(['--input', 'unused', '--frontend', 'observations']).omega_world
    assert parser.parse_args(['--input', 'unused', '--omega_world']).omega_world
    assert not parser.parse_args(['--input', 'unused', '--no_omega_world']).omega_world
    with pytest.raises(SystemExit):
        parser.parse_args(['--input', 'unused', '--omega_world', '--no_omega_world'])


def test_cli_observations_bypasses_legacy_and_omega(tmp_path, monkeypatch):
    import run

    frames = make_frames(tmp_path, [[], []])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['run.py', '--input', str(tmp_path), '--frontend', 'observations'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    select = Mock(return_value=frames)
    monkeypatch.setattr(adapter, 'run_observation_frontend', select)
    forbidden = Mock(side_effect=AssertionError('Legacy/omega path must not execute'))
    monkeypatch.setattr(run, 'YOLO', forbidden)
    monkeypatch.setattr(run, 'load_or_build_cleaned_bboxes', forbidden)
    monkeypatch.setattr('hmr_backends.omega.runner.run_omega_camera_recovery', forbidden)
    bundle = Mock()
    monkeypatch.setattr(run, 'load_backend', Mock(return_value=bundle))
    monkeypatch.setattr(run, 'Renderer', Mock())
    reconstruct = Mock(return_value=_assemble_results([], frames))
    monkeypatch.setattr(run, 'run_mesh_recovery', reconstruct)
    run.main()
    select.assert_called_once()
    reconstruct.assert_called_once()
    forbidden.assert_not_called()
    out = tmp_path / 'test_data/hand_proc' / f'{tmp_path.name}_observations'
    assert (out / f'{tmp_path.name}_hawor.pkl').exists()
    assert not (out / 'pass1_raw.pkl').exists()
    assert not (out / 'omega_camera.npz').exists()


def test_cache_signature_tracks_input_paths_assets_and_fps(tmp_path):
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'image')
    for name in [
        '_DATA/detectron2/model_final_f05665.pkl',
        'hmr_backends/configs/cascade_mask_rcnn_vitdet_h_75ep.py',
        '_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth', 'vitpose_model.py',
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    first = adapter.cache_signature([image], tmp_path, 30)
    assert first == adapter.cache_signature([image], tmp_path, 30)
    assert first != adapter.cache_signature([image], tmp_path, 15)
    alias = tmp_path / 'alias.jpg'
    alias.symlink_to(image)
    assert first != adapter.cache_signature([alias], tmp_path, 30)
    (tmp_path / 'vitpose_model.py').write_text('changed fixture')
    assert first != adapter.cache_signature([image], tmp_path, 30)


def test_duplicate_track_slot_rejected(tmp_path):
    frames = make_frames(tmp_path, [[observation(), observation(candidate=1)]])
    with pytest.raises(ValueError, match='Duplicate physical track'):
        _collect_inputs(frames, SimpleNamespace(img_focal=550), 'hawor')
