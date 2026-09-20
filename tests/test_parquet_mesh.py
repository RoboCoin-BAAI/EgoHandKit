from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from smplx.lbs import batch_rodrigues

from hmr_backends.utils.parquet_mesh import fit_mano_joints, render_tracking_parquet
from hmr_backends.utils.hand_tracking_parquet import export_hand_tracking_parquet


def test_parquet_render_cli_independent_and_enables_export():
    from run import build_arg_parser, validate_cli_args
    parser = build_arg_parser()
    defaults = parser.parse_args(['--input', 'unused'])
    assert not defaults.render_hand_tracking_parquet
    args = parser.parse_args(['--input', 'unused', '--depth_dir', 'depth',
                              '--render_hand_tracking_parquet'])
    validate_cli_args(parser, args)
    assert args.hand_tracking_parquet
    assert not args.mint_3d_consistency_gate
    args.render_frontend_fallback = True
    with pytest.raises(SystemExit):
        validate_cli_args(parser, args)


@pytest.fixture
def limited_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_real_mano_fit_mirrored_hands_preserves_wrist_and_model(limited_torch_threads):
    asset = Path(__file__).resolve().parents[1] / '_DATA/data/mano/MANO_RIGHT.pkl'
    if not asset.exists():
        pytest.skip('Local MANO assets unavailable')
    from hmr_backends.models.mano_wrapper import MANO
    mano = MANO(model_path=str(asset.parent), use_pca=False)
    pose = torch.zeros((1, 16, 3))
    pose[:, 0] = torch.tensor([0.4, -0.3, 1.0])
    pose[:, 1:, 2] = 0.3
    with torch.no_grad():
        out = mano.query({'pred_rotmat': batch_rodrigues(pose.reshape(-1, 3)).reshape(1, 16, 3, 3),
                          'pred_shape': torch.zeros((1, 10))})
    right = (out.joints[0, :21] - out.joints[0, :1]).numpy() * 1.2
    left = right * [-1, 1, 1]
    targets = np.stack([right, left]) + [0.1, -0.1, 0.6]
    flags = [p.requires_grad for p in mano.parameters()]
    vertices, fitted, errors = fit_mano_joints(mano, targets, ['right', 'left'], steps=150)
    assert vertices.shape == (2, 778, 3)
    assert np.max(errors) < 0.004
    np.testing.assert_allclose(fitted[:, 0], targets[:, 0], atol=1e-6)
    assert [p.requires_grad for p in mano.parameters()] == flags
    assert all(p.grad is None for p in mano.parameters())


@pytest.mark.parametrize('error,rendered', [(0.001, 1), (0.04, 0), (float('nan'), 0)])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_render_reads_only_parquet_presence_and_calibrated_camera(tmp_path, monkeypatch, error, rendered, side):
    frames = []
    for i in range(2):
        path = tmp_path / f'{i}.jpg'
        cv2.imwrite(str(path), np.zeros((40, 60, 3), np.uint8))
        # Deliberately contradictory frontend data must not influence rendering.
        frames.append({'frame_idx': i, 'img_path': str(path), 'hands': []})
    joints = np.ones((21, 3), np.float32) * 0.2
    results = {frames[0]['img_path']: {'joints_3d': [joints],
                'backend_meta': [{'handedness': side, 'backend': 'mint'}]}}
    parquet = export_hand_tracking_parquet(frames, results, tmp_path / 'tracking.parquet')
    original = parquet.read_bytes()
    k = np.array([[100, 0, 12], [0, 110, 14], [0, 0, 1]])
    monkeypatch.setattr('hmr_backends.utils.parquet_mesh.resolve_depth_camera_intrinsics',
                        lambda *a, **kw: (k, None))
    monkeypatch.setattr('hmr_backends.utils.parquet_mesh.resolve_depth_frames',
                        lambda *a: [f['img_path'] for f in frames])
    fitter = Mock(return_value=(np.ones((1, 778, 3)) * 0.2, joints[None], np.array([error])))
    monkeypatch.setattr('hmr_backends.utils.parquet_mesh.fit_mano_joints', fitter)
    monkeypatch.setattr('bbox_utils.create_video_from_images', Mock())
    renderer = Mock()
    renderer.render_rgba_multiple.return_value = (np.zeros((40, 60, 4)), None)
    report = render_tracking_parquet(parquet, frames, None, renderer, tmp_path / 'final', depth_dir=tmp_path)
    assert report['present_hands'] == 1
    assert report['rendered_hands'] == rendered
    assert fitter.call_args.args[2] == [side]
    np.testing.assert_array_equal(fitter.call_args.args[1][0], joints)
    if rendered:
        np.testing.assert_array_equal(renderer.render_rgba_multiple.call_args.kwargs['camera_intrinsics'], k)
        color = (0.2, 0.8, 0.3) if side == 'left' else (0.2, 0.4, 0.9)
        assert renderer.render_rgba_multiple.call_args.kwargs['mesh_base_color'] == [color]
    else:
        renderer.render_rgba_multiple.assert_not_called()
        assert report['hands'][0]['reason'] in ('nonfinite_fit', 'fit_error_exceeds_limit')
    assert len(list((tmp_path / 'final/parquet_render_frames').glob('*.jpg'))) == 2
    assert parquet.read_bytes() == original
