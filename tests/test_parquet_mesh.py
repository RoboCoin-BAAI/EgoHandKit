from pathlib import Path
from unittest.mock import Mock
import importlib.util

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARQUET_MESH = _load_module("parquet_mesh", "hmr_backends/utils/parquet_mesh.py")

blend_rgba_overlay = _PARQUET_MESH.blend_rgba_overlay
draw_openpose21_joints = _PARQUET_MESH.draw_openpose21_joints
fit_mano_joints = _PARQUET_MESH.fit_mano_joints
project_camera_joints = _PARQUET_MESH.project_camera_joints
render_tracking_parquet = _PARQUET_MESH.render_tracking_parquet


def _write_tracking_parquet(path, rows):
    table = pa.Table.from_pylist(rows).replace_schema_metadata({
        b"coordinate_system": b"opencv_x_right_y_down_z_forward",
        b"joint_order": b"openpose21",
    })
    pq.write_table(table, path)
    return path


def _tracking_row(frame_idx, *, left=None, right=None):
    missing = np.full((21, 3), np.nan, dtype=np.float32).tolist()
    return {
        "frame_idx": frame_idx,
        "timestamp_ns": None,
        "source": "hmr" if left is not None or right is not None else "none",
        "left_present": left is not None,
        "right_present": right is not None,
        "left_joints_3d": left.tolist() if left is not None else missing,
        "right_joints_3d": right.tolist() if right is not None else missing,
        "left_confidence": None,
        "right_confidence": None,
        "left_source": "hmr" if left is not None else "none",
        "right_source": "hmr" if right is not None else "none",
    }


def test_parquet_render_cli_independent_and_enables_export():
    pytest.importorskip("pyrender")
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
    batch_rodrigues = pytest.importorskip("smplx.lbs").batch_rodrigues
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


def test_project_camera_joints_uses_open_cv_camera_intrinsics():
    joints = np.zeros((21, 3), np.float32)
    joints[:, 2] = 2.0
    joints[0] = [0.0, 0.0, 2.0]
    joints[9] = [0.2, -0.1, 2.0]
    k = np.array([[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])

    keypoints = project_camera_joints(joints, k)

    np.testing.assert_allclose(keypoints[0], [10.0, 20.0, 1.0])
    np.testing.assert_allclose(keypoints[9], [20.0, 10.0, 1.0])


def test_blend_rgba_overlay_respects_mesh_alpha_scale():
    image = np.zeros((2, 2, 3), np.uint8)
    rgba = np.zeros((2, 2, 4), np.float32)
    rgba[:, :, 0] = 1.0
    rgba[:, :, 3] = 1.0

    blended = blend_rgba_overlay(image, rgba, mesh_alpha=0.25)

    np.testing.assert_array_equal(blended[:, :, 2], np.full((2, 2), 64, np.uint8))


def test_draw_openpose21_joints_marks_projected_joints():
    image = np.zeros((60, 80, 3), np.uint8)
    keypoints = np.zeros((21, 3), np.float32)
    keypoints[:, 2] = 1.0
    keypoints[0, :2] = [20, 30]
    keypoints[9, :2] = [40, 30]

    drawn = draw_openpose21_joints(image, keypoints, side='right')

    assert drawn[30, 20].sum() > 0
    assert drawn[30, 40].sum() > 0
    assert drawn[30, 30].sum() > 0


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
    row = _tracking_row(0, left=joints if side == 'left' else None,
                        right=joints if side == 'right' else None)
    parquet = _write_tracking_parquet(tmp_path / 'tracking.parquet', [
        row,
        _tracking_row(1),
    ])
    original = parquet.read_bytes()
    k = np.array([[100, 0, 12], [0, 110, 14], [0, 0, 1]])
    monkeypatch.setattr(_PARQUET_MESH, 'resolve_depth_camera_intrinsics',
                        lambda *a, **kw: (k, None))
    monkeypatch.setattr(_PARQUET_MESH, 'resolve_depth_frames',
                        lambda *a: [f['img_path'] for f in frames])
    fitter = Mock(return_value=(np.ones((1, 778, 3)) * 0.2, joints[None], np.array([error])))
    monkeypatch.setattr(_PARQUET_MESH, 'fit_mano_joints', fitter)
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


def test_render_tracking_parquet_draws_semitransparent_mesh_and_final_joints(tmp_path, monkeypatch):
    image_path = tmp_path / '0.jpg'
    cv2.imwrite(str(image_path), np.zeros((40, 60, 3), np.uint8))
    frames = [{'frame_idx': 0, 'img_path': str(image_path), 'hands': []}]
    joints = np.zeros((21, 3), np.float32)
    joints[:, 2] = 1.0
    joints[0] = [0.0, 0.0, 1.0]
    joints[9] = [0.1, 0.0, 1.0]
    parquet = _write_tracking_parquet(tmp_path / 'tracking.parquet', [
        _tracking_row(0, right=joints),
    ])
    k = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]])
    monkeypatch.setattr(_PARQUET_MESH, 'resolve_depth_camera_intrinsics',
                        lambda *a, **kw: (k, None))
    monkeypatch.setattr(_PARQUET_MESH, 'resolve_depth_frames',
                        lambda *a: [str(image_path)])
    monkeypatch.setattr(_PARQUET_MESH, 'fit_mano_joints',
                        Mock(return_value=(np.ones((1, 778, 3), np.float32),
                                           joints[None], np.array([0.001]))))
    monkeypatch.setattr('bbox_utils.create_video_from_images', Mock())
    rgba = np.zeros((40, 60, 4), np.float32)
    rgba[:, :, 0] = 1.0
    rgba[:, :, 3] = 1.0
    renderer = Mock()
    renderer.render_rgba_multiple.return_value = (rgba, None)

    render_tracking_parquet(
        parquet, frames, None, renderer, tmp_path / 'final', depth_dir=tmp_path,
        mesh_alpha=0.25,
    )

    rendered = cv2.imread(str(tmp_path / 'final/parquet_render_frames/000000.jpg'))
    assert 40 <= int(rendered[5, 5, 2]) <= 80
    assert int(rendered[20, 20].sum()) > int(rendered[5, 5].sum())
