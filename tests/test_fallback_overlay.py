from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hmr_backends.utils.fallback_overlay import draw_frontend_fallback
from mesh_recovery import _render_results


def hand(side='left'):
    points = np.column_stack((np.linspace(30, 90, 21),
                              np.linspace(45, 90, 21), np.ones(21)))
    return {'handedness': side, 'keypoints_2d': points, 'source': 'mint',
            'meta': {'joints_3d_camera': np.ones((21, 3)),
                     'camera_frame': 'opencv_x_right_y_down_z_forward',
                     'joint_order': 'openpose21'}}


def test_outside_wrist_still_draws_visible_fingers():
    observation = hand()
    observation['keypoints_2d'][0, :2] = (-40, 50)
    image = np.zeros((120, 160, 3), np.uint8)
    draw_frontend_fallback(image, {'hands': [observation]}, {'left'})
    assert image[45:100].any()


def test_invalid_keypoints_do_not_draw():
    observation = hand()
    observation['keypoints_2d'][:] = np.nan
    image = np.zeros((120, 160, 3), np.uint8)
    draw_frontend_fallback(image, {'hands': [observation]}, {'left'})
    assert not image.any()


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('hmr_side', [None, 'left', 'right'])
def test_render_fallback_missing_and_mixed_hands(tmp_path, monkeypatch, enabled, hmr_side):
    path = tmp_path / 'input.jpg'
    cv2.imwrite(str(path), np.zeros((120, 160, 3), np.uint8))
    frames = [{'frame_idx': 0, 'img_path': str(path), 'hands': [hand()]}]
    args = SimpleNamespace(render=True, render_frontend_fallback=enabled, depth_dir=None)
    renderer = Mock()
    renderer.render_rgba_multiple.return_value = (np.zeros((120, 160, 4)), None)
    policy = Mock()
    policy.apply.return_value = (np.zeros((778, 3)), True)
    outputs = [] if hmr_side is None else [SimpleNamespace(
        img_path=str(path), hand_side=hmr_side, pred_vertices=None,
        mano_params=None, cam_trans=None)]
    video = Mock()
    monkeypatch.setattr('mesh_recovery.create_video_from_images', video)
    _render_results(outputs, frames, renderer, args, tmp_path, 30, policy,
                    SimpleNamespace(backend_name='hawor'))
    rendered = cv2.imread(str(tmp_path / 'final/render_frames/input.jpg'))
    assert bool(rendered.any()) == (enabled and hmr_side != 'left')
    video.assert_called_once()


def test_missing_mint_geometry_is_not_rendered(tmp_path, monkeypatch):
    path = tmp_path / 'input.jpg'
    cv2.imwrite(str(path), np.zeros((120, 160, 3), np.uint8))
    observation = hand()
    observation['meta'] = {}
    monkeypatch.setattr('mesh_recovery.create_video_from_images', Mock())
    _render_results([], [{'frame_idx': 0, 'img_path': str(path), 'hands': [observation]}],
                    None, SimpleNamespace(render=True, render_frontend_fallback=True),
                    tmp_path, 30, None, None)
    assert not cv2.imread(str(tmp_path / 'final/render_frames/input.jpg')).any()
