from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from mesh_recovery import _render_results


class MeshRenderTimingTests(unittest.TestCase):
    def test_frames_without_hands_preserve_video_timing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = [root / f'{index:06d}.jpg' for index in range(3)]
            for path in image_paths:
                self.assertTrue(cv2.imwrite(str(path), np.full((32, 48, 3), 80, np.uint8)))
            cleaned = [{'img_path': str(path)} for path in image_paths]
            output = SimpleNamespace(
                img_path=str(image_paths[1]), hand_side='right',
                pred_vertices=np.zeros((778, 3)), mano_params={}, cam_trans=np.zeros(3),
            )
            renderer = Mock(focal_length=600)
            renderer.render_rgba_multiple.return_value = (np.zeros((32, 48, 4)), None)
            policy = Mock()
            policy.apply.return_value = (output.pred_vertices, 1)
            args = SimpleNamespace(backend='hawor', render=True)
            bundle = SimpleNamespace(backend_name='hawor')
            with patch('mesh_recovery.create_video_from_images') as create_video:
                _render_results([output], cleaned, renderer, args, root, 30, policy, bundle)
            self.assertEqual(
                sorted(path.name for path in (root / 'final' / 'render_frames').glob('*.jpg')),
                [path.name for path in image_paths],
            )
            for index in (0, 2):
                np.testing.assert_array_equal(
                    cv2.imread(str(root / 'final' / 'render_frames' / image_paths[index].name)),
                    cv2.imread(str(image_paths[index])),
                )
            renderer.render_rgba_multiple.assert_called_once()
            create_video.assert_called_once_with(
                str(root / 'final' / 'render_frames'), str(root / 'final' / 'render.mp4'), fps=30,
            )

    def test_disabled_render_does_not_write_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(backend='hawor', render=False)
            with patch('mesh_recovery.create_video_from_images') as create_video:
                _render_results([], [], Mock(), args, root, 30, Mock(), Mock())
            self.assertFalse((root / 'final' / 'render_frames').exists())
            create_video.assert_not_called()


if __name__ == '__main__':
    unittest.main()
