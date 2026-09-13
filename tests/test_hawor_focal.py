from types import SimpleNamespace
from unittest.mock import Mock

import mesh_recovery
from mesh_recovery import _resolve_hawor_focal


def test_focal_override_is_stored_per_sequence(tmp_path):
    sequence_dir = tmp_path / 'images' / 'session_left'
    sequence_dir.mkdir(parents=True)
    focal = _resolve_hawor_focal(SimpleNamespace(img_focal=None), sequence_dir)
    assert focal == 600.0
    assert (sequence_dir / 'est_focal.txt').read_text() == '600.0'
    assert not (sequence_dir.parent / 'est_focal.txt').exists()


def test_cli_focal_takes_priority_over_sequence_file(tmp_path):
    sequence_dir = tmp_path / 'images' / 'session_left'
    sequence_dir.mkdir(parents=True)
    (sequence_dir / 'est_focal.txt').write_text('600')
    focal = _resolve_hawor_focal(SimpleNamespace(img_focal=553.9148), sequence_dir)
    assert focal == 553.9148


def test_hawor_renderer_uses_resolved_sequence_focal(tmp_path, monkeypatch):
    inputs = SimpleNamespace(instances=[object()], img_focal=553.9148)
    monkeypatch.setattr(mesh_recovery, '_collect_inputs', Mock(return_value=inputs))
    runner = Mock()
    runner.infer.return_value = []
    monkeypatch.setattr(mesh_recovery, 'build_runner', Mock(return_value=runner))
    monkeypatch.setattr(mesh_recovery, 'build_left_hand_policy', Mock(return_value=Mock()))
    monkeypatch.setattr(mesh_recovery, '_render_results', Mock())
    renderer = SimpleNamespace(focal_length=600.0)

    mesh_recovery.run_mesh_recovery(
        [], SimpleNamespace(backend_name='hawor'), renderer, SimpleNamespace(),
        tmp_path, device=object(),
    )

    assert renderer.focal_length == 553.9148
