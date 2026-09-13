from pathlib import Path

import pytest

from input_naming import sequence_name_for_input


def test_video_name_in_dataset_videos_dir_includes_session_name():
    path = Path('/home/user/ego_data/session_20260825_170413_task/videos/left.mp4')
    assert sequence_name_for_input(path) == 'session_20260825_170413_task_left'


def test_non_video_folder_keeps_folder_name():
    assert sequence_name_for_input(Path('/tmp/frames')) == 'frames'


def test_explicit_name_overrides_derived_name():
    path = Path('/tmp/session/videos/left.mp4')
    assert sequence_name_for_input(path, 'cart_left') == 'cart_left'


@pytest.mark.parametrize('name', ['', '.', '..', 'nested/name'])
def test_explicit_name_must_be_single_path_component(name):
    with pytest.raises(ValueError):
        sequence_name_for_input(Path('/tmp/session/videos/left.mp4'), name)
