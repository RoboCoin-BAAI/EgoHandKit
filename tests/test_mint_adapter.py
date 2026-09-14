from pathlib import Path

import cv2
import numpy as np
import pytest

from observation_frontend.mint_adapter import (
    build_mint_observations,
    compare_mint_yolo_observations,
    load_mint_predictions,
    save_mint_observation_cache,
    validate_mint_observations,
    write_synthetic_mint_fixture,
)


def _images(tmp_path, count=2, hw=(100, 160)):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index:06d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((*hw, 3), np.uint8))
        paths.append(path)
    return paths


def _cache(tmp_path, *, points=None, presence=None, orig_hw=(100, 160), mint_hw=(50, 80)):
    n = len(points) if points is not None else 1
    points = np.asarray(points if points is not None else np.zeros((n, 21, 3)), np.float32)
    if points.ndim == 2:
        points = points[None]
    if not np.any(points[..., 2]):
        points[..., 2] = 2
    presence = np.asarray(presence if presence is not None else [[1, 0]] * n, np.float32)
    path = tmp_path / "mint.npz"
    np.savez(path, schema_version=np.array("mint_prediction_cache.v1"), frame_idx=np.arange(n),
             orig_hw=np.tile(orig_hw, (n, 1)), mint_input_hw=np.tile(mint_hw, (n, 1)),
             camera_intrinsics=np.tile([[[50, 0, mint_hw[1] / 2], [0, 50, mint_hw[0] / 2], [0, 0, 1]]], (n, 1, 1)),
             left_joints_cam=points, right_joints_cam=points, hand_presence=presence,
             metadata_json=np.array('{"intrinsics_space":"mint_input"}'))
    return path


def test_resize_only_projection_and_bbox(tmp_path):
    cache = load_mint_predictions(_cache(tmp_path))
    frames = build_mint_observations(cache, _images(tmp_path, 1))
    selected = frames[0]["selected_for_hamer"]
    assert len(selected) == 1
    assert selected[0]["observation_meta"]["confidence_type"] == "mint_presence_projected"
    assert np.asarray(selected[0]["vitpose_keypoints_2d"]).shape == (21, 3)
    assert selected[0]["bbox_xyxy"][0] >= 0


def test_affine_crop_projection(tmp_path):
    points = np.zeros((1, 21, 3), np.float32)
    points[..., 0] = 0
    points[..., 1] = 0
    points[..., 2] = 2
    path = _cache(tmp_path, points=points)
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    values["orig_to_mint"] = np.array([[2, 0, -20], [0, 2, -10]], np.float32)
    np.savez(path, **values)
    frames = build_mint_observations(load_mint_predictions(path), _images(tmp_path, 1))
    assert frames[0]["selected_for_hamer"][0]["bbox_xyxy"][0] > 0


def test_single_double_and_missing_hands(tmp_path):
    points = np.zeros((3, 21, 3), np.float32); points[..., 2] = 2
    cache = load_mint_predictions(_cache(tmp_path, points=points, presence=[[1, 0], [1, 1], [0, 0]]))
    frames = build_mint_observations(cache, _images(tmp_path, 3))
    assert [len(frame["selected_for_hamer"]) for frame in frames] == [1, 2, 0]
    assert frames[1]["selected_for_hamer"][1]["physical_track_id"] == 1


def test_invalid_depth_and_out_of_bounds_are_rejected(tmp_path):
    points = np.zeros((1, 21, 3), np.float32)
    points[..., 2] = -1
    cache = load_mint_predictions(_cache(tmp_path, points=points))
    assert build_mint_observations(cache, _images(tmp_path, 1))[0]["selected_for_hamer"] == []


def test_observation_schema_and_cache_mismatch(tmp_path):
    path = _cache(tmp_path)
    images = _images(tmp_path, 1)
    cache = load_mint_predictions(path)
    frames = build_mint_observations(cache, images)
    validate_mint_observations(frames, image_paths=images)
    output = tmp_path / "obs.pkl"
    config = {"bbox_scale": 1.2, "presence_threshold": 0.5}
    save_mint_observation_cache(output, frames, image_paths=images, mint_path=path, config=config)
    images[0].write_bytes(images[0].read_bytes() + b"x")
    from observation_frontend.mint_adapter import load_mint_observation_cache
    with pytest.raises(ValueError, match="does not match"):
        load_mint_observation_cache(output, image_paths=images, mint_path=path, config=config)


def test_synthetic_fixture_round_trip(tmp_path):
    path = write_synthetic_mint_fixture(tmp_path / "fixture.npz")
    assert load_mint_predictions(path)["schema_version"] == "mint_prediction_cache.v1"


def test_combined_raw_mint_hand_is_split_at_cache_boundary(tmp_path):
    path = _cache(tmp_path)
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files if key not in {"left_joints_cam", "right_joints_cam", "left_hand", "right_hand"}}
    values["hand"] = np.zeros((1, 218), np.float32)
    np.savez(path, **values)
    loaded = load_mint_predictions(path)
    assert loaded["hands"]["left"].shape == (1, 109)
    assert loaded["hands"]["right"].shape == (1, 109)


def test_yolo_check_is_diagnostic_only():
    mint = [{"frame_idx": 0, "selected_for_hamer": [{"handedness": "left", "bbox_xyxy": [0, 0, 10, 10]}]}]
    yolo = [{"frame_idx": 0, "left_bbox": [1, 1, 11, 11], "left_conf": 0.8}]
    report = compare_mint_yolo_observations(mint, yolo, iou_threshold=0.1)
    assert report["counts"]["match"] == 1
    assert report["frames"][0]["sides"]["left"]["status"] == "match"
    assert report["frames"][0]["sides"]["right"]["status"] == "both_missing"
