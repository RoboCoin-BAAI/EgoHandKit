import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from observation_frontend.schema import (
    OBSERVATION_SCHEMA,
    load_observation_sequence,
    remap_observation_sequence,
    save_observation_sequence,
    validate_observation_sequence,
)


def _images(root, count=2, shape=(48, 64)):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"{index:06d}.jpg"
        assert cv2.imwrite(str(path), np.full((*shape, 3), index, np.uint8))
        paths.append(path)
    return paths


def _sequence(paths, *, frontend="mint", fps=15.0):
    image = cv2.imread(str(paths[0]))
    height, width = image.shape[:2]
    frames = []
    for index, path in enumerate(paths):
        hands = [] if index else [{
            "observation_id": "left-0",
            "handedness": "left",
            "backend_handedness": "left",
            "bbox_xyxy": [4.0, 5.0, 30.0, 35.0],
            "keypoints_2d": np.tile([12.0, 15.0, 0.9], (21, 1)).astype(np.float32),
            "physical_track_id": 7,
            "physical_track_fragment_id": 2,
            "confidence": 0.9,
            "source": frontend,
            "meta": {"producer_detail": "preserved"},
        }]
        frames.append({
            "frame_idx": index,
            "img_path": str(path),
            "timestamp_ns": None,
            "hands": hands,
        })
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": {
            "sequence_name": paths[0].parent.name,
            "frame_count": len(paths),
            "fps": fps,
            "image_width": width,
            "image_height": height,
        },
        "frontend": {"name": frontend, "version": "test"},
        "frames": frames,
    }


def test_canonical_observation_round_trip_preserves_arrays_and_empty_frames(tmp_path):
    sequence = _sequence(_images(tmp_path / "old"))
    path = tmp_path / "observations.pkl"

    save_observation_sequence(sequence, path)
    loaded = load_observation_sequence(path)

    assert loaded["schema_version"] == OBSERVATION_SCHEMA
    np.testing.assert_array_equal(
        loaded["frames"][0]["hands"][0]["keypoints_2d"],
        sequence["frames"][0]["hands"][0]["keypoints_2d"],
    )
    assert loaded["frames"][1]["hands"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="wrong"), "egohand.observations.v1"),
        (lambda value: value["frames"][0]["hands"][0].update(bbox_xyxy=[2, 2, 1, 3]), "positive area"),
        (lambda value: value["frames"][0]["hands"][0].update(keypoints_2d=np.zeros((20, 3))), r"shape \[21,3\]"),
        (lambda value: value["frames"][0]["hands"].append(deepcopy(value["frames"][0]["hands"][0])), "duplicate"),
        (lambda value: value["frames"].reverse(), "consecutive, and ordered"),
        (lambda value: value["frames"].pop(), "frame_count"),
    ],
)
def test_invalid_canonical_observations_fail_clearly(tmp_path, mutation, message):
    sequence = _sequence(_images(tmp_path / "old"))
    mutation(sequence)
    with pytest.raises(ValueError, match=message):
        validate_observation_sequence(sequence)


def test_load_rejects_wrong_schema_version(tmp_path):
    sequence = _sequence(_images(tmp_path / "old"))
    sequence["schema_version"] = "egohand.observations.v0"
    path = tmp_path / "observations.pkl"
    path.write_bytes(__import__("pickle").dumps(sequence))
    with pytest.raises(ValueError, match="egohand.observations.v1"):
        load_observation_sequence(path)


def test_remap_uses_frame_idx_and_preserves_observation_values(tmp_path):
    old_paths = _images(tmp_path / "old")
    current_paths = _images(tmp_path / "current")
    sequence = _sequence(old_paths)
    for frame in sequence["frames"]:
        frame["img_path"] = f"/previous/run/frames/{frame['frame_idx']:06d}.jpg"
    original_hand = deepcopy(sequence["frames"][0]["hands"][0])

    remapped = remap_observation_sequence(
        sequence, current_paths, fps=15.0, sequence_name="current"
    )

    assert [frame["img_path"] for frame in remapped["frames"]] == [str(path) for path in current_paths]
    np.testing.assert_array_equal(remapped["frames"][0]["hands"][0]["bbox_xyxy"], original_hand["bbox_xyxy"])
    np.testing.assert_array_equal(remapped["frames"][0]["hands"][0]["keypoints_2d"], original_hand["keypoints_2d"])
    assert remapped["frames"][0]["hands"][0]["physical_track_id"] == 7
    assert remapped["frames"][0]["hands"][0]["physical_track_fragment_id"] == 2
    assert sequence["frames"][0]["img_path"] == "/previous/run/frames/000000.jpg"


def test_remap_rejects_frame_count_and_image_size_mismatches(tmp_path):
    sequence = _sequence(_images(tmp_path / "old"))
    with pytest.raises(ValueError, match="frame count"):
        remap_observation_sequence(sequence, _images(tmp_path / "short", 1), fps=15.0)
    with pytest.raises(ValueError, match="image size"):
        remap_observation_sequence(sequence, _images(tmp_path / "large", shape=(50, 64)), fps=15.0)
    with pytest.raises(ValueError, match="FPS"):
        remap_observation_sequence(sequence, _images(tmp_path / "wrong-fps"), fps=30.0)


def test_canonical_cli_argument_rules():
    import run

    parser = run.build_arg_parser()
    with pytest.raises(SystemExit):
        run.validate_cli_args(parser, parser.parse_args(["--input", "x", "--frontend", "canonical"]))
    with pytest.raises(SystemExit):
        run.validate_cli_args(parser, parser.parse_args(["--input", "x", "--observations", "x.pkl"]))
    with pytest.raises(SystemExit):
        run.validate_cli_args(parser, parser.parse_args([
            "--input", "x", "--frontend", "canonical", "--observations", "x.pkl",
            "--mint_predictions", "mint.npz",
        ]))


def test_stage_zero_artifact_reaches_hamer_then_hawor_without_detector(tmp_path, monkeypatch):
    import run

    current_paths = _images(tmp_path / "input")
    old_paths = _images(tmp_path / "previous")
    artifact = tmp_path / "source.pkl"
    save_observation_sequence(_sequence(old_paths), artifact)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    forbidden = Mock(side_effect=AssertionError("detector must not run"))
    monkeypatch.setattr(run, "YOLO", forbidden)
    model = Mock()
    model.to.return_value = model
    monkeypatch.setattr(
        run,
        "load_backend",
        lambda backend, *_args: SimpleNamespace(
            model=model, render_cfg=Mock(), backend_name=backend
        ),
    )
    monkeypatch.setattr(run, "Renderer", Mock())
    seen = []

    def reconstruct(frames, backend_bundle, *_args, **_kwargs):
        seen.append((backend_bundle.backend_name, deepcopy(frames)))
        return {frame["img_path"]: {"mano": []} for frame in frames}

    monkeypatch.setattr(run, "run_mesh_recovery", reconstruct)

    stage_artifact = None
    for backend in ("hamer", "hawor"):
        output_root = tmp_path / f"output-{backend}"
        input_artifact = artifact if stage_artifact is None else stage_artifact
        monkeypatch.setattr("sys.argv", [
            "run.py", "--input", str(current_paths[0].parent), "--frontend", "canonical",
            "--observations", str(input_artifact), "--backend", backend,
            "--output_root", str(output_root),
        ])
        run.main()
        run_dir = output_root / "input_canonical"
        stage_artifact = run_dir / "stages/00_frontend/observations.pkl"
        assert stage_artifact.is_file()
        stage_sequence = load_observation_sequence(stage_artifact)
        assert stage_sequence["frontend"]["name"] == "mint"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["frontend"] == {
            "name": "canonical",
            "source_frontend": "mint",
            "input_artifact": str(input_artifact.resolve()),
            "schema_version": OBSERVATION_SCHEMA,
        }
        summary = json.loads((run_dir / "stages/00_frontend/summary.json").read_text())
        assert summary["frontend_mode"] == "canonical"
        assert summary["source_frontend"] == "mint"

    assert [backend for backend, _frames_seen in seen] == ["hamer", "hawor"]
    assert all(
        "hands" in frame and "selected_for_hamer" not in frame
        for _backend, frames_seen in seen
        for frame in frames_seen
    )
    forbidden.assert_not_called()
