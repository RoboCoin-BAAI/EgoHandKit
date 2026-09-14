from pathlib import Path

import cv2
import numpy as np

from observation_frontend.yolo_gate import apply_yolo_gate


class FakeDetector:
    def __init__(self, candidates_by_call):
        self.candidates_by_call = list(candidates_by_call)
        self.calls = []

    def predict(self, image, verbose=False):
        self.calls.append(image)
        return self.candidates_by_call[min(len(self.calls) - 1, len(self.candidates_by_call) - 1)]


def _image(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"{index:06d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((100, 160, 3), np.uint8))
        paths.append(path)
    return paths


def _frames(boxes):
    return [{"frame_idx": i, "img_path": f"{i:06d}.jpg", "selected_for_hamer": [{
        "bbox_xyxy": list(box), "handedness": "left", "physical_track_id": 0,
        "physical_track_fragment_id": 0, "observation_meta": {"source": "mint"},
    }]} for i, box in enumerate(boxes)]


def _candidate(box, confidence, cls="left"):
    return [{"bbox": list(box), "confidence": confidence, "class": cls}]


def _local(box, confidence, cls="left"):
    # For the 30x30 MINT box in these fixtures, the 1.5x ROI starts at 12.5.
    return _candidate(tuple(np.asarray(box, dtype=float) - [12.5, 12.5, 12.5, 12.5]), confidence, cls)


def test_missing_candidate_gets_one_frame_grace_then_rejects(tmp_path):
    images = _image(tmp_path, 3)
    gated, report = apply_yolo_gate(
        _frames([(20, 20, 50, 50)] * 3), images,
        FakeDetector([[], [], _local((20, 20, 50, 50), 0.1)]))
    assert [len(frame["selected_for_hamer"]) for frame in gated] == [1, 0, 0]
    assert report["frames"][0]["hands"]["left"]["classification"] == "no_yolo_candidate"
    assert report["frames"][0]["hands"]["left"]["valid"] is True
    assert report["frames"][1]["hands"]["left"]["classification"] == "no_yolo_candidate"
    assert report["frames"][1]["hands"]["left"]["valid"] is False


def test_low_confidence_is_removed(tmp_path):
    images = _image(tmp_path, 1)
    gated, report = apply_yolo_gate(
        _frames([(20, 20, 50, 50)]), images,
        FakeDetector([_local((20, 20, 50, 50), 0.1)]))
    assert len(gated[0]["selected_for_hamer"]) == 0
    assert report["frames"][0]["hands"]["left"]["classification"] == "low_confidence"


def test_matching_is_side_agnostic_and_strong_updates_anchor(tmp_path):
    images = _image(tmp_path, 1)
    gated, report = apply_yolo_gate(
        _frames([(20, 20, 50, 50)]), images,
        FakeDetector([_local((20, 20, 50, 50), 0.5, cls="right")]))
    row = report["frames"][0]["hands"]["left"]
    assert len(gated[0]["selected_for_hamer"]) == 1
    assert row["classification"] == "strong_match"
    assert row["best_yolo_class"] == "right"
    assert np.allclose(row["anchor_bbox_xyxy"], [19.5, 19.5, 49.5, 49.5])


def test_low_iou_is_rejected(tmp_path):
    images = _image(tmp_path, 1)
    _, report = apply_yolo_gate(
        _frames([(20, 20, 50, 50)]), images,
        FakeDetector([_local((100, 70, 130, 95), 0.9)]))
    row = report["frames"][0]["hands"]["left"]
    assert row["valid"] is False
    assert row["classification"] == "low_iou"


def test_strong_gray_strong_keeps_anchor_on_gray(tmp_path):
    images = _image(tmp_path, 3)
    boxes = [(20, 20, 50, 50)] * 3
    detector = FakeDetector([
        _local((20, 20, 50, 50), 0.5),
        _local((37, 20, 67, 50), 0.4),
        _local((24, 20, 54, 50), 0.5),
    ])
    gated, report = apply_yolo_gate(_frames(boxes), images, detector)
    assert [len(frame["selected_for_hamer"]) for frame in gated] == [1, 1, 1]
    assert report["frames"][1]["hands"]["left"]["classification"] == "gray_grace"
    assert np.allclose(report["frames"][1]["hands"]["left"]["anchor_bbox_xyxy"], [19.5, 19.5, 49.5, 49.5])


def test_second_gray_rejects_and_reacquisition_starts_new_fragment(tmp_path):
    images = _image(tmp_path, 4)
    boxes = [(20, 20, 50, 50)] * 4
    detector = FakeDetector([
        _local((20, 20, 50, 50), 0.5),
        _local((37, 20, 67, 50), 0.4),
        _local((37, 20, 67, 50), 0.4),
        _local((20, 20, 50, 50), 0.5),
    ])
    gated, report = apply_yolo_gate(_frames(boxes), images, detector, gray_grace_frames=1)
    assert [len(frame["selected_for_hamer"]) for frame in gated] == [1, 1, 0, 1]
    assert report["frames"][2]["hands"]["left"]["classification"] == "gray_timeout"
    assert gated[3]["selected_for_hamer"][0]["physical_track_fragment_id"] == 1
    assert report["frames"][3]["hands"]["left"]["classification"] == "reacquired_new_fragment"


def test_gradual_drift_does_not_bridge(tmp_path):
    images = _image(tmp_path, 4)
    boxes = [(20, 20, 50, 50)] * 4
    detector = FakeDetector([
        _local((20, 20, 50, 50), 0.5),
        _local((35, 20, 65, 50), 0.47),
        _local((100, 20, 130, 50), 0.9),
        [],
    ])
    gated, _ = apply_yolo_gate(_frames(boxes), images, detector)
    assert [len(frame["selected_for_hamer"]) for frame in gated] == [1, 1, 0, 1]
