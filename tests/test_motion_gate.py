import cv2
import numpy as np

from observation_frontend.motion_gate import apply_motion_gate


def _images(tmp_path, count=6, shape=(100, 160)):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index:06d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((*shape, 3), dtype=np.uint8))
        paths.append(path)
    return paths


def _observation(box):
    x1, y1, x2, y2 = box
    points = [[(x1 + x2) / 2, (y1 + y2) / 2, 1.0]] * 21
    return {
        "bbox_xyxy": list(box),
        "observation_id": "right",
        "keypoints_2d": points,
        "handedness": "right",
        "backend_handedness": "right",
        "physical_track_id": 1,
        "physical_track_fragment_id": 0,
        "confidence": 1.0,
        "source": "mint",
        "meta": {},
    }


def _frames(boxes):
    return [
        {"frame_idx": index, "img_path": f"{index:06d}.jpg", "timestamp_ns": None, "hands": [_observation(box)]}
        for index, box in enumerate(boxes)
    ]


def test_isolated_spatial_outlier_is_removed_and_fragment_breaks(tmp_path):
    images = _images(tmp_path, 5)
    normal = (20, 20, 50, 50)
    outlier = (120, 20, 150, 50)
    gated, report = apply_motion_gate(_frames([normal, normal, outlier, normal, normal]), images)

    assert [len(frame["hands"]) for frame in gated] == [1, 1, 0, 1, 1]
    assert gated[3]["hands"][0]["physical_track_fragment_id"] == 1
    assert report["counts"]["rejected"] == 1
    assert report["frames"][2]["hands"]["right"]["classification"] == "isolated_or_return"


def test_persistent_jump_reacquires_as_new_segment(tmp_path):
    images = _images(tmp_path, 6)
    normal = (20, 20, 50, 50)
    moved = (120, 20, 150, 50)
    gated, report = apply_motion_gate(_frames([normal, normal, moved, moved, moved, moved]), images)

    assert len(gated[2]["hands"]) == 0
    assert [len(frame["hands"]) for frame in gated[3:]] == [1, 1, 1]
    assert gated[3]["hands"][0]["physical_track_fragment_id"] == 1
    assert report["counts"]["reacquired"] == 1


def test_frontend_fallback_bypasses_motion_without_affecting_neighbors(tmp_path):
    images = _images(tmp_path, 3)
    normal = (20, 20, 50, 50)
    outlier = (120, 20, 150, 50)
    frames = _frames([normal, outlier, normal])
    frames[1]["hands"][0]["meta"]["force_frontend_fallback"] = True

    gated, report = apply_motion_gate(frames, images)

    assert [len(frame["hands"]) for frame in gated] == [1, 1, 1]
    assert gated[2]["hands"][0]["physical_track_fragment_id"] == 0
    diagnostic = report["frames"][1]["hands"]["right"]
    assert diagnostic["classification"] == "frontend_fallback_bypass"
    assert report["counts"]["rejected"] == 0
