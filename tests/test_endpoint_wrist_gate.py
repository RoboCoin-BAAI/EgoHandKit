from types import SimpleNamespace

import numpy as np

from hmr_backends.utils.endpoint_wrist_gate import apply_endpoint_wrist_gate, rotation_geodesic_deg


def _output(frame, rotation=None, fragment=0):
    eye = np.eye(3) if rotation is None else rotation
    return SimpleNamespace(frame_idx=frame, hand_side="right",
        mano_params={"global_orient": eye, "hand_pose": np.tile(np.eye(3), (15, 1, 1))},
        raw_backend_meta={"physical_track_id": 1,
                          "physical_track_fragment_id": fragment})


def _z(degrees):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])


def test_geodesic_helpers():
    assert rotation_geodesic_deg(np.eye(3), np.eye(3)) == 0
    assert np.isclose(rotation_geodesic_deg(np.eye(3), _z(90)), 90)


def test_four_frame_run_rejects_only_original_endpoints():
    rows = [_output(0), _output(1, _z(143)), _output(2, _z(72.2)), _output(3, _z(224.9))]
    kept, report = apply_endpoint_wrist_gate(rows)
    assert [item.frame_idx for item in kept] == [1, 2]
    assert report["frames"][0]["reject_reasons"] == ["segment_start_wrist_jump"]
    assert report["frames"][3]["reject_reasons"] == ["segment_end_wrist_jump"]


def test_normal_endpoints_and_interior_are_kept():
    rows = [_output(0), _output(1, _z(73)), _output(2, _z(150)), _output(3, _z(73))]
    kept, report = apply_endpoint_wrist_gate(rows)
    assert len(kept) == 4
    assert report["frames"][1]["endpoint"]["role"] == "interior"


def test_short_runs_never_reject():
    for length in (2, 3):
        rows = [_output(i, _z(170) if i else None) for i in range(length)]
        assert len(apply_endpoint_wrist_gate(rows)[0]) == length


def test_missing_frame_and_fragment_split_runs():
    rows = [_output(10), _output(11), _output(12), _output(14), _output(15), _output(16)]
    kept, report = apply_endpoint_wrist_gate(rows)
    assert len(kept) == 6
    assert all(row["endpoint"]["role"] == "short_run" for row in report["frames"])
    rows[3].raw_backend_meta["physical_track_fragment_id"] = 1
    kept, report = apply_endpoint_wrist_gate(rows)
    assert len(kept) == 6


def test_finger_and_absolute_mint_wrist_are_diagnostic_only():
    row = _output(0, _z(170))
    row.mano_params["hand_pose"][:] = _z(170)
    kept, report = apply_endpoint_wrist_gate([row])
    assert kept == [row] and report["counts"]["rejected"] == 0
