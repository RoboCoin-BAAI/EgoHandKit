"""Frontend-independent single-pass raw backend endpoint wrist gate."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def rotation_geodesic_deg(ref: Any, pred: Any) -> np.ndarray | float:
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    delta = np.matmul(np.swapaxes(ref, -1, -2), pred)
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    result = np.degrees(np.arccos(cosine))
    return float(result) if result.ndim == 0 else result


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def apply_endpoint_wrist_gate(outputs: Iterable[Any], *, endpoint_wrist_max_deg: float = 100.0) -> tuple[list[Any], dict[str, Any]]:
    """Reject only extreme raw wrist jumps at original run endpoints."""
    report = {"schema_version": "egohand.endpoint_wrist_gate.v1",
              "config": {"endpoint_wrist_max_deg": float(endpoint_wrist_max_deg)},
              "counts": {"evaluated": 0, "accepted": 0, "rejected": 0,
                          "rejected_start": 0, "rejected_end": 0, "not_evaluable": 0},
              "frames": []}
    outputs = list(outputs)
    endpoint_info = {}
    grouped = {}
    for index, output in enumerate(outputs):
        meta = getattr(output, "raw_backend_meta", {})
        key = (meta.get("physical_track_id"), output.hand_side, meta.get("physical_track_fragment_id"))
        grouped.setdefault(key, []).append((index, output))
    for sequence in grouped.values():
        sequence.sort(key=lambda item: item[1].frame_idx)
        runs, run = [], []
        for item in sequence:
            if run and item[1].frame_idx != run[-1][1].frame_idx + 1:
                runs.append(run); run = []
            run.append(item)
        if run: runs.append(run)
        for run in runs:
            length = len(run)
            role = "short_run" if length < 4 else None
            if length >= 4:
                start_jump = float(rotation_geodesic_deg(
                    _numpy(run[0][1].mano_params["global_orient"]).reshape(3, 3),
                    _numpy(run[1][1].mano_params["global_orient"]).reshape(3, 3)))
                end_jump = float(rotation_geodesic_deg(
                    _numpy(run[-2][1].mano_params["global_orient"]).reshape(3, 3),
                    _numpy(run[-1][1].mano_params["global_orient"]).reshape(3, 3)))
                for pos, jump, endpoint_role in ((0, start_jump, "start"), (-1, end_jump, "end")):
                    endpoint_info[run[pos][0]] = {"role": endpoint_role, "run_length": length,
                        "adjacent_wrist_jump_deg": jump, "threshold_deg": float(endpoint_wrist_max_deg),
                        "evaluated": True, "outlier": jump > endpoint_wrist_max_deg}
                for item in run[1:-1]:
                    endpoint_info[item[0]] = {"role": "interior", "run_length": length,
                        "adjacent_wrist_jump_deg": None, "threshold_deg": float(endpoint_wrist_max_deg),
                        "evaluated": False, "outlier": False}
            else:
                for item in run:
                    endpoint_info[item[0]] = {"role": role, "run_length": length,
                        "adjacent_wrist_jump_deg": None, "threshold_deg": float(endpoint_wrist_max_deg),
                        "evaluated": False, "outlier": False}

    kept = []
    for index, output in enumerate(outputs):
        meta = getattr(output, "raw_backend_meta", {})
        endpoint = endpoint_info[index]
        row = {"frame_idx": int(output.frame_idx), "side": output.hand_side,
               "physical_track_id": meta.get("physical_track_id"),
               "physical_track_fragment_id": meta.get("physical_track_fragment_id"),
               "endpoint": endpoint}
        reject = endpoint["role"] == "start" and endpoint["outlier"] or endpoint["role"] == "end" and endpoint["outlier"]
        reason = "segment_start_wrist_jump" if endpoint["role"] == "start" else "segment_end_wrist_jump" if endpoint["role"] == "end" else None
        if reject:
            row.update(valid=False, classification="reject", reject_reasons=[reason]); report["counts"]["rejected"] += 1
            report["counts"]["rejected_start"] += int(endpoint["role"] == "start"); report["counts"]["rejected_end"] += int(endpoint["role"] == "end")
        else:
            row.update(valid=True, classification="accepted", reject_reasons=[]); kept.append(output)
            report["counts"]["accepted"] += 1
        report["counts"]["evaluated"] += 1
        report["frames"].append(row)
        if not reject:
            output.raw_backend_meta["endpoint_wrist_gate"] = {"status": "accepted", "endpoint_role": endpoint["role"],
                "endpoint_wrist_jump_deg": endpoint["adjacent_wrist_jump_deg"], "endpoint_threshold_deg": float(endpoint_wrist_max_deg),
                "run_length": endpoint["run_length"]}
    groups = {}
    for side in ("left", "right"):
        for decision in ("accepted", "rejected"):
            rows = [row for row in report["frames"] if row["side"] == side and row["classification"] == ("accepted" if decision == "accepted" else "reject")]
            reasons = {}
            for row in rows:
                for reason in row.get("reject_reasons", []): reasons[reason] = reasons.get(reason, 0) + 1
            groups[f"{side}_{decision}"] = {"count": len(rows),
                "reject_reason_counts": reasons}
    report["groups"] = groups
    return kept, report
