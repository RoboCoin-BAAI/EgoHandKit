"""Conservative HMR-first decisions using independently supported 3D priors."""

import numpy as np

from hmr_backends.utils.mint_3d_consistency import (
    PALM_JOINTS, _vector_angle_deg, convert_hmr_to_camera_joints,
    convert_mint_to_camera_joints,
)


def reference_quality(metadata, tolerance_m=0.08):
    """A wrist sample alone is not evidence that it landed on the hand."""
    anchor = metadata.get('depth_anchor', {})
    depth = anchor.get('mint_wrist_depth_m')
    estimates = np.asarray(anchor.get('mint_mcp_wrist_depths_m', []), dtype=float)
    estimates = estimates[np.isfinite(estimates) & (estimates > 0)]
    confidence = metadata.get('confidence')
    reason = None
    if confidence is None or not np.isfinite(confidence) or confidence < 0.5:
        reason = 'low_or_missing_confidence'
    elif depth is None or not np.isfinite(depth) or depth <= 0:
        reason = 'missing_sensor_wrist'
    elif estimates.size < 3:
        reason = 'insufficient_palm_depth_support'
    elif np.ptp(estimates) > tolerance_m:
        reason = 'inconsistent_palm_depth'
    elif abs(float(np.median(estimates)) - depth) > tolerance_m:
        reason = 'wrist_palm_depth_disagreement'
    return {'trusted': reason is None, 'reason': reason,
            'palm_sample_count': int(estimates.size),
            'palm_wrist_depth_m': float(np.median(estimates)) if estimates.size else None}


def _reject(record, reason):
    record.update(accepted=False, reject_reason=reason, reject_reasons=[reason])


def refine_decisions(outputs, records, *, primary, wrist_max_m, angle_max_deg,
                     confirm_frames, depth_tolerance_m, duplicate_gate,
                     duplicate_distance_m, separation_m, assignment_margin_m):
    """Change decisions in place; never relabel a mirrored MANO prediction.

    Confirmation is offline: all samples of a sustained severe-error run are
    rejected together, not fed into a smoother while waiting for confirmation.
    """
    limits = [wrist_max_m, angle_max_deg, depth_tolerance_m,
              duplicate_distance_m, separation_m, assignment_margin_m]
    if (not np.isfinite(limits).all() or min(limits) <= 0 or angle_max_deg > 180
            or confirm_frames < 1 or int(confirm_frames) != confirm_frames):
        raise ValueError('Invalid HMR-first selection thresholds')
    by_frame, tracks = {}, {}
    for output, record in zip(outputs, records):
        meta = output.raw_backend_meta
        quality = reference_quality(meta, depth_tolerance_m)
        anchor = meta.get('depth_anchor', {})
        mint = convert_mint_to_camera_joints(
            meta, wrist_depth_m=anchor.get('mint_wrist_depth_m'),
            camera_intrinsics=anchor.get('camera_intrinsics'))
        hmr = convert_hmr_to_camera_joints(output)
        if mint is None or np.any(mint[:, 2] <= 0):
            quality.update(trusted=False, reason='missing_mint_geometry')
        record['reference_quality'] = quality
        record['legacy_reject_reasons'] = list(record['reject_reasons'])
        record['severe_reasons'] = []
        if hmr is not None and mint is not None:
            record['wrist_distance_m'] = float(np.linalg.norm(hmr[0] - mint[0]))
            angles = [_vector_angle_deg(hmr[j] - hmr[0], mint[j] - mint[0])
                      for j in PALM_JOINTS]
            record['vector_angle_deg'] = max(angles) if all(a is not None for a in angles) else None
            scale = np.linalg.norm(mint[9] - mint[0])
            record['scale_ratio'] = float(np.linalg.norm(hmr[9] - hmr[0]) / scale) if scale > 1e-8 else None
        if primary:
            record.update(accepted=True, reject_reason=None, reject_reasons=[])
            if hmr is None:
                _reject(record, 'missing_hmr_geometry')
            elif quality['trusted']:
                # A borrowed wrist cannot independently establish a position error.
                assisted = meta.get('partial_hand_recovery') is not None
                if not assisted and record['wrist_distance_m'] > wrist_max_m:
                    record['severe_reasons'].append('severe_wrist_distance')
                angle = record['vector_angle_deg']
                if angle is not None and angle > angle_max_deg:
                    record['severe_reasons'].append('severe_wrist_vector_angle')
            key = (output.hand_side, meta.get('physical_track_id'),
                   meta.get('physical_track_fragment_id'))
            tracks.setdefault(key, []).append((output.frame_idx, record))
        by_frame.setdefault(int(output.frame_idx), {}).setdefault(output.hand_side, []).append((hmr, mint, record))

    def finish(run):
        confirmed = len(run) >= confirm_frames
        for record in run:
            record['severe_error_confirmed'] = confirmed
            if confirmed:
                _reject(record, record['severe_reasons'][0])

    for track in tracks.values():
        run, previous = [], None
        for frame, record in sorted(track, key=lambda row: row[0]):
            if previous is not None and frame != previous + 1:
                finish(run)
                run = []
            if record['severe_reasons']:
                run.append(record)
            else:
                finish(run)
                run = []
            previous = frame
        finish(run)

    pairs = []
    for frame, sides in sorted(by_frame.items()):
        if 'left' not in sides or 'right' not in sides:
            continue
        pair = {'frame_idx': frame, 'duplicate_candidate': False,
                'swapped_assignment_candidate': False, 'decision_effect': 'none'}
        pairs.append(pair)
        if any(len(sides[side]) != 1 for side in ('left', 'right')):
            pair['not_evaluated_reason'] = 'ambiguous_same_side_tracks'
            continue
        entries = [sides[s][0] for s in ('left', 'right')]
        if not all(h is not None and m is not None and r['reference_quality']['trusted']
                   for h, m, r in entries):
            pair['not_evaluated_reason'] = 'missing_geometry_or_untrusted_reference'
            continue
        h, m = [np.stack([entry[i] for entry in entries]) for i in (0, 1)]
        distances = np.linalg.norm(h[:, None, 0] - m[None, :, 0], axis=-1)
        h_sep = float(np.linalg.norm(h[0, 0] - h[1, 0]))
        m_sep = float(np.linalg.norm(m[0, 0] - m[1, 0]))
        palm_sep = float(np.linalg.norm(h[0, list(PALM_JOINTS)].mean(0)
                                       - h[1, list(PALM_JOINTS)].mean(0)))
        nearest = distances.argmin(axis=1)
        decisive = np.all(np.abs(distances[:, 0] - distances[:, 1]) > assignment_margin_m)
        close_match = np.all(distances.min(axis=1) < separation_m)
        duplicate = bool(h_sep < duplicate_distance_m and palm_sep < duplicate_distance_m
                         and m_sep > separation_m and nearest[0] == nearest[1]
                         and decisive and close_match)
        swapped = bool(m_sep > separation_m and list(nearest) == [1, 0]
                       and decisive and close_match)
        pair.update(hmr_wrist_distance_m=h_sep, mint_wrist_distance_m=m_sep,
                    wrist_distance_difference_m=abs(h_sep - m_sep),
                    hmr_palm_distance_m=palm_sep, assignment_distances_m=distances.tolist(),
                    nearest_reference_side={side: ('left', 'right')[int(nearest[i])]
                                            for i, side in enumerate(('left', 'right'))},
                    duplicate_candidate=duplicate, swapped_assignment_candidate=swapped)
        if duplicate and duplicate_gate:
            wrong_side = 1 - int(nearest[0])
            record = entries[wrong_side][2]
            _reject(record, 'duplicate_hand_on_other_side')
            pair.update(decision_effect='frontend_fallback', rejected_side=('left', 'right')[wrong_side])
    return pairs
