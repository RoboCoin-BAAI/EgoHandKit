"""Image-space hand identity evidence, independent of frontend metric depth."""

from dataclasses import asdict, dataclass

import numpy as np

from observation_frontend.failure_diagnostics import bbox_iou
from observation_frontend.schema import camera_joints_from_observation


@dataclass(frozen=True)
class DuplicateImageConfig:
    iou_min: float = 0.4
    depth_max_m: float = 0.06
    reference_separation: float = 0.5
    assignment_margin: float = 0.15

    def __post_init__(self):
        values = list(asdict(self).values())
        if (not np.isfinite(values).all() or min(values) <= 0 or self.iou_min > 1):
            raise ValueError('Duplicate image thresholds must be finite and positive; IoU <= 1')


def _projection(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3):
        return None
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if not valid[[0, 5, 9, 17]].all():
        return None
    xy = points[valid, :2]
    box = np.concatenate((xy.min(axis=0), xy.max(axis=0)))
    if np.any(box[2:] - box[:2] <= 0):
        return None
    return points[0, :2], box


def position_reference(metadata):
    """Return projected wrist/box and quality without consulting sensor depth."""
    confidence = metadata.get('confidence')
    if confidence is None or not np.isfinite(confidence) or confidence < 0.5:
        return None, {'trusted': False, 'reason': 'low_or_missing_confidence'}
    projection = _projection(metadata.get('keypoints_2d'))
    return projection, {'trusted': projection is not None,
                        'reason': None if projection is not None else 'missing_wrist_or_palm_projection'}


def _independent_wrist_depth(metadata):
    depth = metadata.get('depth_anchor', {}).get('hmr_wrist_depth_m')
    recovery = metadata.get('partial_hand_recovery', {})
    if depth is None and (recovery.get('status') == 'recovered'
                          and recovery.get('source') == 'visible_mcp_sensor'):
        wrist = np.asarray(recovery.get('wrist_camera'), dtype=float)
        if wrist.shape == (3,):
            depth = wrist[2]
    return float(depth) if depth is not None and np.isfinite(depth) and depth > 0 else None


def diagnose_image_duplicates(outputs, config):
    """Return pair evidence without mutating outputs or assigning anatomical labels.

    Image distances are normalized by mean frontend hand-box diagonal. Crop
    overlap alone is never used. Frontend depth is deliberately not consulted.
    """
    frames = {}
    for output in outputs:
        frames.setdefault(int(output.frame_idx), {}).setdefault(output.hand_side, []).append(output)
    report = []
    for frame, sides in sorted(frames.items()):
        pair = {'frame_idx': frame, 'duplicate_candidate': False,
                'swapped_assignment_candidate': False, 'rejected_side': None,
                'decision_effect': 'none', 'position_reference_quality': {}}
        report.append(pair)
        if any(len(sides.get(side, [])) != 1 for side in ('left', 'right')):
            pair['not_evaluated_reason'] = 'missing_or_ambiguous_side'
            continue
        hands = [sides[side][0] for side in ('left', 'right')]
        # Export selection uses canonical handedness. Do not veto a stable
        # backend slot when its reference would be exported into another slot.
        if any(hand.raw_backend_meta.get('handedness', hand.hand_side) != hand.hand_side
               or hand.raw_backend_meta.get('backend_handedness', hand.hand_side) != hand.hand_side
               for hand in hands):
            pair['not_evaluated_reason'] = 'frontend_backend_side_disagreement'
            continue
        references = []
        for side, hand in zip(('left', 'right'), hands):
            reference, quality = position_reference(hand.raw_backend_meta)
            references.append(reference)
            pair['position_reference_quality'][side] = quality
        projections = [_projection(hand.pred_keypoints_2d) for hand in hands]
        if any(value is None for value in references + projections):
            pair['not_evaluated_reason'] = 'missing_or_untrusted_projection'
            continue
        depths = [_independent_wrist_depth(hand.raw_backend_meta) for hand in hands]
        if any(depth is None for depth in depths):
            pair['not_evaluated_reason'] = 'missing_independent_hmr_depth'
            continue
        ref_wrists = np.stack([value[0] for value in references])
        hmr_wrists = np.stack([value[0] for value in projections])
        scale = float(np.mean([np.linalg.norm(value[1][2:] - value[1][:2]) for value in references]))
        distances = np.linalg.norm(hmr_wrists[:, None] - ref_wrists[None, :], axis=-1) / scale
        separation = float(np.linalg.norm(ref_wrists[0] - ref_wrists[1]) / scale)
        nearest = distances.argmin(axis=1)
        decisive = bool(np.all(np.abs(distances[:, 0] - distances[:, 1]) > config.assignment_margin))
        # Each reconstruction must be closer than half the reference separation
        # to its alleged target, not merely nearer than a very distant alternative.
        close_match = bool(np.all(distances.min(axis=1) < separation / 2))
        overlap = float(bbox_iou(projections[0][1], projections[1][1]))
        depth_difference = abs(depths[0] - depths[1])
        distinguishable = separation > config.reference_separation
        duplicate = bool(overlap >= config.iou_min and depth_difference <= config.depth_max_m
                         and distinguishable and decisive and close_match and nearest[0] == nearest[1])
        pair.update(duplicate_candidate=duplicate, hmr_bbox_iou=overlap,
                    hmr_wrist_depth_difference_m=depth_difference,
                    reference_wrist_separation=separation, reference_scale_px=scale,
                    assignment_distances=distances.tolist(),
                    nearest_reference_side={side: ('left', 'right')[int(nearest[i])]
                                            for i, side in enumerate(('left', 'right'))},
                    swapped_assignment_candidate=bool(distinguishable and decisive and close_match
                                                       and list(nearest) == [1, 0]))
        if duplicate:
            wrong_side = 1 - int(nearest[0])
            fallback = camera_joints_from_observation(hands[wrong_side].raw_backend_meta)
            if fallback is None or np.any(fallback[:, 2] <= 0):
                pair['not_evaluated_reason'] = 'missing_frontend_fallback_geometry'
            else:
                pair['rejected_side'] = ('left', 'right')[wrong_side]
    return report
