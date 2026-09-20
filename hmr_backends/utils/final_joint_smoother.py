"""Apply the existing camera smoother after HMR/frontend hand selection."""

import numpy as np

from hmr_backends.utils.mint_style_smoother import smooth_camera_joint_sequence


def smooth_final_rows(rows, segment_keys, *, q=0.6, r=0.6, beta=2.0, max_jump_m=0.2):
    """Smooth in place without changing presence, provenance, or timestamps.

    Source switches are intentionally continuous. Missing hands/frame indices
    and physical identity/fragment changes are hard boundaries.
    """
    if not np.isfinite([q, r, beta]).all() or q <= 0 or r <= 0 or beta < 0:
        raise ValueError('Invalid final joint smoother parameters')
    if not np.isfinite(max_jump_m) or max_jump_m <= 0:
        raise ValueError('Final smoother jump limit must be finite and positive')
    report = {'q': q, 'r': r, 'beta': beta, 'max_jump_m': max_jump_m,
              'smoothed_hands': 0, 'jump_boundaries': [], 'segments': []}

    def flush(indices, side):
        if not indices:
            return
        points = np.asarray([rows[i][f'{side}_joints_3d'] for i in indices], dtype=np.float32)
        record = {'side': side, 'start_frame': rows[indices[0]]['frame_idx'],
                  'end_frame': rows[indices[-1]]['frame_idx'], 'count': len(indices),
                  'smoothed': False, 'max_displacement_m': 0.0}
        report['segments'].append(record)
        if len(indices) < 4:
            record['reason'] = 'short_segment'
            return
        try:
            result = smooth_camera_joint_sequence(
                points, [rows[i]['frame_idx'] for i in indices], q=q, r=r, beta=beta)
        except (ValueError, np.linalg.LinAlgError) as exc:
            record['reason'] = str(exc)
            return
        for i, joints in zip(indices, result):
            rows[i][f'{side}_joints_3d'] = joints.tolist()
        record['smoothed'] = True
        record['max_displacement_m'] = float(np.linalg.norm(result - points, axis=-1).max())
        report['smoothed_hands'] += len(indices)

    for side in ('left', 'right'):
        run = []
        for i, row in enumerate(rows):
            if not row[f'{side}_present']:
                flush(run, side)
                run = []
                continue
            jump = False
            if run:
                previous = np.asarray(rows[run[-1]][f'{side}_joints_3d'])
                current = np.asarray(row[f'{side}_joints_3d'])
                jump = np.max(np.linalg.norm(current - previous, axis=-1)) > max_jump_m
                if jump:
                    report['jump_boundaries'].append({'side': side, 'frame_idx': row['frame_idx']})
            if run and (row['frame_idx'] != rows[run[-1]]['frame_idx'] + 1
                        or segment_keys[i][side] != segment_keys[run[-1]][side] or jump):
                flush(run, side)
                run = []
            run.append(i)
        flush(run, side)
    return report
