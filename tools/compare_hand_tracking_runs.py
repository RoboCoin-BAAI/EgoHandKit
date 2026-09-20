"""Compare selection and projected-wrist continuity without claiming accuracy."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def summarize(root, intrinsics):
    rows = pq.read_table(root / 'final/hand_tracking.parquet').to_pylist()
    gate = json.loads((root / 'stages/45_mint_3d_consistency/mint_3d_consistency.json').read_text())
    fit = json.loads((root / 'final/parquet_mesh_fit.json').read_text())
    hands = gate['hands']
    selected = {(h['frame_idx'], h['side']): h for h in hands}
    windows = set(range(87, 95)) | set(range(117, 125)) | set(range(295, 303)) | set(range(356, 362))
    steps, samples = [], []
    for side in ('left', 'right'):
        previous = None
        for row in rows:
            if not row[f'{side}_present']:
                previous = None
                continue
            wrist = np.asarray(row[f'{side}_joints_3d'])[0]
            point = (intrinsics @ wrist)[:2] / wrist[2]
            step = None
            if previous is not None and row['frame_idx'] == previous[0] + 1:
                step = float(np.linalg.norm(point - previous[1]))
                steps.append(step)
            previous = (row['frame_idx'], point)
            if row['frame_idx'] in windows:
                diagnostic = selected.get((row['frame_idx'], side), {})
                samples.append({'frame_idx': row['frame_idx'], 'side': side,
                                'source': row.get(f'{side}_source', row['source']),
                                'wrist_pixel': point.tolist(), 'wrist_depth_m': float(wrist[2]),
                                'step_px': step, 'gate_accepted': diagnostic.get('accepted'),
                                'reference_quality': diagnostic.get('reference_quality')})
    return {'frame_count': len(rows),
            'present_hands': sum(row[f'{side}_present'] for row in rows for side in ('left', 'right')),
            'gate_accepted_hands': sum(h['accepted'] for h in hands),
            'gate_rejected_hands': sum(not h['accepted'] for h in hands),
            'reject_reasons': dict(Counter(h['reject_reason'] for h in hands if not h['accepted'])),
            'reference_quality': dict(Counter(h.get('reference_quality', {}).get('reason') or 'trusted_or_legacy'
                                              for h in hands)),
            'rendered_hands': sum(h['rendered'] for h in fit['hands']),
            'duplicate_candidates': sum(p['duplicate_candidate'] for p in gate.get('pair_diagnostics', [])),
            'wrist_step_p95_px': float(np.percentile(steps, 95)) if steps else None,
            'wrist_steps_over_50px': sum(s > 50 for s in steps),
            'reported_windows': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--intrinsics', type=float, nargs=4, required=True, metavar=('FX', 'FY', 'CX', 'CY'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fx, fy, cx, cy = args.intrinsics
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    report = {name: summarize(root, k) for name, root in
              [('baseline', args.baseline), ('candidate', args.candidate)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({name: {key: value for key, value in result.items() if key != 'reported_windows'}
                      for name, result in report.items()}, indent=2))


if __name__ == '__main__':
    main()
