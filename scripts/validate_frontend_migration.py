"""Replay only pre-network fields from a roboego-hand-vis diagnostic cache."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_frontend.pre_hamer_observation_frontend import select_pre_hamer_observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/frontend_migration/replay_report.json'))
    args = parser.parse_args()
    original_bytes = args.cache.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    cached = json.loads(original_bytes)
    frames = []
    lookup = {}
    for frame in cached['frames']:
        observations = []
        for observation in frame['observations']:
            item = {key: observation[key] for key in (
                'frame_idx', 'candidate_id', 'original_candidate_id', 'official_candidate_id',
                'person_index', 'person_score', 'handedness', 'bbox_xyxy',
                'vitpose_keypoints_2d', 'official_gate_passed',
            ) if key in observation}
            observations.append(item)
            candidate = item.get('original_candidate_id', item.get('official_candidate_id', item['candidate_id']))
            lookup[(frame['frame_idx'], candidate)] = item
        frames.append({'frame_idx': frame['frame_idx'], 'time_s': frame['time_s'], 'observations': observations})
    start = time.monotonic()
    result = select_pre_hamer_observations(
        frames, image_size=(cached['video_info']['width'], cached['video_info']['height']),
    )
    elapsed = time.monotonic() - start
    mismatches = []
    max_bbox_error, max_keypoint_error = 0.0, 0.0
    keys = ('track_id', 'state', 'selected_candidate_id', 'selected_cluster_id', 'track_fragment_id')
    for frame, expected in zip(result['frames'], cached['frames']):
        actual_tracks = [{key: row[key] for key in keys} for row in frame['physical_tracks']]
        expected_tracks = [{key: row[key] for key in keys} for row in expected['temporal_tracks']]
        if actual_tracks != expected_tracks:
            mismatches.append(frame['frame_idx'])
        for selected in frame['selected_for_hamer']:
            candidate = selected.get('original_candidate_id', selected.get('official_candidate_id', selected['candidate_id']))
            original = lookup[(frame['frame_idx'], candidate)]
            max_bbox_error = max(max_bbox_error, float(np.abs(np.asarray(selected['bbox_xyxy']) - original['bbox_xyxy']).max()))
            max_keypoint_error = max(max_keypoint_error, float(np.abs(np.asarray(selected['vitpose_keypoints_2d']) - original['vitpose_keypoints_2d']).max()))
            assert selected['handedness'] == original['handedness']
    report = {
        'source_cache': str(args.cache.resolve()), 'source_sha256': original_hash,
        'source_unchanged': hashlib.sha256(args.cache.read_bytes()).hexdigest() == original_hash,
        'elapsed_seconds': elapsed, 'summary': result['summary'],
        'mismatched_frames': mismatches, 'max_bbox_difference': max_bbox_error,
        'max_keypoint_difference': max_keypoint_error,
        'network_inference': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert len(result['frames']) == len(cached['frames'])
    assert not mismatches and max_bbox_error == 0 and max_keypoint_error == 0
    assert report['source_unchanged']


if __name__ == '__main__':
    main()
