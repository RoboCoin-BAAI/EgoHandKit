# Observation Frontend

## External observation frontend

External models such as MINT or ACE must be converted to the shared
`egohand.observations.v1` artifact before entering the runtime. The converter
owns each model's private cache format and projection conventions; EgoHandKit
only consumes the validated canonical observations.

Example:

```bash
python run.py --input /path/to/frames --frontend canonical \
  --observations /path/to/observations.pkl --backend hamer
```

The downstream depth gate, motion gate, endpoint gate and temporal smoother are
frontend-independent and work identically for canonical observations from any
producer.

## Optional Depth Veto

For sequences with exported stereo depth, use the generic `--depth_gate
--depth_dir` options. `--yolo_check` is diagnostic only: it compares canonical
boxes against legacy YOLO boxes and never changes HMR input.

## Scope

The migrated frontend is an explicit, offline evaluation path, not a claim of
better hand-pose accuracy. Enable it with `--frontend observations`. The legacy
frontend remains available and is the default. Omega is now disabled by default
for both paths; use `--omega_world` only when world-space output is wanted.

```text
Highest-score Detectron2 person detection (wearer-first by default)
  -> ViTPose original whole-body keypoints
  -> existing hand keypoint gate and EgoHandKit-compatible padded bboxes
  -> same-frame duplicate consolidation
  -> unchanged offline physical-hand temporal association
  -> original selected observation, with physical track/fragment metadata
  -> backend's existing crop generator
  -> HaWoR / WiLoR / HaMeR / HTM
  -> camera-space MANO parameters and full-timeline overlay
```

This path does not merge YOLO scores with ViTPose scores. It replaces both
legacy Pass 1 selection and Pass 2 cleanup. It never invokes the old handedness
correction, bbox interpolation, or short-track removal. `--use_vitpose` and
`--no_clean_bbox` are therefore rejected with this frontend rather than silently
ignored. Use `--observation_all_person` only for a recall experiment; injecting
other people's proposals into a two-slot association is not the default.
Same-frame consolidation remains enabled by default because the ViTPose left
and right hypotheses can be two overlapping detections of the same physical
hand. Use `--observation_no_consolidation` only as an ablation; keeping both
boxes can make one hand appear as two meshes that jump around.
Source provenance and licensing are in `observation_frontend/NOTICE.md`.

## Detection And Geometry

Detectron2 class 0 detections with score strictly greater than 0.5 are retained;
by default only the highest-scoring person is processed (all detections can be
enabled explicitly). Hand points use the last 42 whole-body points, split into
left/right groups of 21. The existing gate requires more than three points with
confidence strictly greater than 0.5. Their coordinate minima/maxima define the
raw hand bbox, then the same 1.2x padding as the legacy ViTPose detector is
applied. Original keypoints and original handedness are retained without
correction or averaging.

The padded bbox goes through the target backend's original crop rules,
including padding, resizing, normalization and left-hand flipping. There is no
new crop algorithm. The selected record also carries `backend_handedness`, a
track-level anatomical side used only by the HMR adapter to keep crop flipping
stable when ViTPose flickers for an isolated frame. The backend side is fixed to
the track's dominant side so it cannot flip the HaWoR crop on every ViTPose
label change. Raw `handedness` remains in the observation metadata for
diagnostics.

Physical track slots are anonymous, not anatomical left/right identities. Two
selected observations can have the same original handedness and must both
survive. HaWoR inference is split at a missing frame or a fragment change; the
backend side is fixed to each track's dominant side instead of following raw
per-frame labels. It never concatenates different physical hands solely because
their side labels match. This necessary temporal adaptation is not evidence that
HaWoR reproduces the source HaMeR results.

The association module also contains an optional two-frame constant-velocity
prior (`motion_prediction_weight`). It is disabled by default while sequence-
level tuning is in progress; enabling it adds predicted center/keypoint
residuals to the soft association cost without turning handedness into a hard
identity constraint.

This is a single-target, at-most-two-hands selection baseline. It does not
identify the camera wearer. With several real people visible, selected hands
can belong to different people. Long gaps may reuse a track slot with a new
fragment; equal track IDs alone do not establish identity across that gap.

## Run

```bash
conda activate egohandkit
python run.py \
  --input test_data/images/session_20260804_224442_clip_0649_60s_right_rectified \
  --frontend observations --backend hawor --gpu 0 --fps 30 \
  --img_focal 553.9148003055425

# Same selected observations, different hand model:
python run.py \
  --input test_data/images/session_20260804_224442_clip_0649_60s_right_rectified \
  --frontend observations --backend wilor --gpu 0 --fps 30 --batch_size 2
```

The original left-camera migration experiments used a different rectified image
stream (1920x1074). Never apply that stream's cached pixel coordinates to the
right-camera frames above. Its historical selection counts and accuracy claims
are not transferable to this new camera/backend combination.

## Output And Cache

Outputs are isolated in `test_data/hand_proc/<sequence>_observations/`, leaving
legacy results intact:

- `observations_raw.pkl`: all gated observations and image/frame metadata.
- `observations_selected.pkl`: full consolidation/association result.
- `observation_selection.json`: inspectable selection, clusters, unassigned
  observations, physical tracks, fragments, scores and actual algorithm configs.
- `<sequence>_<backend>.pkl`: reconstructed results, including empty frames.
- `render_<backend>.mp4`: camera-space hand overlay, including empty frames.

The pickle caches contain `signature` and `result` fields. Their signature
includes ordered image paths, resolved file sizes/mtimes, detector/pose assets,
config and adapter/selection code, and FPS. `--force_detect` rebuilds both caches.
These caches do not reuse the legacy highest-confidence-per-side cache.

With this frontend, output `tracked_ids` are anonymous physical slots;
`track_id_semantics` declares this explicitly. Per-hand `backend_meta` preserves
candidate ID, physical track/fragment IDs, original handedness, original bbox
and keypoints, and selection diagnostics. Anatomical side remains in
`mano[*].is_right`, independent of the track slot. Missing track states are kept
in `physical_tracks`; no missing mesh or bbox is fabricated.

Optional `timestamps.txt` uses zero-based frame index and timestamp seconds,
one row per input frame. It is retained as integer nanoseconds in results.
Association still uses frame distance, not timestamp differences. Images must
have consistent dimensions. Detector/pose models are released before loading
the hand model to bound GPU memory use.

## Verification

```bash
python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python scripts/validate_frontend_migration.py \
  /home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/failure_mining/full_60s/cache/full_sequence_temporal_pipeline.json
```

Plugin autoload is disabled because this host exports ROS pytest plugins that
are unrelated to this environment. The replay strips all downstream model and
depth fields, checks selected candidate/track/state/cluster/fragment equivalence,
verifies unchanged bbox/keypoints/handedness and hashes the source cache before
and after. This checks selection equivalence, not pose accuracy or end-to-end
speed.

### Migration Validation (2026-09-09)

- Automated suite: 47 tests passed, plus 6 subtests.
- Source-cache replay: all 1800 frames matched; 3489 raw observations became
  2973 consolidated observations and 2872 selected observations. No selected
  bbox/keypoint differences, no frame assignment differences, and the source
  cache hash stayed unchanged. Report: `outputs/frontend_migration/replay_report.json`.
- Real inference: HaWoR and WiLoR each processed the first 32 right-rectified
  frames through the new frontend, producing 64 hands. All numeric outputs
  were finite; original bbox/keypoints/side and physical-track metadata matched
  the selected observations exactly. Both videos decoded all 32 nonblank
  1920x1088 frames at 30 FPS. Sample overlays were also inspected visually.
- Smoke outputs: `test_data/hand_proc/observation_frontend_smoke_observations/`.
  Logs: `outputs/frontend_migration/hawor_smoke.log` and `wilor_smoke.log`.
  Omega was disabled. Full 60-second inference with this frontend has not yet
  been run; the 1800-frame check above is selection replay only.

The existing ViTPose checkpoint/config pairing reports unexpected
`backbone.blocks.*.mlp.experts.*` checkpoint keys. This migration leaves that
pairing unchanged, as required by the observation-only scope. Successful smoke
inference does not establish that all intended checkpoint weights were loaded
or that hand-pose accuracy improved. Resolve that configuration mismatch before
treating these runs as an accuracy benchmark.
