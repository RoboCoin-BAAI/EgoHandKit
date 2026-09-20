# Image/Depth Hand Identity Validation

Validated on 2026-09-21 against `8649e28` (HMR-first selection).

## Scope

The optional `--hmr_duplicate_image_gate` separates frontend image-position
support from frontend metric-depth trust. The production wrapper enables it.
The former 3D-only duplicate gate remains independently available. This change
does not modify the 1m depth veto, motion gate, anchoring, single-hand severe
thresholds, endpoint gate, final smoother, frontend contract, assets or loaders.
The exact evidence requirements and CLI defaults are in `PIPELINE_CONTRACT.md`.
No frame indices, video paths or temporal-window exceptions exist in the gate.

## Automated Checks

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/miniconda3/envs/egohandkit/bin/python -X faulthandler -m pytest tests -q`

200 tests and 6 subtests passed, with two existing SciPy deprecation warnings.
Compilation, shell syntax and diff whitespace checks passed. The numeric-only
fixture `tests/fixtures/overlapping_hmr_hands.json` captures both predictions
at frame 599 and is portable without model assets, dataset images or CUDA.
Tests cover depth/position decoupling, correct-side retention, single-side
Parquet fallback, true overlap/uncertain assignments, missing evidence,
conflicting side labels, ambiguous tracks, resolution changes, IoU boundaries,
CLI validation and disabled-gate compatibility.

## Same-Video Verification

Input: `/home/user/1/data/long/left_rectified_600.mp4` (600 frames, 30 FPS).
The existing canonical artifact was reused for full HaMeR inference, selection,
export and rendering. Output root:
`/home/user/1/data/long/egohandkit_identity_test_final/mint/left_rectified_600_canonical`.

The initial 0.5 IoU trial found only two frames in the sustained duplicate
event because articulation changed output-box extents. The configurable
default is 0.4, combined with independent depth and unambiguous reference
assignment, not used as a standalone rejection rule. Full-clip replay found
four candidates, all in the reported event; the final full run used 0.4.

| Measure | Before | Identity check |
|---|---:|---:|
| Exported present frame-hands | 994 | 994 |
| Consistency/identity accepted HMR | 994 | 990 |
| HMR after endpoint gate | 990 | 986 |
| Rendered meshes | 992 | 992 |
| Projected wrist steps > 50px | 17 | 17 |

Only left-hand sources at frames 596-599 changed from HMR to MINT. Their right
hands remain HMR. All exported joints before frame 596 are numerically identical
to the previous run (maximum absolute difference 0). Thus the earlier 3s, 4s,
9.9s and 11.97s outputs have not regressed or been retuned in this change.
Final left/right wrist distances over frames 596-599 are approximately
0.170, 0.167, 0.165 and 0.161 metres rather than both wrists occupying the
right-hand location. Rendered frames 597 and 599 were inspected: green left
geometry is restored over the left hand while blue right geometry is retained.
These are regression observations, not general pose-accuracy measurements.

The final video has all 600 frames; the existing renderer still declines two
other mesh fits. The existing 8 pre-inference and 7 partial-anchor depth-limit
rejections are unchanged. This is not a guarantee of coverage for every hand.

Comparison artifacts under `/home/user/1/data/long/egohandkit_identity_test_final`:
`comparison.mp4` (previous on left, identity check on right), `comparison.json`.

## Review

Standards: no remaining hard findings. Side-label disagreement now prevents a
new pair veto from targeting a different fallback export slot. The existing
raw-metadata protocol remains a non-blocking maintainability concern.

Spec: no remaining findings. Per-hand image-position quality is recorded even
when a pair cannot be evaluated. These final guards and diagnostics have unit
coverage; they do not alter the confirmed duplicate event's decisions.

## Limits

Position support is still a frontend-confidence/projection heuristic. If the
frontend itself confuses identities or both references coincide, this gate may
abstain. It does not relabel already mirrored MANO outputs, invent missing
geometry, or attempt to fix arbitrary single-hand pose errors. Larger/different
datasets are needed to establish general false-positive and recall rates.
The deferred sensor-background/1m problem is intentionally unchanged.

## Configuration Consolidation

The subsequent behavior-preserving cleanup centralizes duplicate-image defaults
and CLI/runtime configuration construction in `DuplicateImageConfig`. Validation:
201 tests and 6 subtests passed. Replaying all 994 captured HMR outputs produced
an identical complete gate-report SHA256 before and after the cleanup:
`a9f3c973d487cf7ab8d17740068593e616a750c3a2d19f2961c9114f3c7dc0d2`.
Thresholds, selection decisions and the existing rendered outputs are unchanged;
this cleanup did not rerun inference or rendering.
