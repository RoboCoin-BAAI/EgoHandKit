# HMR-First Refinement Validation

Validated on 2026-09-20 against baseline commit `a7df7a6`.

## Scope

This refinement keeps canonical input, backends, MANO assets, depth/motion/
endpoint gates, final UKF/RTS smoothing and Parquet-only colored mesh rendering.
New opt-in behavior is documented in `PIPELINE_CONTRACT.md`. It does not promise
to solve every occlusion, pose error or source transition.

## Automated Checks

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/miniconda3/envs/egohandkit/bin/python -m pytest tests -q`

177 tests and 6 subtests passed. Two pre-existing smplx/chumpy SciPy deprecation
warnings remain. Python compilation, shell syntax and diff whitespace checks
passed. Spec review found two issues (missing nearest-reference labels and
ambiguous same-side tracks); both were fixed with regression tests. Standards
review found no hard violations; optional geometry/config consolidation remains
a maintainability suggestion, not part of this behavioral change.

## Same-Video Run

Input: `/home/user/1/data/long/left_rectified_600.mp4`, 600 frames at 30 FPS.
The canonical artifact from `egohandkit_partial_hand_test` was reused unchanged.
Output: `/home/user/1/data/long/egohandkit_hmr_primary_test/mint/left_rectified_600_canonical`.

Enabled new options: `--hmr_primary_policy --hmr_stable_wrist_anchor`.
Severe thresholds: 0.15m, 80 degrees, 3 consecutive frames.
Duplicate decisions remained diagnostic-only; no duplicate candidate was found
in this clip. Synthetic tests cover duplicate suppression, actual hand overlap,
unreliable references and ambiguous same-side tracks. Real duplicate-positive
examples are still needed before recommending automatic suppression by default.

| Measure | Previous partial-hand run | HMR-first run |
|---|---:|---:|
| Valid exported frame-hands | 994 | 994 |
| Accepted by consistency gate | 845 | 994 |
| HMR after endpoint gate | 844 | 990 |
| Rendered meshes | 988 | 992 |
| Adjacent projected wrist steps > 50px | 24 | 17 |
| Projected wrist step P95 | 39.26px | 34.06px |

There were 1009 canonical hands, 8 pre-inference depth rejections and 7
partial-anchor depth rejections. The HMR-first consistency policy rejected none
in this clip: 543 references had sufficient depth support, and no supported
severe discrepancy persisted for 3 frames. Endpoint gating rejected 4 HMR
samples, which used MINT fallback. Final sensor wrist depths did not exceed 1m.
Two present hands exceeded the renderer's existing fit validity/error policy;
presence in Parquet does not guarantee a rendered mesh.

Right-wrist projected steps at reported problem frames:

| Frame | Previous | HMR-first |
|---|---:|---:|
| 118 | 121.50px | 14.74px |
| 119 | 213.43px | 8.59px |
| 122 | 203.24px | 13.52px |
| 297 | 155.13px | 48.45px |
| 298 | 108.84px | 94.58px |
| 359 | 45.96px | 4.97px |

Frame 298 remains a large step. Both frames now use HMR; its raw projected
wrist moves about 64.6px and the canonical fragment changes at 298, so the
existing smoother cannot smooth across that boundary. This is not evidence
that all residual jitter is fixed. Continuity and retention are not ground-truth
accuracy metrics.

The final video and side-by-side comparison have all 600 frames. Frame overlays
119 and 359 were inspected for anchoring and side colors. Comparison artifacts:
`/home/user/1/data/long/egohandkit_hmr_primary_test/comparison.mp4` (old left,
new right), and `comparison.json`. Reproduce metrics using
`tools/compare_hand_tracking_runs.py` with the registered RGB intrinsics.

The wrapper's first MINT conversion attempt and one comparison-tool invocation
exited with native SIGSEGV. Direct retries with Python faulthandler succeeded;
the canonical pipeline completed normally. No environment or assets were
changed, and the intermittent native-process failure is not claimed fixed.
