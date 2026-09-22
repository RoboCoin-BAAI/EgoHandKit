# EgoHandKit Project Memory

Last updated: 2026-09-21.

## Authority and Snapshot

- `PIPELINE_CONTRACT.md`: current interface and behavior contract.
- `README.md`: recommended commands and production/compatibility option split.
- `HMR_PRIMARY_VALIDATION.md`, `HAND_IDENTITY_VALIDATION.md`: historical run evidence.
- Branch at this update: `feature/mint-frontend`; pre-fix baseline `d644f31`.
- This memory accompanies the projection fix and documentation commit. Inspect
  `git status` and recent history for the current revision and pending edits.

## Agreed Production Strategy

```text
canonical -> sensor wrist depth plus wrist-surface compensation / existing 1m limit -> motion gate
-> HMR inference / optional partial-hand depth recovery
-> image/depth duplicate identity check + HMR-first MINT consistency
-> endpoint wrist gate -> HMR/MINT selection -> final joint smoother
-> Parquet -> common colored MANO fitting -> final render
```

The entry point is `scripts/run_without_front.sh`. It enables HMR-first policy,
image/depth duplicate checking, partial recovery, stable anchoring, final joint
smoothing, Parquet export, and Parquet rendering. Direct `run.py` defaults differ.

- Severe rejection: wrist discrepancy > 0.15m or palm-vector discrepancy > 80deg,
  supported MINT metric reference, and 3 consecutive frames in one track fragment.
- Scale ratio is diagnostic in HMR-first mode. Old 0.08m/40deg/scale veto remains
  available outside this policy, not the production tuning interface.
- Duplicate checking uses HMR joint-box overlap, independent depth, and frontend
  image assignments. It does not require MINT metric-depth trust. Only the wrong
  side falls back when evidence is sufficient; overlap alone is not a veto.
- Final smoothing operates on selected joints. Missing frames, fragment changes,
  and excessive jumps split smoothing; source changes alone do not.
- Retained CLI alternatives are not dead code. Do not delete them merely because
  the wrapper does not enable them.
- As of the wrist-surface update, the production wrapper enables
  `--wrist_surface_depth_compensation`. Registered depth sampled at the wrist
  pixel is treated as visible surface depth and converted to joint-center depth
  with an orientation-aware positive Z offset, defaulting to 0.015-0.030m:
  palm/back-facing views use the smaller end and edge-on views use the larger
  end. Raw surface depth, offset, and compensated depth are diagnostic fields.

## Latest Confirmed Fix: Partial Recovery Projection

At frame 110 (3.667s), both predicted right wrists were outside the image. Visible
MCP depth estimates disagreed, so recovery borrowed uncalibrated MINT wrist Z.
Previously, stable recovery preserved only the HMR wrist ray and translated the
original relative XYZ hand. Other joints no longer matched HMR image predictions.

Measured example: index tip originally (1236.64, 847.97) px became approximately
(1274.6, 952.9) px after anchoring. Maximum joint projection displacement was
143.18px. Final smoothing was not the principal source of that displacement.

Current fix in `hmr_backends/utils/mint_3d_consistency.py`:

- For stable recovery (`wrist_ray_source == 'hmr'`), use recovered wrist Z plus
  original HMR relative joint Z, and back-project each valid HMR joint on its own
  image ray through registered camera intrinsics.
- Reuse the normal depth-anchoring helper. This preserves valid projections and
  relative Z, not original relative XYZ or guaranteed metric hand shape.
- Stable mode off retains legacy behavior. No new rejection thresholds added.

Verification: 205 tests and 6 subtests passed; two existing dependency deprecation
warnings. Replayed all 177 recovered stable hands from the reference run: maximum
valid-joint projection error < 4e-13px. These checks are before final smoothing
and MANO fitting, not proof of final visual accuracy.

The user reran the test and reported it as broadly satisfactory on 2026-09-21,
then requested committing the changes and documenting recommended parameters.
The supplied run command targets `egohandkit_projection_fix_test`. This is user
visual feedback, not a new agent-measured full-run artifact audit or a claim
that every remaining risk is resolved. Keep this as the working baseline.

## Deferred Case: 0.767-0.833s

Frames 23-25, left: HMR/MINT wrist discrepancy about 0.62-0.63m and maximum palm
vector angle 123-127deg. MINT depth support was inconsistent, so HMR-first policy
accepted HMR. The abnormal joints already existed before final smoothing.
The projected hand lies near another person's clothing, without clear visible
support for the wearer's left hand. This suggests a wrong-region prediction;
it does not establish a general anatomical impossibility test.

User explicitly agreed to defer this problem: no generic first-person pose
deletion gate now. MINT unreliability is not proof of HMR validity, but disagreement
alone is also insufficient to reject both. A future solution needs independent
support and false-positive evaluation, not thresholds fitted to these timestamps.

## Remaining Risks and Next Steps

1. The user reports the new run is broadly satisfactory. Avoid adding more rules
   without a new reproducible symptom; rendering can still differ from input
   joints due to smoothing and MANO fitting.
2. Existing sampled depth > 1m can remove both sources even if the sample landed
   on background. Known risk, deliberately not changed. Prioritize after visual
   acceptance, with an explicit depth-reliability policy and user agreement.
3. Borrowed MINT Z can be wrong even when 2D alignment is preserved. Do not equate
   image fit with accurate metric 3D.
4. Depth-source changes and smoothing boundaries can still produce jumps; more
   smoothing can introduce lag and is not a universal remedy.
5. A valid Parquet hand may fail MANO fitting and not render. Distinguish fit
   failure from selection/depth rejection or absent observations.

## Local Reproduction Data

All paths below are workstation-specific:

- Video: `/home/user/1/data/long/left_rectified_600.mp4` (600 frames, 30fps,
  1920x1050; timestamps here use zero-based frame index divided by FPS).
- MINT: `/home/user/1/data/long/mint/prediction.npz`.
- Depth root:
  `/home/user/1/新数据测试/processed_data/session_20260826_100651_整理货架_20260827190358/depth`.
- Extracted frames: `test_data/images/left_rectified_600/000000.jpg` etc.
- Reference run:
  `/home/user/1/data/long/egohandkit_identity_test_final/mint/left_rectified_600_canonical`.
- New run target:
  `/home/user/1/data/long/egohandkit_projection_fix_test/mint/left_rectified_600_canonical`.

Reference artifacts: `stages/40_backend_raw/outputs.pkl`,
`stages/45_mint_3d_consistency/checked_outputs.pkl`, consistency JSON,
`final/hand_tracking.parquet`, `final/final_joints_smoother.json`,
`final/parquet_mesh_fit.json`, and `final/render.mp4`.

The reference identity run retained 994 hand samples, rendered 992, and rejected
four duplicate left HMR outputs at frames 596-599 while retaining the right side.
These counts describe that historical run, not the projection-fix rerun.

Temporary diagnostic files `/tmp/egohandkit_cases_audit.py` and
`/tmp/egohandkit_cases_audit.json` may exist, but are not durable project assets.
