# EgoHandKit Pipeline Contract

The visual frontend is the only component that knows how observations are
obtained. It emits `egohand.observations.v1`: sequence metadata plus one frame
record for every input image, with `hands`, bounding boxes, 21 keypoints,
handedness, physical track and fragment identifiers, confidence, and source
metadata. `observation_frontend/schema.py` validates this boundary.

The downstream stages are frontend-independent:

```text
frontend -> canonical observations -> depth gate -> motion gate -> backend
         -> MINT 3D consistency gate -> endpoint wrist gate
         -> optional HMR temporal smoother -> canonical results
         -> HMR/MINT selection -> optional final joint smoother -> Parquet export
```

Depth and motion gates preserve canonical frames and split fragments when they
reject an observation. YOLO checking is diagnostic-only and has
`decision_effect: none`.

## Optional HMR-First Refinement

The production wrapper enables `--hmr_primary_policy` within the existing
consistency stage, without changing canonical input or backend loading. The old
8cm/40-degree/scale policy remains available when this flag is off. In HMR-first
mode, wrist errors above 15cm or wrist-to-palm vector errors above 80 degrees
can veto HMR only against a reliable MINT reference. Scale is diagnostic only.
Reference reliability requires confidence >= 0.5, positive finite geometry,
a sensor wrist sample, and at least three MCP-derived wrist-depth samples.
Their spread and difference from wrist depth must each be <= 8cm. Missing or
unreliable reference evidence does not veto valid HMR. This is a conservative
depth support heuristic, not proof that MINT is ground truth.

Severe errors must persist for three consecutive frames in one side/track/
fragment (configurable, set `--hmr_error_confirm_frames 1` for instantaneous
decisions). The offline pass rejects the entire confirmed run before smoothing;
it neither waits inside the smoother nor interpolates rejected HMR. Canonical
MINT remains the fallback. Missing HMR still keeps available canonical data.
The existing 1m sensor limit, motion and endpoint gates retain their behavior.

With `--hmr_stable_wrist_anchor`, partial recovery prefers the HMR sensor wrist,
then consistent visible HMR MCP depth, then supported MINT sensor depth, then
original uncalibrated MINT depth. All auxiliary anchors supply only Z: XY is
back-projected along the HMR wrist ray, even outside the image. This avoids
transplanting a differently projected MINT wrist into a calibrated HMR track.
The older full-position anchoring remains available when the flag is off.

Pair diagnostics report wrist separation differences, nearest-reference
assignments, and possible swapped assignments. `--hmr_duplicate_hand_gate`
additionally rejects the wrong-side HMR when both HMR wrists AND palm centres
are within 6cm, reliable MINT wrists are separated by more than 15cm, and both
HMR wrists clearly match the same MINT hand (assignment margin > 8cm and each
match < 15cm). The correctly labelled HMR is retained and MINT fills the other
side. True frontend hand overlap does not meet these conditions. Swapped-label
and separation differences alone remain diagnostic; no mirrored MANO result is
silently relabelled. Automatic duplicate rejection is opt-in, not enabled by the
production wrapper until real duplicate examples are validated.

All selection occurs before the existing final UKF/RTS joint smoother. Source
changes alone do not split a track; missing frames, identity/fragment changes,
and large jumps still do. HMR-only and final smoothing cannot be enabled
together. Rendering continues to fit colored MANO meshes to final Parquet
joints, not to raw rejected HMR. Parquet adds `left_source` and `right_source`
so mixed-source frames remain auditable; its existing columns are unchanged.
This refinement does not claim to eliminate raw HMR pose jitter or independent
MANO fitting error. Diagnostics remain in `mint_3d_consistency.json` and backend
metadata, including original strict-policy reasons and reference quality.

## Existing Stage Semantics

For production MINT runs, `--mint_depth_sensor_anchor` samples registered
dataset depth at the projected MINT wrist. A valid sample becomes the wrist's
absolute depth anchor; disagreement with MINT's original absolute Z is kept as
diagnostic data and does not reject the observation. Sensor wrist depth greater
than `--depth_max_m` rejects the observation without HMR or MINT fallback (the
production script sets 1.0 metres; equality is accepted). When sensor depth is
unavailable, including a wrist outside the image, the observation is marked
`frontend_fallback`, is not
sent to HMR, and its original frontend 3D is retained in Parquet with source
`mint_frontend_fallback`. The older `--mint_depth_wrist_only` comparison mode
remains available for compatibility.

`--hmr_partial_hand_recovery` optionally overrides the frontend-only bypass
when at least `--hmr_partial_min_visible_joints` canonical joints are in view
and the crop intersects the image. The original observation is not modified.
Insufficiently visible hands still use the previous frontend fallback.
After HMR inference, this mode retains mirrored, root-relative HMR geometry
and chooses an absolute wrist anchor in this order: HMR sensor wrist, MINT
sensor wrist, consistent visible HMR MCP depth estimates, original MINT wrist.
MCP estimation requires at least three samples whose inferred wrist depths
span at most `--hmr_partial_depth_spread_max_m`. Original MINT wrist anchoring
is explicitly uncalibrated. Sensor-backed anchors beyond `--depth_max_m`
exclude both the HMR output and its MINT fallback; the frame is retained.
Assisted results bypass the independent 3D consistency veto, since their
position may depend on MINT and the missing wrist depth cannot support that
comparison. Normal fully anchored hands keep the original thresholds.
Endpoint gating remains enabled and inference/geometry failure still falls
back to MINT. Final smoothing and Parquet mesh rendering are unchanged.
`stages/42_partial_hand_recovery/report.json` and per-output backend metadata
record the anchor source and depth quality. Parquet sources distinguish
`hamer_mint_frontend_wrist`, `hamer_mint_sensor_wrist`,
`hamer_visible_mcp_sensor`, and `hamer_hmr_sensor_wrist` from full MINT fallback.

The optional MINT 3D consistency gate runs after backend inference. When a
canonical hand carries `meta.joints_3d_camera`, it compares camera-space wrist
position, wrist-to-palm direction and wrist-to-middle-MCP scale against the HMR
result. A rejected HMR sample is removed before endpoint gating and smoothing;
the canonical observation remains available as the Parquet fallback. Missing
MINT geometry never rejects an HMR sample.

`--render_frontend_fallback` adds orange canonical skeletons to the final video
for hands without accepted HMR meshes. It shares the Parquet fallback validity
and depth policy, draws the original canonical 2D projection, and never creates
or interpolates a MANO mesh. Off-screen joints are clipped for display only.
This remains an optional legacy visualization.

The production `scripts/run_without_front.sh` instead enables
`--render_hand_tracking_parquet`. It fits a common neutral-shape MANO mesh to
each present hand's final Parquet joints, including MINT fallback hands. Fitting
optimizes pose and uniform scale independently per sample, fixes the wrist,
mirrors left hands, and never edits Parquet or bridges missing frames. Rendering
uses the calibrated camera's fx/fy/cx/cy (scaled from depth to RGB dimensions).
No backend mesh or original frontend hand observation is read by this renderer.
Mesh colors match the HMR overlay: left green, right blue, regardless of source.
This is a visualization approximation, not a new HaMeR prediction. Absent hands,
invalid geometry and fits exceeding `--parquet_mano_fit_max_rmse_m` are not
drawn; per-hand errors are recorded in `final/parquet_mesh_fit.json`.
`tools/render_hand_tracking_parquet.py` can rerender an existing Parquet with
the original image directory and depth camera calibration, without running HMR.

Production uses `--final_joints_smoother` rather than `--temporal_smoother` to
avoid smoothing HMR twice. After HMR/MINT selection it applies the existing
UKF/RTS camera smoother to the wrist and root-relative 3D joints. Source changes
do not break a continuous track; missing hands/frame indices and physical
track/fragment changes do. Segments shorter than four samples remain unchanged.
Joint jumps above `--final_smoother_max_jump_m` (0.2m) also restart the smoother,
so an incompatible depth jump cannot drag neighboring valid hands across space.
No missing hands are created and presence/provenance/timestamps are preserved.
The resulting joints are written to Parquet and used by the mesh renderer;
`final_joints_smoother.json` reports segment boundaries and displacements.
The HMR-only smoother remains independently available for compatibility.

Before that comparison, both sources are anchored independently to registered
sensor depth at their own projected wrist pixel. Their remaining joint depths
come from root-relative MANO geometry, and each joint's image projection is
back-projected with the declared camera intrinsics. HaMeR's virtual
weak-perspective `cam_trans.z` is used for rendering only and is never treated
as metric depth. The same sensor-anchored representation is written to Parquet.
Back-projection prefers the calibrated RGB intrinsics in
`fast_foundation/fast_foundation_stereo_video_meta.json`; canonical/MINT
intrinsics are only a compatibility fallback when depth calibration metadata
is unavailable.
An observation without valid sensor wrist depth is exported using original
frontend 3D only when explicitly marked `frontend_fallback`; this is not
sensor-calibrated depth. Otherwise Parquet presence is false and its fixed-size
joint array contains NaNs.

The endpoint wrist gate groups raw backend outputs by track, side, and fragment,
splits on missing frame indices, and for runs of at least four frames rejects
only the original start or end when its adjacent raw wrist rotation exceeds
`--endpoint_wrist_max_deg` (default 100 degrees). Decisions are single-pass;
interior frames are never removed by this stage. The smoother runs afterward
and retains its existing gap and fragment behavior.

Runs are written below `<output_root>/<sequence>_<frontend>/` with a manifest,
stage artifacts under `stages/`, and versioned final files under `final/`.
`final/results.pkl` uses `egohand.results.v1` and contains every input frame,
including frames whose `hands` list is empty. New frontends only need to
implement the canonical observation adapter; no downstream algorithm changes
are required.

## Using an External Frontend

An external vision model can write one `egohand.observations.v1` artifact and
leave detection, tracking, and handedness decisions entirely outside
EgoHandKit:

```text
External model -> egohand.observations.v1 -> EgoHandKit canonical input
               -> Depth / Motion / HMR / Endpoint / Smoother
```

The minimal Python shape is:

```python
import numpy as np

from observation_frontend.schema import save_observation_sequence

sequence = {
    "schema_version": "egohand.observations.v1",
    "sequence": {
        "sequence_name": "clip",
        "frame_count": 1,
        "fps": 30.0,
        "image_width": 1920,
        "image_height": 1080,
    },
    "frontend": {"name": "custom", "version": "1.0"},
    "frames": [{
        "frame_idx": 0,
        "img_path": "/exported/frames/000000.jpg",
        "timestamp_ns": None,
        "hands": [{
            "observation_id": "left-0",
            "handedness": "left",
            "backend_handedness": "left",
            "bbox_xyxy": [100.0, 120.0, 420.0, 520.0],
            "keypoints_2d": np.zeros((21, 3), dtype=np.float32),
            "physical_track_id": 0,
            "physical_track_fragment_id": 0,
            "confidence": 0.95,
            "source": "custom",
            "meta": {},
        }],
    }],
}
save_observation_sequence(sequence, "observations.pkl")
```

Consume it directly with:

```bash
python run.py \
  --input /path/to/video.mp4 \
  --frontend canonical \
  --observations observations.pkl \
  --backend hamer \
  --output_root /path/to/output
```

The loader validates the canonical schema, frame count, ordered frame indices,
resolution, and FPS. Since extracted-frame directories can differ between
runs, `frame_idx` is the stable identity and each validated `img_path` is
remapped to the current input frame. The current run writes that remapped
sequence to `stages/00_frontend/observations.pkl`.

For the optional MINT 3D and wrist-depth gates, producers may include a finite
`[21,3]` OpenPose-ordered camera-space array in each hand's
`meta.joints_3d_camera`. This is optional metadata and does not change the
canonical observation schema. The same metadata mapping must declare
`joint_order: openpose21` and
`camera_frame: opencv_x_right_y_down_z_forward`; unsupported or undeclared
conventions are treated as a missing reference rather than guessed.
