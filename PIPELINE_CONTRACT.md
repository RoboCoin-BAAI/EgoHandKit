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
         -> temporal smoother -> canonical results -> Parquet export
```

Depth and motion gates preserve canonical frames and split fragments when they
reject an observation. YOLO checking is diagnostic-only and has
`decision_effect: none`.

The optional MINT 3D consistency gate runs after backend inference. When a
canonical hand carries `meta.joints_3d_camera`, it compares camera-space wrist
position, wrist-to-palm direction and wrist-to-middle-MCP scale against the HMR
result. A rejected HMR sample is removed before endpoint gating and smoothing;
the canonical observation remains available as the Parquet fallback. Missing
MINT geometry never rejects an HMR sample.

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
An observation without a valid sensor wrist depth is not exported as metric
3D; Parquet presence is false and its fixed-size joint array contains NaNs.

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
