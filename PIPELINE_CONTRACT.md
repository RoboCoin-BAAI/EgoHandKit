# EgoHandKit Pipeline Contract

The visual frontend is the only component that knows how observations are
obtained. It emits `egohand.observations.v1`: sequence metadata plus one frame
record for every input image, with `hands`, bounding boxes, 21 keypoints,
handedness, physical track and fragment identifiers, confidence, and source
metadata. `observation_frontend/schema.py` validates this boundary.

The downstream stages are frontend-independent:

```text
frontend -> canonical observations -> depth gate -> motion gate -> backend
         -> endpoint wrist gate -> temporal smoother -> canonical results
```

Depth and motion gates preserve canonical frames and split fragments when they
reject an observation. YOLO checking is diagnostic-only and has
`decision_effect: none`.

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
