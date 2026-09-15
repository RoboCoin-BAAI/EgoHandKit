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
