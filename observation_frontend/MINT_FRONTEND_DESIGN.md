# MINT Frontend Boundary

This first-stage integration treats MINT as an offline frontend only. MINT's
prediction cache is decoded and projected into EgoHandKit observations; HaMeR
still reads the original RGB frames and remains the only MANO backend used by
`run.py`.

## Confirmed MINT conventions

- `pose_enc` is `[T, 9]`: camera translation `[0:3]`, quaternion `[3:7]`,
  field of view `[fov_h, fov_w]` at `[7:9]` in the canonical
  `absT_quaR_FoV` encoding. MINT's `StudentEngine.predict()` returns one row
  per frame.
- `hand` is `[T, 218]`, two contiguous `[T, 109]` hands (left, right). Each
  hand is translation `[0:3]`, global orientation 6D `[3:9]`, 15 joint poses
  6D `[9:99]`, and shape `[99:109]`. `hand_presence_logits` is `[T, 2]`;
  probabilities are sigmoid(logits), ordered left then right.
- MINT hand parameters are camera-frame MANO parameters. MINT's decoder
  subtracts the canonical MANO root and applies the documented left-hand
  pose-axis flip before forwarding MANO, so decoded joints/vertices are in an
  OpenCV camera frame (`x` right, `y` down, `z` forward).
- A pose encoding reconstructs intrinsics at the image size passed to
  `pose_encoding_to_extri_intri`: `fx = W/(2*tan(fov_w/2))`,
  `fy = H/(2*tan(fov_h/2))`, principal point `(W/2,H/2)`. The cache contract
  therefore stores `mint_input_hw` and optionally explicit
  `camera_intrinsics`.

## Cache schema (`mint_prediction_cache.v1`)

The `.npz` loader accepts these canonical arrays: `frame_idx [T]`,
`orig_hw [T,2]`, `mint_input_hw [T,2]`, `pose_enc [T,9]` or
`camera_intrinsics [T,3,3]`, `hand [T,218]` (or split
`left_hand/right_hand [T,109]`), and
`hand_presence` or `hand_presence_logits [T,2]`. Separate
`left_presence/right_presence` fields are also accepted. A dependency-light
export can include `left_joints_cam/right_joints_cam [T,21,3]` (the raw hand
arrays are still retained when available). `orig_to_mint [T,2,3]` and
`timestamp_ns [T]` are optional. `metadata_json` is a JSON object and may set
`intrinsics_space` to `mint_input` or `original`.

## Pixel mapping

The cache stores `orig_hw` and `mint_input_hw`. With no crop metadata, the
adapter performs the explicit resize-only mapping from MINT input pixels to
the original image. A crop or letterbox must provide `orig_to_mint` (`[2,3]`
or `[T,2,3]`) mapping original pixels to MINT input pixels; the adapter
inverts that affine before generating bboxes. Explicit intrinsics may declare
`metadata_json.intrinsics_space` as `mint_input` (default) or `original`.
There is no guessed crop offset or guessed camera principal point.

Projection rejects non-finite points, non-positive depth, and hands with no
projected point inside the original image. Remaining points are clipped only
when constructing the padded square bbox. All observation coordinates remain
in original-image pixels.

## Minimal HaMeR contract

Each emitted frame has `frame_idx`, `img_path`, and `selected_for_hamer`. Each
selected hand has `bbox_xyxy`, 21 projected `vitpose_keypoints_2d` rows
`[x,y,confidence]`, `handedness`, `backend_handedness`, stable physical track
and fragment IDs, and `observation_meta.source == "mint"` with
`confidence_type == "mint_presence_projected"`. This is consumed by the
existing `_collect_observation_inputs()` and `run_mesh_recovery()` paths.

`mint_adapter.py` accepts raw `left_hand`/`right_hand` arrays and decodes them
when a MANO model directory is supplied. For dependency-light exports and
tests, `left_joints_cam`/`right_joints_cam` may be included alongside the raw
arrays; no MINT Python import is required.

## Optional post-HMR smoothing

`--mint_style_smoother` applies a local port of MINT's UKF + unscented RTS
filter after HaMeR inference. It smooths each physical track in camera-space
translation, quaternion global orientation, axis-angle joint pose, and shape,
then converts rotations back to the backend's MANO matrices and recomputes
vertices for rendering. Tracks with fewer than four observations are left
unchanged. This stage is opt-in so the original MINT frontend behavior remains
reproducible.

## Optional pre-HaMeR depth gate

When `--mint_depth_gate` is enabled, `depth_gate.py` reads the exported
uint16 millimetre PNG sequence from `--mint_depth_dir`, rescales it to each
RGB frame if necessary, and computes the median positive depth inside the
MINT-projected bbox. A sample is rejected when it has no valid depth, falls
outside the configured absolute range, or jumps outside the configured ratio
relative to that side's recent valid-depth median. Rejected observations are
removed before `_collect_observation_inputs()` and before post-HMR smoothing.
Any rejection increments `physical_track_fragment_id`; the smoother groups by
that fragment, preventing a bad interval from being interpolated into a valid
MANO track. The depth report is stored as `mint_depth_gate.json` and included
in the observation cache configuration, so changing the depth source or
thresholds cannot silently reuse an old cache.

## Optional pre-HaMeR motion gate

`--mint_motion_gate` runs after the depth gate and before
`_collect_observation_inputs()`. It compares each side's accepted observation
with the previous consecutive observation using normalized bbox-center motion,
bbox area ratio, bbox IoU, and median projected-joint motion. By default at
least two failed tests are required before rejecting a sample. A two-frame
lookahead labels an isolated return (`A -> X -> A`) separately from a
persistent jump (`A -> B -> B`), but both cases start a new physical fragment
after the rejected transition. Therefore the optional MANO smoother never
interpolates from the old location into the new one. Motion thresholds and
lookahead length are included in the observation-cache configuration.
