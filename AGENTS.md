# Working on EgoHandKit

## Read First

1. Inspect `git status`, the current branch, and recent commits. Never revert
   existing edits or overwrite a previous experiment's outputs.
2. Read `README.md` for supported entry points and production versus CLI defaults.
3. Read `PIPELINE_CONTRACT.md` for interfaces and stage behavior.
4. Read `PROJECT_MEMORY.md` for current decisions, evidence, and pending work.
5. Read the affected code and tests before editing. Historical validation reports
   are evidence for their named runs, not necessarily the current implementation.

## Constraints

- Canonical observations are the unified external frontend input. Do not bypass
  or redesign that contract to accommodate a particular model or example frame.
- Keep new validation features independently switchable through CLI flags.
  Preserve existing stages and compatibility unless explicitly asked otherwise.
- Do not change backend loading, MANO assets, or environment setup incidentally.
- The production path does not use YOLO for hand-presence validation.
- HMR is preferred for image fit. MINT is supporting evidence and fallback,
  not unconditional ground truth. Separate image-location support from metric
  depth trust. A sampled depth is not automatically a hand depth.
- Preserve missing-frame boundaries. Do not interpolate across missing hands or
  restore rejected HMR through smoothing.
- Final production smoothing is after HMR/MINT selection. Final colored MANO
  rendering uses Parquet joints; it is not the raw backend mesh.
- Do not implement timestamp-specific fixes. Diagnose raw inference, depth
  anchoring, selection, smoothing, and fitting separately before changing rules.
- Keep unrelated cleanup out of behavioral fixes. Removing an option from the
  recommended workflow does not authorize deleting its CLI or implementation.

## Validation and Outputs

On the current workstation:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/miniconda3/envs/egohandkit/bin/python -X faulthandler -m pytest tests -q
git diff --check
```

Add focused regression tests for changed behavior and cover disabled-mode
compatibility. Replay saved stage artifacts when possible before expensive
inference. Keep new runs in separate output roots. Do not claim visual acceptance
from unit tests or joint reprojection checks alone: verify final rendering too.

## Maintaining Memory

Update `PROJECT_MEMORY.md` when decisions, known risks, or validation status
change. Label observations, hypotheses, fixes, and unverified outcomes explicitly.
Update `PIPELINE_CONTRACT.md` for behavior changes and `README.md` for user-facing
commands/options. Keep stable rules here; avoid accumulating a conversation log.
Machine-specific dataset paths belong in project memory, not portable defaults.
