# Migrated Observation Selection

Source repository: `/home/user/roboego-hand-vis`
Source commit: `3c2cc6aacbc4e56b0819b742bd3eca68dc0fcda5`
Source documentation: `docs/pipeline/pre_hamer_observation_frontend_migration.md`
License: Apache-2.0; the source license is preserved in `LICENSE` here.

The following files are copied from `roboego_hand_vis/egohand/`, with only the
internal import prefix changed from `roboego_hand_vis.egohand` to
`observation_frontend`:

- `pre_hamer_observation_frontend.py`
- `hand_observation_consolidation.py`
- `physical_hand_temporal_association.py`
- `failure_diagnostics.py`

Their three corresponding source test files are copied into `tests/` with the
same import-prefix adaptation. Selection weights, thresholds, DP, geometry and
representative selection are unchanged. `adapter.py` is the new EgoHandKit
integration, not an upstream algorithm change. No source repository files or
cached model outputs were modified.
