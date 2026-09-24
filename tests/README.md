# ibo-biomech tests

The suite covers containers and selected C3D/HDF5/MOT processing workflows.
Fixtures generate deterministic signals and real C3D files in pytest temporary
directories. No private recordings or OpenSim installation are required.

## Run

```bash
python -m pip install -e . pytest
python -m pytest tests -q -rx
# Or use the existing project environment:
.venv/bin/python -m pytest tests -q -rx
```

The root `tests.py` is an interactive local-recording/OpenSim example script,
not part of this automated suite.

## Priority 1 implementation coverage

The implementation gives **167 passed, 1 expected failure**. All priority 1
regressions now pass. The remaining strict expected failure documents the
priority 2 subject-filter bug; use `--runxfail` to expose it as a failure.

- `test_priority1_review.py` covers the original review findings: current-format
  round trips, geometry, clocks, vector moments, unit/sample metadata, filtering,
  repeated loading and processed C3D writes.
- `test_priority1_safety.py` checks validation before mutation, unknown geometry,
  schema rejection, cropped validity/frame metadata, annotation archival,
  atomic saves, channel removals, skipped collections, processed/raw C3D
  ownership, force-edit rejection, analog remapping and duplicate labels.
- Marker arithmetic tests also verify clock/frame preservation needed by cropped
  virtual-marker workflows.

HDF5 supports only the current schema. Tests require clear rejection of older
or malformed schemas, scalar free moments and incompatible layouts. The force
fixture uses vector `Tz` of shape `(3, n)`, nonzero geometry and proper rotations.

C3D export supports processed markers/source analogs and aligned cropping.
Direct derived-force edits must be written as HDF5/MOT; tests assert that C3D
rejects them before touching the destination.

See [remaining issues](../docs/remaining-issues.md) for the priority 2/3 backlog.
OpenSim execution, gait-event correctness and release installation need separate
coverage. Some older tests still characterize existing ID units/EMG caching.
