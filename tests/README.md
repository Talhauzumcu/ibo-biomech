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

## Priority 1 implementation coverage

All priority 1 regressions pass. The remaining strict expected failure documents the
priority 2 subject-filter bug; use `--runxfail` to expose it as a failure.

- `test_priority1_review.py` covers the original review findings: current-format
  round trips, geometry, clocks, vector moments, unit/sample metadata, filtering,
  repeated loading and processed C3D writes.
- `test_priority1_safety.py` checks validation before mutation, unknown geometry,
  layout validation, cropped validity/frame metadata, event removal,
  atomic saves, channel removals, skipped collections, processed/raw C3D
  ownership, force-edit rejection, analog remapping and duplicate labels.
- Marker arithmetic tests also verify clock/frame preservation needed by cropped
  virtual-marker workflows.

`test_sample_metadata.py` covers nested metadata, C3D parameter preservation,
subject/dataframe metadata, source sample indices, crops, force downsampling,
and HDF5/C3D round trips. Flat metadata and missing frame attributes are rejected.
Event annotations and independent rigid-body clocks have their own coverage in
`test_events.py`, `test_rigid_body.py`, `test_rigid_body_h5.py` and
`test_h5_standard_format.py`. Optional uncommitted reference recordings are
skipped when absent.

HDF5 uses one fixed layout without version metadata. Tests reject malformed
arrays, scalar free moments and incompatible layouts. The force
fixture uses vector `Tz` of shape `(3, n)`, nonzero geometry and proper rotations.

C3D export supports processed markers/source analogs and aligned cropping.
Direct derived-force edits must be written as HDF5/MOT; tests assert that C3D
rejects them before touching the destination.

The 2026-10-02 run produced **352 passed, 1 skipped, 1 expected failure**.
See [remaining issues](../development/remaining-issues.md) for the repository-only
priority 2/3 backlog.
OpenSim execution, gait-event correctness and release installation need separate
coverage. Some older tests still characterize existing ID units/EMG caching.
