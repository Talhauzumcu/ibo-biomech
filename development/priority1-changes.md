# Priority 1 implementation notes

Reviewed **2026-10-02** against `ibo-biomech` 0.3.6. These notes are maintained
outside the public Sphinx documentation. The [remaining issues](remaining-issues.md)
track outstanding work.

The force layout uses the current HDF5 format. `Tz` is a three-component
moment-at-CoP vector, shape `(3, n_samples)`; incompatible force datasets must
be regenerated from C3D.

## Implementation by file

| File | Current behavior |
| --- | --- |
| `ibo_biomech/containers/forceData.py` | Full-length signal defaults; unknown geometry; shape/rotation validation; signal-only filtering and geometry-preserving downsampling; source sample spacing through `frame_step`. |
| `ibo_biomech/containers/_validation.py` | Shared clock, sampling-rate, source sample range and channel alignment validation. |
| `ibo_biomech/containers/markerData.py` | Residuals, per-frame types and source frame offsets; cropped metadata; virtual-marker clock/frame preservation. Camera visibility is not a stored field. |
| `ibo_biomech/handlers/_force_schema.py` | One force layout for conversion, loading and saving; vector, frame and sample metadata validation. |
| `ibo_biomech/handlers/h5Handler.py` | Unit/channel/frame persistence; atomic saves; signal removals; selective loading; nested metadata; independent rigid-body clocks; event annotations and range filtering. IK/ID replacement remains separate work. |
| `ibo_biomech/handlers/c3dHandler.py` | Shared processed trial; processed and raw writers; deterministic reloads; frame/residual/event preservation; rejection of unsupported derived-force edits. |
| `ibo_biomech/biomech_io/file_converter.py` | Reuses HDF5 serializers; preserves source parameter metadata; supports marker-only conversion. |
| `ibo_biomech/utils/utils.py` | Full-vector MOT export, clock validation and orthogonal plate axes from measured corners. |

## Saving contracts

Processed C3D output serializes markers and source analog channels. Platforms
are reconstructed from calibrated analogs. Direct derived-force edits,
independent force resampling, and point-unit changes with existing platforms
are rejected before overwriting the destination; use HDF5/MOT for those results.
Separate EMG/IK/ID collections require HDF5. Source-aligned cropping, residuals,
frame offsets, in-range event annotations and analog reordering are supported.
`write_raw_c3d()` preserves the original recording structure.

HDF5 saves supplied rigid-body samples with their own clocks and filters events
to the saved marker source-frame range. It does not create archival groups or
format-version metadata. Events include context, subject, icon ID and generic
flag; readers default these annotations when the optional datasets are absent.
Camera-mask datasets are removed on processed saves.

## Verification

The 2026-10-02 suite run produced **352 passed, 1 skipped, 1 expected failure**.
The skip needs an optional local HDF5 reference file. The expected failure is
subject filtering; all priority 1 regressions pass.

- `test_priority1_review.py` and `test_priority1_safety.py` cover force layouts,
  clocks, geometry, atomic processing/saves, metadata, selective saves and C3D
  ownership/writing.
- `test_sample_metadata.py` covers nested metadata, parameter snapshots,
  source frame ranges, force sample spacing and round trips.
- `test_events.py`, `test_rigid_body.py`, `test_rigid_body_h5.py` and
  `test_h5_standard_format.py` cover current event annotations, independent
  rigid-body clocks and fixed-format round trips.

The previous notes described a local `.priority1-undo/restore.py` tool and
backups. That directory is absent from the current checkout; those restore
commands cannot be used here. OpenSim execution is not covered by this suite.
