# Priority 1 implementation and per-file undo

The priority 1 fixes use the **current HDF5 format only**. `Tz` is always a
three-component moment-at-CoP vector, shape `(3, n_samples)`. Older HDF5 files
must be regenerated from C3D; there are no migration paths.

## Review the implementation by file

| File | Main changes |
| --- | --- |
| `ibo_biomech/containers/forceData.py` | Full-length defaults; unknown geometry; shape/rotation validation; atomic processing; signal-only filtering and geometry-preserving downsampling. |
| `ibo_biomech/containers/_validation.py` | Shared clock, sampling-rate and channel alignment validation. |
| `ibo_biomech/containers/markerData.py` | Residuals, camera visibility, per-frame types and frame offsets; cropped metadata; virtual-marker clock/frame preservation. |
| `ibo_biomech/handlers/_force_schema.py` | One current force schema for conversion, loading and saving; strict vector and metadata validation. |
| `ibo_biomech/handlers/h5Handler.py` | Unit/channel/frame persistence; atomic saves; removals; selective loading; source annotation archival. |
| `ibo_biomech/handlers/c3dHandler.py` | Shared processed trial; processed versus raw writers; deterministic reloads; frame/validity/event preservation; explicit rejection of unsupported force edits. |
| `ibo_biomech/biomech_io/file_converter.py` | Reuses the HDF5 serializers; preserves source metadata; supports marker-only conversion. |
| `ibo_biomech/utils/utils.py` | Vector MOT documentation, clock validation and orthogonal plate axes from measured corners. |
| `tests/test_priority1_review.py` | Original priority 1 findings are now passing regression tests. |
| `tests/test_priority1_safety.py` | Additional validation, metadata, ownership, current-schema and failed-write checks. |
| `tests/conftest.py`, `tests/test_force_data.py`, `tests/test_marker_data.py` | Shared synthetic C3D fixtures and updated API/clock assertions. |
| `README.md`, `tests/README.md`, `docs/remaining-issues.md`, `docs/tutorials/*.md` | Updated behavior, validation evidence and remaining limitations. |

## Undo exactly one file

Pre-implementation copies are stored locally in `.priority1-undo/before/`.
These copies include your uncommitted work as it existed before this
implementation request. They are not copies of Git HEAD. The local directory
is ignored by Git and must be retained if you want to use these commands.

From the repository root, list available files without changing anything:

```bash
.venv/bin/python .priority1-undo/restore.py --list
```

Inspect only the changes to one file:

```bash
.venv/bin/python .priority1-undo/restore.py --diff ibo_biomech/containers/forceData.py
```

Restore just that file:

```bash
.venv/bin/python .priority1-undo/restore.py ibo_biomech/containers/forceData.py
```

Or, independently, restore just the HDF5 implementation:

```bash
.venv/bin/python .priority1-undo/restore.py ibo_biomech/handlers/h5Handler.py
```

Substitute any single path from `--list`. There is no restore-all action.
Newly created files can also be undone individually, which removes only the
selected new file. Before restoring, the script verifies that the file still
matches the completed implementation; it refuses to overwrite newer edits.
It also saves the version being undone under `.priority1-undo/undone/`.
You can instead ask for a selective undo of a named file if you have edited it
since completion.

Files can depend on one another. Restoring a helper alone can require matching
changes to its callers; restores never modify those other files automatically.
Run the relevant tests after choosing which implementation pieces to keep.

## Supported C3D writes

```python
from ibo_biomech import C3DHandler

handler = C3DHandler("recording.c3d")
trial = handler.load_data()  # shared processed state
trial.rotate_markers("x", 90)
handler.write_c3d("processed.c3d", trial)
handler.write_raw_c3d("raw_copy.c3d")  # explicitly the original structure
```

C3D platforms are reconstructed from calibrated analog channels. Direct edits
to derived force vectors/geometry, point-unit changes with existing force
platforms and independent force resampling must be saved as HDF5/MOT. They are
rejected before overwriting a C3D destination. Separate EMG/IK/ID results also
require HDF5. Source-aligned cropping, residuals, camera masks, frame offsets,
in-range events and analog reordering are supported.

For processed HDF5 crops, opaque events, rigid bodies and unlabeled trajectories
are retained under `SourceData`, labeled as belonging to the original recording.
They are not silently treated as aligned annotations for the new clock.

## Verification

- Automated suite: **167 passed, 1 expected failure**. The remaining expected
  failure is the existing priority 2 subject-filter bug.
- Two local recordings passed converted-unit HDF5/MOT and processed-marker C3D
  checks, using temporary destinations.
- All Python blocks in the synthetic processing tutorial passed.
- Per-file restore was exercised in an isolated temporary workspace, including
  preservation of unrelated files and refusal to overwrite newer edits.

OpenSim execution was not part of this implementation.
