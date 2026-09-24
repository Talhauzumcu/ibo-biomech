# Remaining issues and proposed fixes

Updated **2026-09-24** after implementing the priority 1 fixes. The HDF5 policy
is **current format only**, as requested: old files must be regenerated from
source recordings. There are no legacy aliases, layout migrations or scalar
free-moment compatibility paths.

## Priority 1 review outcome

| Issue | Status | Implemented behavior |
| --- | --- | --- |
| 1. HDF5 force schema | Resolved for the current format | Converter/reader/saver share one schema; required shapes, counts and coordinate metadata are checked and refreshed. |
| 2. Downsampling | Resolved | Signals are anti-aliased; geometry and clocks are sampled at original indices; validation precedes mutation. |
| 3. Units and sample metadata | Resolved for markers/forces/analog/EMG | Current units, channels, clocks, frame limits, residuals, camera visibility and virtual status survive saving. |
| 4. Free-moment vector | Resolved | `Tz` always contains all three components, shape `(3, n_samples)`; scalar/missing HDF5 `Tz` is rejected. |
| 5. Geometry versus filtering | Resolved | Absent signals use the full sample count; unknown geometry is NaN; rejected operations leave objects unchanged. |
| 6. C3D ownership/writing | Resolved with an explicit supported scope | The returned trial shares processed containers; markers/analogs are serialized; unsupported derived-force edits are rejected; raw export is separate. |

## Implemented contracts and limitations

### 1. Current HDF5 force schema

`handlers/_force_schema.py` defines schema **2**, the only accepted force schema.
The required arrays are `Force`, `Moment`, `COP`, `Tz`, `Corners`, `Origin`,
`Position` and `Rotation`; `Time` is optional only when a clock is unavailable.
`SchemaVersion`, `NumSamples`, `CoordinateSystem` and `FreeMomentFrame` must be
consistent. `Corners` is `(3, 4, n)` and `Origin` is `(3, 1)`.

Regenerate older HDF5 files using `FileConverter.c3d_to_h5()`. Files with scalar
`Tz`, lowercase geometry, incompatible layouts or missing schema information
are rejected rather than guessed or migrated. Entirely unknown geometry is
stored as NaN, including orientation; it is never replaced with a zero matrix
or a claimed identity pose.

### 2. Force processing

`downsample()` accepts positive integer factors, excluding booleans. It checks
shapes, sampling rate, uniform/increasing timestamps and rotation validity
before computing outputs. Force, moment, CoP and `Tz` use FIR decimation;
corners, position, orientations and time use matching original indices. Factor
1 is a validated no-op. A missing rate is inferred from a supplied uniform
clock; with no rate or clock, time remains unknown.

Low/high-pass filters leave geometry unchanged. All force processing validates
before changing arrays; filters and downsampling compute results before
assignment. Static geometry can be supplied once to `ForceData` and is expanded
to the sample count. Missing signals initialize as `(3, n)` zeros. Rotations
must be proper orthonormal matrices or explicitly unknown. The C3D corner helper
now constructs orthogonal axes even with slightly skewed measured corners.

### 3. Processed HDF5 metadata

Marker units, residuals, camera masks, virtual status, per-sample type codes,
source frame offsets and clocks are retained. Crop slices the measurement
metadata too. Unknown residuals are NaN; absent camera information is tracked
explicitly. Force groups refresh units, rates, counts and coordinate metadata.
Analog/EMG groups persist per-channel units and identifiers.

Saving validates shared shapes/clocks/rates, writes to a temporary file, then
atomically replaces the destination. Same-path saves are supported. Removed
loaded channels disappear; collections intentionally skipped during loading
are preserved. If a retained clock changes, opaque events, rigid bodies and
unlabeled trajectories move under `SourceData` with an explicit original-recording
scope. They are not presented as aligned annotations for the processed trial.
IK/ID column metadata remains priority 2 work (issues 9 and 11).

### 4. Vector free moments and exports

`Tz` is the **moment-at-CoP vector**, not a scalar Z value or the moment at the
plate origin. It is expressed in the same declared frame as force and CoP.
Rotation, masking, filtering, cropping, downsampling and unit conversion operate
on all components. MOT exports all three and defaults to the first plate's
actual clock, checking alignment before opening the destination.

### 5. Processed C3D ownership and writing

`handler.load_data()` returns the handler's shared processed `TrialData`.
Repeated loads rebuild channels; duplicate labels receive deterministic suffixes.
`slice_c3d(start, end)` crops the shared trial (end inclusive), retaining frame
offsets, validity and in-range events when written.

`handler.write_c3d(path, trial)` writes processed marker and analog containers;
omitting `trial` uses the shared one. C3D force platforms are reconstructed from
calibrated analog channels. Direct edits to derived force vectors/geometry,
force resampling, or point-unit changes with existing force platforms are
rejected before writing; save those processed results as HDF5 or MOT. Separate
EMG/IK/ID result collections also require HDF5. Valid source-aligned crops and
analog channel reordering are supported, with plate-channel references updated.

`handler.write_raw_c3d(path)` explicitly writes the original raw structure.
C3D has no unknown-residual state: missing residual information is written as
invalid (`-1`), while source residuals/camera masks are preserved. HDF5 retains
unknown residuals as NaN. There is no arbitrary inverse-calibration writer for
independently processed force vectors.

## Priority 2: make common workflows predictable

### 7. Finish missing-data handling

`clean_nan()` remains an explicit preprocessing step; filters do not call it.
Leading/trailing NaNs remain missing and still contaminate the entire filtered
axis; all-NaN data raises `ValueError`. There is no maximum interior gap length,
and `_gap_fill(method=...)` ignores `method`.

Add a maximum gap duration, preserve the original validity mask, reject or
segment unresolved gaps before filtering, and handle all-missing channels with
a clear result/error. Interpolation should use actual time when it is not
uniform. Test short/long, boundary, and all-missing gaps. C3D residuals now survive loading, cropping and saving; integrate their
validity information into this gap-filling policy.

Locations: `containers/markerData.py`, `utils/utils.py` (`apply_filter`).

### 8. Preserve time, units, and cache validity across derived channels

Marker addition and scalar/marker division now preserve timestamps and source
frame offsets. Addition/marker division reject mismatched supplied clocks and
units. A missing clock on one operand is still accepted. `parse_EMG_data()`
preserves analog time but shares the signal array. EMG filtering or assignment
after reading `processed_data` leaves its cache stale. Empty/single-sample
`Data` time vectors still raise indexing errors; repeated/decreasing clocks
are not consistently rejected across all container types.

Create shared shape/time/rate validation, preserve times in derived containers,
copy analog arrays when constructing EMG, and invalidate cached envelopes on
supported mutations. Since raw arrays are publicly writable, either avoid the
cache or define an explicit recomputation contract. Handle tiny time vectors
without attempting to infer an unavailable rate.

Locations: `containers/markerData.py`, `trialData.py`, `emgData.py`, `data.py`.

### 9. Finish HDF5 result replacement semantics

The priority 1 implementation also fixed atomic/same-path saves, metadata
updates, selective loading, subject loading, missing optional time datasets and
removal of marker/analog/EMG/force channels. Intentionally unloaded collections
are preserved; loaded collections replace their serialized counterparts.

IK/ID savers still merge existing column labels, retain removed result groups,
and do not preserve all per-column unit metadata. Extend replacement semantics
and column validation to results together with issue 11. Define a public way
to opt an initially skipped collection into a subsequent save.

Location: `handlers/h5Handler.py` (result saving).

### 10. Fix convenience methods and array behavior

- `TrialData.as_df(forces)` now uses valid `cop_x`/`cop_y` accessors. Still add
  `cop_z`, timestamps and alignment checks.
- The `time_normalize` argument to `as_df()` is ignored. Implement it with clear
  semantics or remove it from the API.
- `Subject.lowpass_filter()` calls a removed method and catches every resulting
  error. Dispatch to current per-type filters and return structured failures.
- Adding the first channel does not initialize a trial's cached rate. Derive
  labels/rates from current data or update and validate every mutation.
- `ForceData.__setitem__()` writes to a temporary stack. Implement write-through
  indexing or remove mutable indexing; document the axis meaning of `len()`.
- `__array__()` lacks `dtype`/`copy` support. Support the NumPy protocol or use
  explicit array accessors; cover `np.asarray(obj, dtype=float)`.

Locations: `containers/trialData.py`, `subject.py`, `_mixins.py`, `analogData.py`,
`forceData.py`.

### 11. Give result columns their own physical units and clocks

The shared `MotResults.read()` applies angular units to ID forces/moments and
IK translations. IK skips only the three named pelvis translations when
converting angles. `add_column()` does not default to the parent time vector;
`read()` does not clear columns left over from an earlier file.

Keep generic storage parsing unit-neutral. Let IK use known coordinate types
(or caller-supplied mappings) and ID use explicit per-column units. Preserve
these through HDF5. Inherit/validate the parent clock for added columns and
replace the column mapping on read. Make malformed headers and unequal row
lengths clear errors; support valid whitespace-separated storage files if those
are part of the supported input contract.

Locations: `containers/motResults.py`, `IKResults.py`, `IDResults.py`,
`handlers/h5Handler.py`, `utils/utils.py` (`read_storage`).

### 12. Make gait-event boundaries and frame assumptions explicit

A force above threshold at sample zero skips every later contact on that plate.
No-contact results contain `None`, which crashes the stance helper. Multiple
peaks can describe the same contact; incomplete contacts yield NaNs that are
cast to invalid marker indices. The detector assumes Z-up coordinates, matching
force/analog rates, and aligned time origins. Step lengths use hardcoded toe
names and the X direction.

Detect threshold-based contact intervals first, then choose/refine events within
each interval. Return typed empty events and flag incomplete contacts. Use actual
force/marker timestamps for correspondence, configurable axes/marker labels, and
explicit task-specific step-length definitions. Test initial loading followed by
a later contact, double-peaked stance, no contact, final incomplete contact,
unequal rates, and shifted clocks.

Location: `analysis/gaitAnalyzer.py`.

### 13. Remove conversion and OpenSim orchestration traps

Marker-only C3D → HDF5 now writes valid empty analog groups. Convenience C3D converters
share `.temp_conversion.h5`, so concurrent jobs collide and exceptions leave
files behind. They also perform unnecessary disk round-trips.

Export directly from a loaded trial where possible. Otherwise use unique
context-managed temporary directories. Add `try/finally` cleanup for OpenSim
logging and temporary files, check tool success/output existence, and validate
requested time ranges against all inputs. Resolve setup-relative paths against
the setup file. Keep the two-plate assumptions in `build_extloads()` explicit,
or accept a mapping of plate columns to model bodies.

Locations: `biomech_io/file_converter.py`, `handlers/osimHandler.py`,
`utils/utils.py` (`build_extloads`).

## Priority 3: release checks and usability

The automated suite gives **167 passed, 1 expected failure**, with no unexpected
failures or setup errors. The expected failure is the existing subject filtering
bug (issue 10). All priority 1 checks are ordinary passing tests. The previous
marker-division timing failures also pass because virtual markers now retain
their source clock/frame information.

Run tests before the PyPI publish job and test installation of the built wheel.
Fixtures generate small real C3D files and temporary HDF5/MOT files; no ignored
private recordings are needed for the suite. Add plotting/test extras, a license
file and tested dependency ranges. The lockfile records 0.3.2, matching
`pyproject.toml`. Sphinx builds in the documentation deployment workflow;
add runnable examples and pull-request checks. OpenSim execution needs a
separate optional integration job with a validated model.

## Verification and file-specific undo

```bash
.venv/bin/python -m pytest tests -q -rx
```

Regression coverage is in `tests/test_priority1_review.py` and
`tests/test_priority1_safety.py`. It includes current-schema rejection, clocks,
proper rotations, filter atomicity, two-save metadata round trips, marker-only
conversion, failed-save destination preservation, selective saves, C3D processed
versus raw writes, force-edit rejection, source frame offsets, events, point
validity, channel reordering and duplicate labels.

Separate local checks used `example_data/test_c3d.c3d` (68 markers, five plates)
and `example_data/06_PRE_GANG_12_15.c3d` (42 markers, two plates) for converted-unit
HDF5 round trips, MOT export and processed-marker C3D round trips. Original
recordings and existing output files were not overwritten. Python 3.12.2 and
pytest 9.1.1 were used. OpenSim execution and a full priority 2/3 audit remain
outside this implementation.

See [implementation details and per-file undo](priority1-changes.md). Saved
pre-implementation copies include the user's earlier uncommitted work; each
restore command accepts exactly one file and protects newer edits.
