# Remaining issues and proposed fixes

Follow-up review of the working tree on 2026-09-23, including the local changes
to markers, forces, HDF5, IK results, and OpenSim scaling. This document records
implementation work still to do; the documentation update does not change the
library's behavior.

## Fixes verified in this review

- HDF5 marker loading now reads the stored `Unit` attribute.
- Force saving now uses `corners`/`origin` rather than the removed attributes;
  saving a valid C3D-derived trial preserved both arrays on reload. The
  converter/reader schema mismatch below remains.
- IK degree/radian conversion updates `metadata["inDegrees"]`; a radians HDF5
  save/reload preserved both the value and its unit in an isolated marker trial.
- Force orientation rotation now matches frame-by-frame matrix multiplication;
  plate corners are also rotated.
- Force unit conversion now scales position, corners, and local origin, and
  updates the in-memory moment/position unit metadata.
- Filtering and low-force masking now include `Tz` when its shape is valid.
- Explicit `MarkerData.clean_nan()` fills an interior gap successfully.
- Scaling now calls `setSubjectMass` and `setSubjectHeight` (verified by source
  inspection; OpenSim execution was not available).

## Priority 1: complete before relying on general processing/export

### 1. Unify the HDF5 force schema across all three paths

**Remaining behavior:** `FileConverter.c3d_to_h5()` writes lowercase `corners`
and `origin`. `H5Handler._load_forces()` reads `Corners` and `Origin`. Missing
corners default to `(4, 3, n)`, whereas `ForceData` expects `(3, 4, n)`. Geometry
is silently lost, then `rotate_forces()` raises a matrix-dimension error.
The saver now uses `Corners`/`Origin`, so its output can contain both the old
lowercase and new uppercase datasets. The loader also ignores `CoordinateSystem`.

**How to handle it:** choose one versioned schema, use shared field definitions
in reader/writer/converter, normalize supported legacy names and layouts on
input, preserve coordinate-system metadata, and validate shape before creating
a plate. Consolidate the saved geometry datasets and refresh all attributes for
existing as well as new groups. Avoid silently replacing present-but-unrecognized
geometry with zeros.

**Verification:** C3D → HDF5 → TrialData → HDF5 → TrialData must preserve every
force and geometry array, units, time, and frame metadata. Cover a legacy lab
file and a freshly converted file. Confirm MOT conversion succeeds afterwards.

Locations: `handlers/h5Handler.py` (`_load_forces`, `_save_forces`),
`biomech_io/file_converter.py` (`c3d_to_h5`).

### 2. Resample timestamps and rotations according to their meaning

**Remaining behavior:** `ForceData.downsample()` now decimates every array,
including `time` and rotation matrices. This fixes lengths but filters the
clock and destroys rotation orthogonality. On the example recording with a
5-second offset, time started near 3.126 s rather than 5 s and became
nonmonotonic. The first downsampled rotation had a maximum `R.T @ R - I` error
of about 0.609.

**How to handle it:** validate a positive integer factor and the input clock;
anti-alias force, moment, CoP, and free-moment signals. For a uniform clock,
use matching original timestamps (`old_time[::factor]`) or regenerate from the
preserved start time and new sampling rate. Preserve static geometry; sample
orientations at those indices rather than FIR-filtering their matrix entries.
If interpolating moving orientations becomes necessary, use a rotation-aware
interpolator. Compute and validate results before mutating the container.

SciPy's [decimate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html)
applies an anti-aliasing filter; that filtering explains why it is unsuitable
for a timestamp vector or matrix-valued orientation.

**Verification:** nonzero start times, factors that do not divide the sample
count, monotonic/uniform output time, equal channel lengths, and orthonormal
orientations with determinant +1. Compare exported MOT times to expected times.

Location: `containers/forceData.py` (`downsample`).

### 3. Persist units and sample metadata after processing

**Remaining behavior:** `_save_markers()` still does not update `Unit`.
Converting 1,000 mm to 1 m, saving, then reloading returned `1.0 mm` in a
marker-only reproduction. Existing force groups also retain their old unit
attributes. Marker `NumLabeled`, frame limits, residuals, and measured/virtual
status can become inconsistent after adding or cropping markers.

**How to handle it:** derive serialized units from current containers, validate
consistent units within each shared dataset, and update all associated counts
and time/frame information. Preserve/crop residual and validity data where
available; explicitly mark information that cannot be preserved. Round-trip
analog/EMG units and channel identifiers as well. Define how retained events
and rigid bodies relate to a cropped trial.

**Verification:** unchanged and converted units, cropped trials, virtual-marker
addition, channel removal, and a second save/reload must preserve meaning as
well as numeric values.

Location: `handlers/h5Handler.py` (all `_save_*` methods).

### 4. Represent the free moment as a vector in a declared frame

**Remaining behavior:** C3D parsing keeps only `plate['Tz'][2, :]`, and the MOT
writer always emits `(0, Tz, 0)`. Filtering/masking the scalar is now fixed, but
arbitrary coordinate rotations and tilted force plates are still unsupported.

**How to handle it:** retain the full moment-at-CoP vector from ezc3d, rotate it
with the force/CoP frame, and export its three components. Keep any scalar `Tz`
compatibility accessor explicitly tied to a known axis. Do not substitute
moments about the plate origin for moments about the CoP.
The [ezc3d reference](https://github.com/pyomeca/ezc3d#force-platform-filter-2)
distinguishes those quantities.

**Verification:** default -90-degree X conversion, identity transform, another
rotation, an inclined plate, and unloaded samples.

Locations: `handlers/c3dHandler.py`, `containers/forceData.py`,
`utils/utils.py` (`write_mot`).

### 5. Keep geometry separate from signal filtering

**Remaining behavior:** force filters now process `position`; high-pass
filtering a static position removes it. Corners and orientations are left
unchanged, producing inconsistent geometry. A normally constructed force with
100 signal samples but default one-sample `Tz`/position fails filtering after
some signal arrays have already been changed.

**How to handle it:** initialize absent time-varying signals with the correct
length; distinguish static geometry from moving-plate signals. Force filtering
should have explicit targets and leave geometry alone by default. Validate all
inputs before assigning any filtered arrays.

**Verification:** minimal valid plate construction, fully populated C3D plates,
static geometry preservation, and unchanged objects after rejected operations.

Location: `containers/forceData.py` (`__post_init__`, filters).

### 6. Make processed data ownership and C3D writing explicit

**Remaining behavior:** raw C3D data, handler containers, and the returned deep
copy can diverge. Rotating a handler marker then writing C3D saved the original
coordinates in the first review; the write path is unchanged. Repeated loading
also adds duplicate analog channels.

**How to handle it:** use `TrialData` as the processed source of truth and have
writers explicitly accept it. If C3D writing remains raw-structure-only, name
and document that scope and provide a separate supported processed writer.
Clear/rebuild channel mappings on each load. Preserve source frame offsets,
events, and point-validity metadata when reconstructing a C3D.

**Verification:** process/save/reload comparisons for each supported operation,
repeated loading, slicing, and duplicate channel labels.

Location: `handlers/c3dHandler.py`.

## Priority 2: make common workflows predictable

### 7. Finish missing-data handling

`clean_nan()` remains an explicit preprocessing step; filters do not call it.
Leading/trailing NaNs remain missing and still contaminate the entire filtered
axis; all-NaN data raises `ValueError`. There is no maximum interior gap length,
and `_gap_fill(method=...)` ignores `method`.

Add a maximum gap duration, preserve the original validity mask, reject or
segment unresolved gaps before filtering, and handle all-missing channels with
a clear result/error. Interpolation should use actual time when it is not
uniform. Test short/long, boundary, and all-missing gaps. Preserve C3D residuals
in containers so invalid points can participate in this policy.

Locations: `containers/markerData.py`, `utils/utils.py` (`apply_filter`).

### 8. Preserve time, units, and cache validity across derived channels

Marker arithmetic discards the original timestamps and accepts mismatched units
or clocks. `parse_EMG_data()` shares the analog array and resets time to zero.
EMG filtering or assignment after reading `processed_data` leaves its cache
stale. Empty/single-sample `Data` time vectors raise indexing errors; repeated
or decreasing timestamps are not consistently rejected.

Create shared shape/time/rate validation, preserve times in derived containers,
copy analog arrays when constructing EMG, and invalidate cached envelopes on
supported mutations. Since raw arrays are publicly writable, either avoid the
cache or define an explicit recomputation contract. Handle tiny time vectors
without attempting to infer an unavailable rate.

Locations: `containers/markerData.py`, `trialData.py`, `emgData.py`, `data.py`.

### 9. Repair HDF5 convenience and save semantics

Disabling marker/analog loading passes `None` to dictionaries. Default metadata
editing and same-path saving call `copy2()` onto the source. `load_subject_data()`
uses nonexistent `load_trial_data()` and `trial_name`. Removed marker/analog
labels are retained and written as zeros; removed force groups are retained.
Some savers assume optional `Time` datasets exist.

Use empty dictionaries for absent loaded collections, and separately track
which collections were intentionally not loaded so saving cannot erase them.
Use `load_data()` and `trial.name` for subjects. Define replacement versus merge
semantics. Validate the whole trial before writing, create missing optional
datasets, and write through a temporary file followed by replacement so a failed
save cannot leave a partially updated destination.

Location: `handlers/h5Handler.py`.

### 10. Fix convenience methods and array behavior

- `TrialData.as_df(forces)` uses nonexistent `CoPx`/`CoPy`. Use `cop_x`, `cop_y`,
  and `cop_z`; include timestamps and check alignment.
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

Marker-only C3D → HDF5 writes `None` HDF5 attributes. Convenience C3D converters
share `.temp_conversion.h5`, so concurrent jobs collide and exceptions leave
files behind. They also perform unnecessary disk round-trips.

Represent absent analog data explicitly without unsupported attributes, and
export directly from a loaded trial where possible. Otherwise use unique
context-managed temporary directories. Add `try/finally` cleanup for OpenSim
logging and temporary files, check tool success/output existence, and validate
requested time ranges against all inputs. Resolve setup-relative paths against
the setup file. Keep the two-plate assumptions in `build_extloads()` explicit,
or accept a mapping of plate columns to model bodies.

Locations: `biomech_io/file_converter.py`, `handlers/osimHandler.py`,
`utils/utils.py` (`build_extloads`).

## Priority 3: release checks and usability

The current suite still gives **68 passed, 2 failed, 23 setup errors**. Its force
fixtures and assertions use `location`/`offset`. Update them to the intended
public fields, then add the round-trip and semantic checks above: checking only
array lengths would miss the new timestamp and orientation corruption.
The subject-filter test currently passes while all trials fail to filter.
Assert successful processing as well as expected failure reporting.

Run tests before the PyPI publish job and test installation of the built wheel.
Include small synthetic or distributable fixtures; most local example recordings
are ignored by Git. Add plotting/test extras, a license file, tested dependency
ranges, and refresh the lockfile (it still records project version 0.2.9).
Run runnable documentation examples and build Sphinx in CI. OpenSim tests should
be a separate optional integration job using a small validated model.

For the next feature work, prioritize a public crop-by-time operation,
resampling/alignment, recording validation reports, events/cycle segmentation,
and processing provenance. Keep one validated institutional end-to-end workflow
as the release acceptance case.

## Review verification

The follow-up used Python 3.12.2, the existing tests, the local C3D example,
synthetic marker-only HDF5 files, and numerical checks for timestamp and rotation
invariants. OpenSim execution was unavailable. Documentation examples are checked
separately from the existing unit suite; passing examples do not imply that the
remaining implementation issues above are repaired.
