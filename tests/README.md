# ibo-biomech container characterization tests

Golden-behavior tests for `ibo_biomech.containers`, written **before** the
container consolidation refactor so any behavior change during the refactor
shows up as a failing test instead of a silent regression.

## Scope

Covers: `Data`, `MarkerData`, `ForceData`, `AnalogData`, `EMGData`,
`IKResults`, `IDResults`, `TrialData`, `Subject`.

**Not covered** (needs real fixture files, not synthetic data):
- `handlers/` (`C3DHandler`, `H5Handler`, `osimHandler`) — needs sample
  `.c3d`/`.h5` files, e.g. from `example_data/`.
- `biomech_io/file_converter.py` (`FileConverter`) — same.
- `analysis/gaitAnalyzer.py` — untouched by this refactor scope anyway.

Add a `test_handlers.py` once real fixture files are available (worth
copying 1-2 small anonymized trials into `tests/data/` for this).

## Install & run

```bash
pip install pytest scipy pandas numpy ibo-biomech
pytest -v
```

Or against a local checkout instead of the PyPI release:
```bash
pip install -e /path/to/ibo-biomech
pytest -v
```

## What's deliberately pinned down (read before refactoring)

These are the specific quirks discussed alongside the consolidation plan —
each has a comment in the relevant test explaining *why* it's asserted this
way, not just what:

- **`MarkerData.rotate` / `ForceData.rotate` sign convention** — a +90deg
  rotation about X maps `(y, z) -> (-z, y)`. Verified empirically against
  `get_rotation_matrix`, not assumed.
- **`EMGData._highpass_filter`** (private) returns a filtered copy without
  mutating `self.data`; the public `highpass_filter` mutates in place. These
  must stay behaviorally distinct if `AnalogData`/`EMGData` get merged.
- **`EMGData.crop`** re-slices the cached `_processed_data` envelope using
  raw-signal indices. `test_emg_crop_invalidates_processed_cache` locks in
  today's behavior but flags that "invalidate + lazily recompute" would be
  the more correct fix, if you want to make that call during the refactor.
- **`AnalogData.get_data()` vs `EMGData.get_raw_data()`** — same job,
  different name. `test_analog_has_no_get_raw_data` / `test_emg_has_get_raw_data_not_get_data`
  pin down today's asymmetry so a merge doesn't accidentally drop one.
- **`IKResults.write()` emits `inDegrees=`; `IDResults.write()` does not.**
  Both writers are otherwise near-identical — this is the one real
  difference to preserve when collapsing them into a shared `MotResults`.
- **`IKResults.to_deg`/`to_rad`** skip `time`, `pelvis_tx`, `pelvis_ty`,
  `pelvis_tz` — translations, not angles.
- **`IKResults`/`IDResults.__getitem__`** — added upstream after the PyPI
  version this suite was built against, so those two tests currently
  *skip* rather than fail. The tests assume it's keyed by column name
  (`ik["hip_flexion_r"]` -> the `Data` object) — **please confirm that
  assumption**, or correct `test_getitem_by_column_name_if_implemented` in
  `test_mot_results.py` if the real signature is different (e.g. row-index
  based instead).

## Bug found while writing this suite (independent of the refactor)

`TrialData.marker_labels` / `analog_labels` / `marker_rate` / `analog_rate`
are computed once in `__post_init__` and never refreshed by
`add_marker()`/`add_analog()`. Since your own README's "add a virtual
marker" example calls `trial.add_marker(mid)` on an already-constructed
`TrialData`, that marker silently never shows up in `marker_labels`, and
`marker_rate` stays `None` if the trial started empty. Handlers dodge this
because they pass the full `markers=` dict into the constructor directly.
Locked in as `test_KNOWN_BUG_add_marker_does_not_refresh_cached_labels` in
`test_trial_data.py` — worth fixing (have `add_marker`/`add_analog` refresh
the cached list/rate) either alongside the refactor or before it, since
it's a correctness bug independent of the consolidation.
