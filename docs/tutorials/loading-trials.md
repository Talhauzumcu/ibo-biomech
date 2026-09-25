# Load and organize trials

## Read a C3D recording

Use this route for markers, force plates, and raw analog channels. Replace the
path with a recording from your lab. A file with markers but no force plates
can still be loaded.

```python
from copy import deepcopy
from ibo_biomech import C3DHandler

handler = C3DHandler("walking.c3d")
raw = handler.load_data()
trial = deepcopy(raw)

print(trial.name)
print(trial.get_marker_names())
print(trial.get_force_names())
for name, analog in trial.analogs.items():
    print(name, analog.channel, analog.sampling_rate, analog.unit)

marker_name = trial.get_marker_names()[0]
marker = trial.markers[marker_name]
print(marker.get_trajectory().shape)  # (3, n_samples)
print(marker.time[:5], marker.unit)
```

Dictionary access raises `KeyError` for a missing label. `get_marker(name)`,
`get_force(name)`, `get_analog(name)`, and `get_analog_by_channel(index)` return
`None` when no match exists. C3D analog channel indices start at zero.

The returned trial shares the handler's processed containers; the explicit
`deepcopy()` above preserves a separate processing copy. Write that copy with
`handler.write_c3d("processed.c3d", trial)`. The writer supports marker and
source-analog processing and source-aligned cropping. It rejects direct edits
to derived force vectors/geometry; use HDF5/MOT for those. Separate EMG/IK/ID
results also belong in HDF5. `handler.write_raw_c3d()` explicitly writes the
original raw structure. Repeated loading rebuilds all channel mappings.

## Read the institute HDF5 format

This is an alternative loading route; HDF5 files must follow the institute's
`MetaData`, `Trajectories`, `Analog`, and `ForcePlates` schema.

```python
from ibo_biomech import H5Handler

h5_handler = H5Handler("walking.h5")
trial = h5_handler.load_data()
print(trial.name, trial.metadata)
print(trial.marker_labels, trial.marker_rate)
```

Only the current force schema (2) is accepted, with vector `Tz` of shape
`(3, n_samples)`. Regenerate older files using `FileConverter.c3d_to_h5()`;
there are no legacy layout or scalar-moment compatibility paths. Selective
loading uses empty collections and tracks them so later saves preserve the
intentionally skipped data.

## Add metadata without processing arrays

`modify_metadata()` returns a new handler and writes atomically. Omit
`out_path` to update the source, or supply a new destination.

```python
from pathlib import Path

Path("output").mkdir(exist_ok=True)
annotated = h5_handler.modify_metadata(
    {"SubjectID": "P01", "Condition": "walking", "BodyMass": 70.0},
    out_path="output/walking_annotated.h5",
)
annotated_trial = annotated.load_data()
print(annotated_trial.metadata["SubjectID"])
```

Assigning values to `trial.metadata` only changes the in-memory dictionary;
`save_data()` does not persist those edits. Use `modify_metadata()` for file
attributes.

## Save processed HDF5 data

The public signature is `h5_handler.save_data(trial, out_path="output/processed.h5")`.
It uses the original file as a template and replaces selected datasets.

For a C3D recording with analog channels, the following route supplies geometry
directly from the C3D containers and preserves the original units and sample
counts. It filters the force data, saves it, and checks the reloaded values:

```python
import numpy as np
from ibo_biomech import FileConverter

source_trial = C3DHandler("walking.c3d").load_data()
FileConverter.c3d_to_h5("walking.c3d", "output/walking_template.h5")
template = H5Handler("output/walking_template.h5")
source_trial.lowpass_filter_forces(cutoff_freq=20.0)
template.save_data(source_trial, out_path="output/walking_filtered.h5")
reloaded = H5Handler("output/walking_filtered.h5").load_data()
for name, plate in source_trial.forces.items():
    assert np.allclose(reloaded.forces[name].force, plate.force)
    assert np.allclose(reloaded.forces[name].corners, plate.corners)
```

Saving validates shared clocks, units and array shapes before replacing the
destination atomically; same-path saves are supported. Loaded channel collections
replace their stored counterparts, including channel removals. Marker residuals,
camera visibility, virtual status and source frame offsets are preserved/cropped.
Analog/EMG units and channel identifiers are retained.

Rigid bodies in `trial.rigid_bodies` are saved and loaded automatically:

```python
from ibo_biomech import RigidBody

trial.add_rigid_body(RigidBody(
    "pelvis", markers=["LASI", "RASI"],
    position=np.zeros((3, 100)), rotation=np.eye(3), sampling_rate=100,
))
h5_handler.save_data(trial, out_path="output/with_bodies.h5")
loaded = H5Handler("output/with_bodies.h5").load_data()
pelvis = loaded.rigid_bodies["pelvis"]
```

`RigidBodies` schema version 1 stores a numbered group per body. Each group has
`Position`, `Rotation`, UTF-8 `Markers`, and optional `Time` datasets, plus
`Name`, `NumSamples`, `Unit`, and optional `SamplingFrequency` attributes.
Bodies can have independent sample counts and clocks. `RPY` is recalculated
from `Rotation` on access. Removing a loaded body removes it from the saved file.
Use `load_data(load_rigid_bodies=False)` to skip this collection and preserve it
unchanged during a subsequent save.

Unversioned legacy rigid-body payloads remain opaque and are preserved. Adding
typed bodies to such a file moves the legacy payload to `SourceData/RigidBodies`.
When clocks change, opaque legacy events, legacy rigid bodies and unlabeled trajectories move
under `SourceData`, explicitly scoped to the original recording. They are not
silently presented as aligned annotations for the cropped trial. IK/ID column
units and result-removal semantics still have the limitations listed in
[remaining issues](../remaining-issues.md).

## Read, edit and save events

`Event` represents one event with `name` (string), `frame` (integer), and
`time` (seconds). `TrialData.events` is an ordered list because labels such as
`Foot Strike` may occur repeatedly. The `description` field defaults to an empty
string. These four fields are the complete event representation.

```python
from ibo_biomech import C3DHandler, Event

handler = C3DHandler("example_data/test_c3d_events.c3d")
trial = handler.load_data()
for event in trial.events:
    print(event.name, event.description, event.frame, event.time)

strikes = trial.get_events("Foot Strike")
trial.add_event(Event(name="contact", frame=180, time=180 / trial.marker_rate))
handler.write_c3d("output/with_events.c3d")
```

Frames use the same zero-based **source** numbering as `MarkerData.first_frame`.
They are not indices relative to a cropped trajectory. At point rate `rate`,
the nearest source frame is `first_frame + round((event.time - first_time) * rate)`;
time retains sub-frame precision. This simplifies to `round(event.time * rate)`
only on the native C3D clock, where `first_time = first_frame / rate`.
For example, frame 121 in a trajectory starting at frame 88 has array index 33.
To create an event at marker array index 100, use
`Event("contact", marker.first_frame + 100, float(marker.time[100]))`.
C3D EVENT times are decoded as `minutes * 60 + seconds`. Legacy header-only
event times are offset from the first stored frame onto the trial clock.

`FileConverter.c3d_to_h5`, `H5Handler.save_data`, and `H5Handler.load_data`
preserve events, duplicates, annotations, and insertion order. HDF5 event schema
1 uses only `Name`, `Description`, `Frame`, and `Time` datasets under `Events`.
Extra event fields in older files are ignored and omitted on saving loaded events.
Older converter files with
`Time` and `LABELS` are also readable when the trajectory sampling rate is known.
`load_data(load_events=False)` skips events and preserves them on later saves.

Use `trial.crop_events(start_frame, end_frame)` to keep events in a half-open
source-frame range. Neither frames nor times are rebased. Cropping individual
signal collections does not implicitly crop events; HDF5 saves the explicit
event list. `C3DHandler.slice_c3d` crops the shared list along with the signals,
and processed C3D export includes only events in the exported frame range.
Edits, additions and removals are written by `write_c3d`; `write_raw_c3d` retains
the original recording. Subject caches include events automatically.
HDF5 preserves explicit timestamp offsets. C3D export requires the marker
clock to start at `first_frame / rate`; independently shifting timestamps
requires HDF5 to retain both the original frame numbers and the new clock.

## Organize a participant's trials

```python
from ibo_biomech import Subject

subject = Subject(id="P01", body_mass=70.0)
subject.add_trial(trial_name=trial.name, trial_data=trial)
print(subject.get_trial_by_idx(0))
print(subject.trials.keys())
```

Alternatively, `H5Handler.load_subject_data()` builds the subject from file
metadata and attaches the loaded trial. See
[results and subjects](results-and-subjects.md) for combining trial tables.
