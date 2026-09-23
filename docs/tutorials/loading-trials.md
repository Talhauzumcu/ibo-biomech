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

The returned trial is a deep copy of the handler's parsed containers. Treat it
as your processing dataset. `handler.write_c3d()` writes the raw C3D structure;
it does not synchronize arbitrary changes made to the trial or handler
containers. Use the [processed export workflow](opensim-export.md) for TRC/MOT.
Create a new handler for a fresh load: repeated loading on one handler currently
duplicates analog channels.

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

Use the default load flags for now. Disabling marker or analog loading passes
`None` into a container that expects dictionaries. Force geometry has a separate
schema mismatch: check [the remaining issues](../remaining-issues.md) before
rotating HDF5 forces or converting them to MOT.

## Add metadata without processing arrays

`modify_metadata()` returns a new handler. Supply a different destination path;
the current default tries to copy the source file onto itself.

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

General processed saving is still limited: force geometry is not consistently
loaded across schema variants, changed units are not saved consistently, and
cropping does not update residuals/events or all frame metadata. The saver also
expects some template datasets, including marker `Time`, to exist. A destination
equal to the source raises `SameFileError`.

Use TRC/MOT for transformed OpenSim inputs and DataFrames for analysis tables.
Keep the original recording. The
[remaining-issues page](../remaining-issues.md) specifies the fixes and round-trip
checks needed for general HDF5 saving.

## Organize a participant's trials

```python
from ibo_biomech import Subject

subject = Subject(id="P01", body_mass=70.0)
subject.add_trial(trial_name=trial.name, trial_data=trial)
print(subject.get_trial_by_idx(0))
print(subject.trials.keys())
```

Use this explicit construction instead of `H5Handler.load_subject_data()`, whose
implementation still calls an obsolete loader. See
[results and subjects](results-and-subjects.md) for combining trial tables.
