# ibo-biomech

Python tools for motion-capture workflows: load C3D and the institute's HDF5
format, process markers and signals, export OpenSim inputs, and inspect IK/ID
results. Both file handlers return a `TrialData` containing named channels.

The package is in alpha. See the [current limitations and development priorities](docs/remaining-issues.md)
for the status of HDF5 saving, force resampling, and coordinate assumptions.

## Install

```bash
python -m pip install ibo-biomech
```

For this checkout, including the latest local changes:

```bash
python -m pip install -e .
```

Plotting methods require `matplotlib`, installed separately:

```bash
python -m pip install matplotlib
```

OpenSim is only required to run scaling, inverse kinematics, or inverse dynamics.
File conversion and result containers work without it. Follow the
[OpenSim Python setup instructions](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346)
and verify `import opensim` in the same Python environment.

## Try a complete example without a data file

This example creates a marker sampled at 100 Hz, filters it, crops a copy, and
exports a DataFrame with timestamps. The cutoff is illustrative; choose processing
parameters for your recording and analysis.

```python
from copy import deepcopy
import numpy as np
from ibo_biomech import MarkerData, TrialData

time = np.arange(300) / 100.0
marker = MarkerData(
    name="R_Ankle",
    x=1000.0 + 50.0 * np.sin(2 * np.pi * time),
    y=np.zeros_like(time),
    z=np.full_like(time, 100.0),
    unit="mm",
    sampling_rate=100.0,
    time=time,
)
trial = TrialData(name="demo", markers={marker.name: marker})
processed = deepcopy(trial)
processed.lowpass_filter_markers(cutoff_freq=6.0, order=4)
processed.crop("markers", start_idx=50, end_idx=200)

frame = processed.as_df(processed.markers)
frame.insert(0, "time", processed.markers["R_Ankle"].time)
print(processed.name, frame.shape)  # demo (150, 4)
# processed.markers["R_Ankle"].plot()  # requires matplotlib
```

Filtering, cropping, rotation, and unit conversion modify containers in place.
Use `deepcopy()` to keep a raw trial. Marker arithmetic returns a new marker;
`EMGData.process_emg()` returns an envelope, while also cleaning NaNs in raw data.

## Load a recording

Replace the paths with your files. HDF5 support is specific to the institute's
schema, not arbitrary `.h5` files.

```python
from ibo_biomech import C3DHandler, H5Handler

trial = C3DHandler("walking.c3d").load_data()
# Alternatively:
# trial = H5Handler("walking.h5").load_data()

print(trial.name)
print(trial.get_marker_names())
print(trial.get_force_names())
print(trial.get_analog_names())
print(trial.marker_rate, trial.analog_rate, trial.force_rate)

# Use labels present in your recording:
# marker = trial.markers["R_Ankle"]
# plate = trial.forces["forceplate_0"]
# analog = trial.get_analog_by_channel(3)
```

A C3D handler shares its processed containers with the returned trial. Use
`handler.write_c3d(path, trial)` for supported marker/analog edits, or
`handler.write_raw_c3d(path)` for the original structure. Derived force edits
belong in HDF5/MOT; see the [export tutorial](docs/tutorials/opensim-export.md).

## Current processing API

| Operation | Method | Notes |
| --- | --- | --- |
| Filter markers | `trial.lowpass_filter_markers(cutoff_freq=6.0)` | Frequency in Hz; no automatic gap filling. |
| Filter analogs | `trial.lowpass_filter_analogs(cutoff_freq=100.0)` | Cutoff must be below every affected channel's Nyquist frequency. |
| Filter forces | `trial.lowpass_filter_forces(cutoff_freq=20.0)` | Processes vector `Tz`; plate geometry stays unchanged. |
| Filter EMG channels | `trial.lowpass_filter_emgs(cutoff_freq=10.0)` | Filters raw EMG; this is not the envelope pipeline. |
| Filter one channel | `marker.lowpass_filter(cutoff=6.0)` | Channel methods use `cutoff`; trial methods use `cutoff_freq`. |
| Fill marker gaps | `marker.clean_nan()` | Linear interpolation of interior NaNs; inspect gaps first. |
| Crop one data type | `trial.crop("markers", 100, 200)` | End index excluded; indices belong to the selected data type. |
| Rotate markers | `trial.rotate_markers(axis="x", angle_deg=-90)` | Requires the correct lab-to-model transform. |
| Rotate forces | `trial.rotate_forces(axis="x", angle_deg=-90)` | Rotates vectors and geometry in the declared frame. |
| Convert units | `trial.convert_units("m")` | Supports mm ↔ m; force magnitudes remain unchanged. |
| Select EMG channels | `trial.parse_EMG_data([3])` | Channel indices are zero-based for C3D imports. |
| Attach results | `trial.attach_IK_results("ik.mot")` / `trial.attach_ID_results("id.sto")` | Results have named `Data` columns. |

There is no single trial-wide `lowpass_filter()`, `rotate_data()`, or
`convert_to_meters()` method. To crop the same interval across sampling rates,
select indices using each channel's time vector; see the processing tutorial.

## File conversion and results

`FileConverter.c3d_to_h5(c3d_path, h5_path, **metadata)` writes an HDF5 file in
acquisition coordinates and units, including marker-only recordings. Only the
current HDF5 force schema is supported; regenerate older files from C3D.

OpenSim converters are `h5_to_trc`, `h5_to_mot`, `h5_to_opensim`, `c3d_to_trc`,
`c3d_to_mot`, and `c3d_to_opensim`. Their defaults are `axis="x"`, `angle=-90`,
and `convert_to_meters=True`. Those defaults describe one lab convention.
The combined converters accept `(source_path, mot_path, trc_path)`.

The [export tutorial](docs/tutorials/opensim-export.md) shows a route from
a processed C3D trial to TRC/MOT using `write_trc()` and `write_mot()`.

```python
from ibo_biomech import IKResults, IDResults

ik = IKResults(filepath="walking_IK.mot")
ik.to_rad()
print(ik["hip_flexion_r"].data)
ik.write("walking_IK_radians.mot")

id_results = IDResults(filepath="walking_ID.sto")
print(id_results.columns)
```

Check per-column units when reading results: the current shared reader assigns
angle units to all columns, including translations and ID forces/moments. See
[results and subjects](docs/tutorials/results-and-subjects.md) for explicit units,
DataFrames, normalization, and multi-trial organization.

## Tutorials and reference

Start with the [documentation overview](docs/index.md):

- [Load and organize trials](docs/tutorials/loading-trials.md): C3D, HDF5, metadata, ownership, and saving constraints.
- [Process markers, forces, and EMG](docs/tutorials/processing.md): synthetic examples, virtual markers, and time-aware cropping.
- [Export OpenSim inputs](docs/tutorials/opensim-export.md): processed TRC/MOT output and converter signatures.
- [Work with results and subjects](docs/tutorials/results-and-subjects.md): `Data`, IK/ID, normalization, and tables.
- [Run OpenSim tools](docs/tutorials/opensim-tools.md): scaling, IK, external loads, and ID.
- [Detect gait events](docs/tutorials/gait-events.md): an executable example and the detector's assumptions.

API reference pages are in `docs/api/`. To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. To run the existing tests:

```bash
python -m pip install pytest
python -m pytest tests
```
