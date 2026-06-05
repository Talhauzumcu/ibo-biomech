# ibo-biomech

A Python library for loading, processing, and converting biomechanical motion capture data. Supports C3D files, lab-specific HDF5 files, and OpenSim formats (`.trc`, `.mot`).

> Full API documentation will be available as a generated docs site in a future release.

## Installation

```bash
pip install ibo-biomech
```

## Imports

Everything commonly used is available directly from the top-level package:

```python
from ibo_biomech import C3DHandler, H5Handler, FileConverter
from ibo_biomech import MarkerData, ForceData, AnalogData, TrialData, Subject
```

You can also import from submodules directly:

```python
from ibo_biomech.handlers import C3DHandler, H5Handler
from ibo_biomech.containers import MarkerData, ForceData, AnalogData, TrialData, Subject
from ibo_biomech.biomech_io import FileConverter
```

---

## Loading an HDF5 File

`H5Handler` reads lab HDF5 files and returns a `TrialData` object.

```python
from ibo_biomech import H5Handler

handler = H5Handler("trial.h5")
trial = handler.load_data()   # returns TrialData

print(trial.trial_name)
print(trial.marker_labels)
print(trial.analog_labels)
print(trial.marker_rate)

# Access individual channels
marker = trial.markers["L_Ankle"]
force  = trial.forces["forceplate_0"]
emg    = trial.analogs["EMG_VastusLat"]
```

### Saving processed data back to HDF5

The handler uses the original file as a template and overwrites only the data arrays, preserving all other groups (events, rigid bodies, etc.).

```python
handler.save_data(trial, out_path="trial_processed.h5")
```

---

## File Conversion

`FileConverter` provides static methods for converting between formats. No instance needed.

```python
from ibo_biomech import FileConverter

# C3D → HDF5
FileConverter.c3d_to_h5("trial.c3d", "trial.h5")

# HDF5 → OpenSim TRC (markers) and MOT (forces)
FileConverter.h5_to_trc("trial.h5", "trial.trc")
FileConverter.h5_to_mot("trial.h5", "trial.mot")
FileConverter.h5_to_opensim("trial.h5", "trial.mot", "trial.trc")

# C3D → OpenSim (direct, no intermediate file kept)
FileConverter.c3d_to_trc("trial.c3d", "trial.trc")
FileConverter.c3d_to_mot("trial.c3d", "trial.mot")
FileConverter.c3d_to_opensim("trial.c3d", "trial.mot", "trial.trc")
```

All conversion methods apply a default `-90° rotation around the X axis` and convert units to metres to match OpenSim's coordinate system. You can override these:

```python
FileConverter.h5_to_trc("trial.h5", "trial.trc", axis='x', angle=-90, convert_to_meters=True)
```
---
## Working with TrialData

`TrialData` is the central data container returned by `H5Handler.load_data()`.

```python
trial = H5Handler("trial.h5").load_data()

# Convenience accessors
print(trial.get_marker_names())
print(trial.get_force_names())
print(trial.get_analog_names())

marker = trial.get_marker("R_Knee")
fp     = trial.get_force("forceplate_0")
emg    = trial.get_analog("EMG_VastusLat")
emg2   = trial.get_analog_by_channel(3)

# Add a virtual marker
import numpy as np
mid = MarkerData(
    name="MidKnee",
    x=(trial.markers["R_Knee"].x + trial.markers["L_Knee"].x) / 2,
    y=(trial.markers["R_Knee"].y + trial.markers["L_Knee"].y) / 2,
    z=(trial.markers["R_Knee"].z + trial.markers["L_Knee"].z) / 2,
    sampling_rate=trial.marker_rate
)
trial.add_marker(mid)

# Filter all channels with separate cutoffs per data type
trial.lowpass_filter(cutoff_marker=10, cutoff_analog=500, cutoff_force=50)
```

## Signal Processing

All filtering and processing methods modify data **in-place**.

### Filtering markers

```python
marker = trial.markers["R_Knee"]
marker.lowpass_filter(cutoff=10.0, order=4)   # 4th-order Butterworth
marker.highpass_filter(cutoff=20.0, order=4)
```

### Filtering force data

```python
fp = trial.forces["forceplate_0"]
fp.lowpass_filter(cutoff=50.0, order=4)
fp.highpass_filter(cutoff=10.0, order=4)
fp.filter_low_forces(threshold=10.0)  # zero out frames where |F| < 10 N
fp.downsample(factor=4)
```

### Filtering analog signals

```python
emg = trial.analogs["EMG_VastusLat"]
emg.lowpass_filter(cutoff=500.0)
emg.highpass_filter(cutoff=20.0)
```

### Filtering everything at once via C3DHandler

```python
handler.lowpass_filter_all(cutoff=10.0, order=4)   # markers + forces
handler.lowpass_filter_markers(cutoff=10.0)
handler.lowpass_filter_force(cutoff=50.0)
```

### Cropping

```python
# Crop a single channel
trial.markers["R_Knee"].crop(start_idx=100, end_idx=500)

# Crop everything in a TrialData at once
trial.crop(start_idx=100, end_idx=500)
```

### Rotation

Rotates all position data (markers, forces, CoP) around the specified axis.

```python
trial.rotate_data(axis='x', angle_deg=-90)

# Or on an individual marker
marker.rotate(axis='x', angle_deg=90)

# Or on a TrialData (markers and forces)
trial.rotate_markers(axis='x', angle_deg=-90)
trial.rotate_forces(axis='x', angle_deg=-90)
```

### Unit conversion

```python
trial.convert_to_meters()                 # trial-level (mm or cm → m)

marker.convert_units('m')                   # single marker
trial.convert_marker_units('m')             # all markers in TrialData
trial.convert_force_units('m')              # CoP and moments in TrialData
trial.convert_units('m')                    # markers + forces together
```

### Plotting

All container types have a `.plot()` method for quick inspection:

```python
trial.markers["R_Knee"].plot()
trial.forces["forceplate_0"].plot()
trial.analogs["EMG_VastusLat"].plot()
```

---



## Loading a C3D File

`C3DHandler` reads a C3D file and parses it into typed container objects.

```python
from ibo_biomech import C3DHandler

handler = C3DHandler("trial.c3d")
handler.load_data()

# Markers: dict of {name: MarkerData}
print(handler.markers.keys())

# Force plates: dict of {"forceplate_0": ForceData, ...}
print(handler.forces.keys())

# Raw analog signals: dict of {name: AnalogData}
print(handler.analogs.keys())
```

### Accessing marker data

```python
marker = handler.markers["R_Knee"]

print(marker.x)            # np.ndarray, position along X axis
print(marker.y)
print(marker.z)
print(marker.sampling_rate)
print(marker.unit)         # e.g. 'mm'

traj = marker.get_trajectory()  # (3, n_samples) array
```

### Accessing force plate data

```python
fp = handler.forces["forceplate_0"]

print(fp.Fx)   # np.ndarray — force along X
print(fp.Fy)
print(fp.Fz)
print(fp.cop_x, fp.cop_y, fp.cop_z)  # center of pressure
print(fp.Mx, fp.My, fp.Mz)           # moments
print(fp.sampling_rate)
```

### Accessing analog signals

```python
emg = handler.analogs["EMG_VastusLat"]
print(emg.data)          # np.ndarray
print(emg.sampling_rate)
print(emg.unit)
```

### Exporting directly from C3DHandler

If you have already loaded and processed a C3DHandler (e.g. filtered or rotated), you can export without going through HDF5:

```python
handler = C3DHandler("trial.c3d")
handler.load_data()

# Rotate and export to TRC + MOT in one call
handler.process_and_export("output/trial", axis='x', angle_deg=-90, convert_to_meters=True)
# Writes: output/trial.trc and output/trial.mot
```

---

