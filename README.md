# ibo-biomech

`ibo-biomech` is a Python library for processing, managing, and analyzing biomechanical data. It provides straightforward interfaces to handle standard motion capture formats (C3D, OpenSim formats, and lab-custom HDF5), filter data, calculate joint centers, and run OpenSim pipelines directly from Python.

## Core Features

- **Format Conversion & File Handling**
  - Read and parse C3D files (powered by `ezc3d`).
  - Read and write lab-specific HDF5 files while preserving metadata.
  - Export data to OpenSim `.trc` (marker trajectories) and `.mot` (force plate data) files.
  - Direct conversion tools (e.g., C3D to H5).
- **Data Structures**
  - Clean and typed container classes: `Subject`, `TrialData`, `MarkerData`, `ForceData`, and `AnalogData`.
  - Convenient caching using Python's `pickle` to save and load stateful `Subject` datasets quickly.
- **Signal Processing & Transformations**
  - Butterworth low-pass filtering for markers, analogs, and force plates.
  - Signal cropping and slicing.
  - Easy rotation of 3D data (markers and force plate properties) to match different coordinate systems.
  - Automatic conversion to meters for OpenSim preparation.
  - Calculation of joint centers and addition of virtual markers (e.g., virtual C3D preparation for scaling).
- **OpenSim Integration**
  - Wrapper tools to natively run OpenSim Inverse Kinematics (`runIK`) and Scaling (`runScaling`) routines programmatically.

## Code Examples

### 1. Parsing a C3D file and applying transformations
```python
from handlers.c3dHandler import C3DHandler

# Load C3D file and parse data into memory
handler = C3DHandler("path/to/trial.c3d")
handler.load_data()

# Apply a 4th-order low-pass filter to all markers and force plates
handler.lowpass_filter_all(cutoff=15.0, order=4)

# Rotate data 90 degrees around the X-axis and export to OpenSim formats (.trc, .mot)
handler.process_and_export("output_folder/trial_rotated", axis='x', angle_deg=90, convert_to_meters=True)
```

### 2. File Conversion (C3D to HDF5)
```python
from biomech_io.file_converter import FileConverter

converter = FileConverter()
converter.c3d_to_h5("input_file.c3d", "output_file.h5")
```

### 3. Managing Subject and Trial Data
```python
from handlers.h5Handler import H5Handler
from containers.subject import Subject

# Load trial from an H5 file directly as a structured Subject
handler = H5Handler("subject_trial.h5")
subject = handler.load_subject_data()

# Process data inside the convenient TrialData container
trial = subject.get_trial_by_idx(0)
trial.crop(start_idx=100, end_idx=500)

# Filter nested analogs, markers, and forces with distinct cutoffs
trial.lowpass_filter(cutoff_marker=10, cutoff_analog=50, cutoff_force=20)

# Cache the processed subject to disk for fast retrieval later
subject.save_cache(cache_dir="cache_folder")
```

### 4. Running OpenSim Tools
```python
from handlers.osimHandler import OsimHandler

# Run OpenSim Scaling Tool
OsimHandler.runScaling(
    model_path="models/base_model.osim",
    trc_path="data/static_pose.trc",
    output="models/scaled_model.osim",
    mass=75.5
)

# Run OpenSim Inverse Kinematics (IK)
OsimHandler.runIK(
    model_path="models/scaled_model.osim",
    trc_path="data/dynamic_trial.trc",
    mot_path="data/dynamic_trial.mot",
    output="results/ik_results.mot"
)
```
