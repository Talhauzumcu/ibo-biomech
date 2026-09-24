# Export OpenSim inputs

TRC stores marker trajectories. A ground-reaction MOT stores forces, points of
application, and torques. Exporting these files does not require the OpenSim
Python bindings.

## Export a processed C3D trial

This workflow exports the processed containers. Replace the input path and
confirm the coordinate transform before using your data. Current-format HDF5
exports use the same validated force/vector representation.

The example assumes a Z-up lab frame, forces in N, and lengths in mm or m. Its
-90-degree X rotation maps +Z to +Y. `write_mot()` exports all three components
of the rotated moment-at-CoP vector, including for tilted plates.

```python
from copy import deepcopy
from pathlib import Path
import numpy as np
from ibo_biomech import C3DHandler
from ibo_biomech.utils.utils import write_trc, write_mot

output = Path("output")
output.mkdir(exist_ok=True)
raw = C3DHandler("walking.c3d").load_data()
trial = deepcopy(raw)
if not trial.markers or not trial.forces:
    raise ValueError("This example exports both markers and forces")

# Reject unresolved marker gaps; fill only suitable gaps before this step.
for marker in trial.markers.values():
    if not np.isfinite(marker.data).all():
        raise ValueError(f"Resolve missing samples in {marker.name} first")
trial.lowpass_filter_markers(cutoff_freq=6.0)
trial.lowpass_filter_forces(cutoff_freq=20.0)
for plate in trial.forces.values():
    plate.filter_low_forces(threshold=10.0)

trial.rotate_markers(axis="x", angle_deg=-90)
trial.rotate_forces(axis="x", angle_deg=-90)
trial.convert_units("m")

first_marker = next(iter(trial.markers.values()))
first_plate = next(iter(trial.forces.values()))
for marker in trial.markers.values():
    if marker.unit != "m" or not np.array_equal(marker.time, first_marker.time):
        raise ValueError("Markers must share units and timestamps")
for plate in trial.forces.values():
    if not np.array_equal(plate.time, first_plate.time):
        raise ValueError("Force plates must share timestamps")
    if plate.unit_force != "N" or plate.unit_moment != "Nm":
        raise ValueError("MOT export requires forces in N and moments in Nm")

header = {
    "data_rate": first_marker.sampling_rate,
    "camera_rate": first_marker.sampling_rate,
    "num_frames": len(first_marker.x),
    "num_markers": len(trial.markers),
    "units": first_marker.unit,
    "orig_data_rate": first_marker.sampling_rate,
    "orig_data_start_frame": 1,
    "orig_num_frames": len(first_marker.x),
    "marker_labels": list(trial.markers),
}
write_trc(str(output / "walking.trc"), header, trial.markers, time=first_marker.time)
write_mot(str(output / "walking_grf.mot"), trial.forces, time=first_plate.time)
```

The header describes this exported segment. If preserving acquisition frame
numbers matters, retain that metadata explicitly. Markers and forces may have
different rates; their time vectors must describe the same physical time origin.
No extra rotation or unit conversion occurs inside these two writers.

## Check the exported arrays

Continue from the preceding example:

```python
from ibo_biomech.utils.utils import read_trc, read_mot

trc_time, marker_at, marker_names = read_trc(str(output / "walking.trc"))
assert np.allclose(trc_time, first_marker.time)
assert np.allclose(marker_at(first_marker.name), first_marker.data.T)

grf = read_mot(str(output / "walking_grf.mot"))
assert np.allclose(grf["time"], first_plate.time)
assert np.allclose(grf["ground_force_1_vy"], first_plate.Fy)
assert np.allclose(grf["ground_moment_1_my"], first_plate.Tz[1])
```

`read_mot()` and `read_sto()` return dictionaries of arrays plus `metadata`.
They currently expect tab-separated columns. Ground-force plate numbering in
MOT starts at **1**, in dictionary insertion order; C3D container names normally
start at `forceplate_0`.

## FileConverter methods

These methods reload a file; they do not use an existing processed `TrialData`.
The OpenSim methods accept `axis`, `angle`, and `convert_to_meters` (not
`angle_deg`). Defaults are `"x"`, `-90`, and `True`.

| Method | Positional arguments | Current scope |
| --- | --- | --- |
| `c3d_to_h5` | `c3d_path, h5_path` | Also accepts metadata keywords; preserves source coordinates and units. |
| `h5_to_trc` | `h5_path, trc_path` | Marker export. |
| `h5_to_mot` | `h5_path, mot_path` | Force and full moment-at-CoP vector export. |
| `h5_to_opensim` | `h5_path, mot_path, trc_path` | Calls both HDF5 exporters. |
| `c3d_to_trc` | `c3d_path, trc_path` | Uses an intermediate current-format HDF5. |
| `c3d_to_mot` | `c3d_path, mot_path` | Uses the current HDF5 force-export route. |
| `c3d_to_opensim` | `c3d_path, mot_path, trc_path` | Uses the current HDF5 force-export route. |

For a C3D recording with analog channels, this alternative creates HDF5 metadata
and exports its markers:

```python
from ibo_biomech import FileConverter

FileConverter.c3d_to_h5(
    "walking.c3d", str(output / "walking.h5"),
    subject_id="P01", condition="walking", body_mass=70.0,
)
FileConverter.h5_to_trc(
    str(output / "walking.h5"), str(output / "walking_from_h5.trc"),
    axis="x", angle=-90, convert_to_meters=True,
)
```

For markers already in the target frame and units, `h5_to_trc(..., angle=0,
convert_to_meters=False)` avoids applying the default transform again. Direct
C3D convenience converters share a temporary filename; do not run them
concurrently until unique temporary paths are implemented.
