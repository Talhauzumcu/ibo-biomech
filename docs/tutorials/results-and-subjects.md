# Work with results and subjects

`IKResults` and `IDResults` share the `MotResults` base for reading, filtering,
cropping, writing, and named-column access. Each column is a `Data` object.
The first example is self-contained and writes a small result file.

## Create, convert, and reload IK results

```python
from pathlib import Path
import numpy as np
from ibo_biomech import Data, IKResults

output = Path("output")
output.mkdir(exist_ok=True)
time = np.arange(300) / 100.0
hip = Data(
    name="hip_flexion_r", data=20.0 + 10.0 * np.sin(2 * np.pi * time),
    time=time, unit="deg",
)
ik = IKResults(
    name="demo", time=time, data={hip.name: hip}, unit="deg",
    metadata={"inDegrees": "yes"},
)
ik.add_column("pelvis_tx", data=0.5 * time, unit="m", time=ik.time)
ik.to_rad()  # pelvis_tx/pelvis_ty/pelvis_tz are excluded from angle conversion
ik.lowpass_filter(cutoff=6.0)
ik.crop(start_idx=50, end_idx=200)
ik.write(str(output / "demo_IK.mot"))

loaded = IKResults(filepath=str(output / "demo_IK.mot"))
assert loaded.unit == "rad"
assert np.allclose(loaded["hip_flexion_r"].data, ik["hip_flexion_r"].data)
print(loaded.columns)
```

For at least two uniformly spaced timestamps, `Data` derives its sampling rate
from `time`, or creates time from `sampling_rate` if time was omitted. Pass
`time=ik.time` when adding a column so it has the result clock for filtering.

The reader initializes column units from the file's `inDegrees` flag. Set
physical units from your model before converting mixed angle/translation
results; `to_rad()` and `to_deg()` skip columns whose unit is neither `deg` nor
`rad`. The three named pelvis translations are also excluded. This example's
translation is in metres:

```python
loaded["pelvis_tx"].unit = "m"
```

## Read ID results and make a table

This file-based example requires an OpenSim inverse-dynamics output. Use column
names present in your model.

```python
import pandas as pd
from ibo_biomech import IDResults

id_results = IDResults(filepath="walking_ID.sto")
print(id_results.columns)

# Set units from the model/output definition, including any prior normalization.
# These example assignments assume unnormalized OpenSim generalized forces.
units = {"hip_flexion_r_moment": "Nm", "pelvis_tx_force": "N"}
for name, unit in units.items():
    if name in id_results:
        id_results[name].unit = unit

table = pd.DataFrame({"time": id_results.time})
for name, column in id_results.data.items():
    table[name] = column.data
table.to_csv(output / "walking_ID.csv", index=False)
```

Set physical units on individual ID columns from the output/model definition,
particularly when a result mixes forces, moments, and normalized values. The
container-level unit initialized from `inDegrees` does not describe those
quantities.

## Normalize a selected interval

`time_normalize()` linearly interpolates a one-dimensional, equally spaced
signal. It returns a percentage axis and a new signal; it does not detect gait
cycles or modify the source.

```python
from ibo_biomech.utils.utils import time_normalize

percent, normalized_hip = time_normalize(
    loaded["hip_flexion_r"].data, num_points=101,
)
assert percent[0] == 0 and percent[-1] == 100
assert normalized_hip.shape == (101,)
```

Here the selected interval is the cropped demonstration segment, not a detected
gait cycle. For gait comparisons, first select a complete cycle using verified
events. The returned percentage axis is not a time vector in seconds.

## Attach results and organize multiple trials

```python
from ibo_biomech import MarkerData, TrialData, Subject

subject = Subject(id="P01", body_mass=70.0)
for number in range(2):
    marker = MarkerData(
        name="R_Ankle", x=np.ones(100) * number,
        y=np.zeros(100), z=np.zeros(100), unit="m", sampling_rate=100.0,
    )
    trial = TrialData(
        name=f"walk_{number + 1:02d}", markers={marker.name: marker},
        metadata={"Project": {"Condition": "walking"}},
    )
    subject.add_trial(trial.name, trial)

# Attaching a result does not align/crop the trial's other channels.
subject.get_trial_by_idx(0).attach_IK_results(str(output / "demo_IK.mot"))
# With your ID file:
# subject.get_trial_by_idx(0).attach_ID_results("walking_ID.sto")
```

Filter using the trial methods in an explicit loop:

```python
frames = []
for name, trial in subject.trials.items():
    trial.lowpass_filter_markers(cutoff_freq=6.0)
    first_marker = next(iter(trial.markers.values()))
    if any(not np.array_equal(marker.time, first_marker.time)
           for marker in trial.markers.values()):
        raise ValueError("Markers must share timestamps for this table")
    frame = trial.as_df(trial.markers)
    frame.insert(0, "time", first_marker.time)
    frame["trial_name"] = name
    frame["subject_id"] = subject.id
    frames.append(frame)
combined = pd.concat(frames, ignore_index=True)
combined.to_csv(output / "markers.csv", index=False)
```

`trial.as_df()` includes scalar fields from `trial.metadata["Project"]`.
Both it and `subject.as_df("markers")` combine samples by array position; check
timestamps first and add the time column explicitly. To normalize an interval,
call `time_normalize()` before creating a table.

`trial.as_df(trial.forces)` includes all force and plate-origin moment
components, plus `cop_x` and `cop_y`. For all three CoP components and the
moment-at-CoP vector, build a table from the arrays explicitly:

```python
from ibo_biomech import ForceData

plate = ForceData(
    name="forceplate_0", force=np.zeros((3, time.size)),
    sampling_rate=100.0, time=time,
    metadata={"unit_force": "N", "unit_moment": "Nm", "unit_position": "m"},
)
force_table = pd.DataFrame({"time": plate.time})
for axis, force, cop, moment in zip("xyz", plate.force, plate.cop, plate.Tz):
    force_table[f"force_{axis}"] = force
    force_table[f"cop_{axis}"] = cop
    force_table[f"moment_at_cop_{axis}"] = moment
force_table.to_csv(output / "forces.csv", index=False)
```

`subject.save_cache(cache_dir="output/cache")` writes a local pickle;
`Subject.load_from_cache("output/cache/P01_cache.pkl")` reads it back. Use caches
from your own trusted workflow and retain source recordings for reproducibility.
