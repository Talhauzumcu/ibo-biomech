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

`Data` derives its sampling rate from a uniform time vector, or creates time
from `sampling_rate` if time was omitted. Explicitly pass `time=ik.time` to
`add_column()`; it currently does not inherit the result time automatically.

The reader currently assigns the file's angle unit to every column. Known
pelvis translations are excluded from angle conversion, but their read-in unit
labels still need correction. Other translational model coordinates are not
automatically recognized.

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

Do not interpret the current `IDResults.unit` as a physical force/moment unit:
the shared reader still derives it from `inDegrees`. Units belong to individual
columns, particularly when a result mixes forces, moments, and normalized values.

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
        metadata={"Condition": "walking"},
    )
    subject.add_trial(trial.name, trial)

# Attaching a result does not align/crop the trial's other channels.
subject.get_trial_by_idx(0).attach_IK_results(str(output / "demo_IK.mot"))
# With your ID file:
# subject.get_trial_by_idx(0).attach_ID_results("walking_ID.sto")
```

Filter using the trial methods in an explicit loop. `Subject.lowpass_filter()`
still calls a removed trial method and currently prints errors without filtering.

```python
frames = []
for name, trial in subject.trials.items():
    trial.lowpass_filter_markers(cutoff_freq=6.0)
    frame = trial.as_df(trial.markers)
    frame.insert(0, "time", next(iter(trial.markers.values())).time)
    frame["trial_name"] = name
    frame["subject_id"] = subject.id
    frames.append(frame)
combined = pd.concat(frames, ignore_index=True)
combined.to_csv(output / "markers.csv", index=False)
```

Within a table, require common timestamps across its channels. `trial.as_df()`
and `subject.as_df("markers")` concatenate samples without aligning time;
`as_df(time_normalize=True)` currently does not perform normalization. Force
DataFrame conversion also has obsolete CoP attribute names. For a force table,
build columns explicitly from `plate.time`, `plate.Fx`, `plate.cop_x`, etc.

`subject.save_cache(cache_dir="output/cache")` writes a local pickle;
`Subject.load_from_cache("output/cache/P01_cache.pkl")` reads it back. Use caches
from your own trusted workflow and retain source recordings for reproducibility.
