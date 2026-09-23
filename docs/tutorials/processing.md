# Process markers, forces, and EMG

The examples below use synthetic data and run in order without recording files.
Filter cutoffs demonstrate the API; select your analysis parameters separately.
Filtering requires a positive sampling rate, a cutoff below half that rate, and
enough samples for the filter's padding. Process a copy when retaining raw data.

## Fill an interior gap and filter markers

```python
from copy import deepcopy
import numpy as np
from ibo_biomech import MarkerData, TrialData

fs = 100.0
time = 2.0 + np.arange(300) / fs
x = 1000.0 + 50.0 * np.sin(2 * np.pi * (time - time[0]))
x[100:103] = np.nan  # a deliberately short interior gap
right = MarkerData(
    name="R_Knee", x=x, y=np.zeros_like(time), z=np.full_like(time, 500.0),
    unit="mm", sampling_rate=fs, time=time,
)
left = MarkerData(
    name="L_Knee", x=x.copy() + 100.0, y=np.zeros_like(time),
    z=np.full_like(time, 500.0), unit="mm", sampling_rate=fs, time=time.copy(),
)
raw = TrialData(name="demo", markers={right.name: right, left.name: left})
trial = deepcopy(raw)
for marker in trial.markers.values():
    marker.clean_nan()
    if not np.isfinite(marker.data).all():
        raise ValueError(f"Unresolved gaps in {marker.name}")
trial.lowpass_filter_markers(cutoff_freq=6.0, order=4)
```

`clean_nan()` must be called explicitly. It linearly fills interior NaNs without
a maximum gap length, leaves leading/trailing gaps missing, and currently raises
for an entirely missing axis. Review gap lengths first. Filtering unresolved
NaNs can contaminate the whole signal.

## Create a virtual marker with units and timestamps

```python
right = trial.markers["R_Knee"]
left = trial.markers["L_Knee"]
if right.unit != left.unit or not np.array_equal(right.time, left.time):
    raise ValueError("Midpoint inputs must share units and timestamps")
mid = MarkerData(
    name="MidKnee",
    x=(right.x + left.x) / 2,
    y=(right.y + left.y) / 2,
    z=(right.z + left.z) / 2,
    unit=right.unit,
    sampling_rate=right.sampling_rate,
    time=right.time.copy(),
    virtual=1,
)
trial.add_marker(mid)
```

Marker arithmetic such as `(right + left) / 2` is available, but currently
regenerates time from zero and does not validate units or time alignment. The
explicit constructor above preserves this example's 2-second offset.

## Crop by each channel's time

`crop(start_idx, end_idx)` keeps `[start_idx, end_idx)`. At trial level,
`trial.crop("markers", start_idx, end_idx)` selects a data type; it does not crop
all data types simultaneously. Identical indices at 100 Hz and 1,000 Hz represent
different intervals.

Use timestamps for a common interval. This helper is tutorial code, not a package
method. It accepts markers, analogs, forces, EMG, or result containers with a
sorted time vector. Filter before cropping when appropriate for your workflow.

```python
def crop_time(container, start, end):
    time = container.time
    if time is None or len(time) == 0 or not np.all(np.isfinite(time)):
        raise ValueError("A finite time vector is required")
    if not np.all(np.diff(time) > 0) or start >= end:
        raise ValueError("Use increasing timestamps and start < end")
    first, stop = np.searchsorted(time, [start, end], side="left")
    if first >= stop:
        raise ValueError("No samples in the requested interval")
    container.crop(int(first), int(stop))

for marker in trial.markers.values():
    crop_time(marker, 2.5, 4.0)
assert trial.markers["MidKnee"].time[0] == 2.5
```

Apply this to each relevant channel when cropping a multi-rate recording. It
preserves original timestamps and does not resample or correct acquisition offsets.

## Force signals and geometry

A manually constructed plate should supply arrays with consistent sample counts.
`corners` has shape `(3, 4, n)` and `rotation` has shape `(3, 3, n)`. The local
`origin` offset is static. A C3D loader normally supplies these fields for you.

```python
from ibo_biomech import ForceData

n = 1000
force_time = np.arange(n) / 1000.0
force = np.vstack([np.zeros(n), np.zeros(n), np.full(n, 700.0)])
plate = ForceData(
    name="forceplate_0", force=force,
    moment=np.zeros((3, n)), cop=np.zeros((3, n)), Tz=np.zeros(n),
    corners=np.repeat(np.array([
        [250., -250., -250., 250.],
        [250., 250., -250., -250.],
        [0., 0., 0., 0.],
    ])[:, :, None], n, axis=2),
    position=np.zeros((3, n)),
    rotation=np.repeat(np.eye(3)[:, :, None], n, axis=2),
    origin=np.zeros((3, 1)), sampling_rate=1000.0, time=force_time,
    metadata={"unit_force": "N", "unit_moment": "Nmm", "unit_position": "mm"},
)
plate.lowpass_filter(cutoff=20.0)
plate.filter_low_forces(threshold=10.0)
plate.rotate(axis="x", angle_deg=-90)
plate.convert_units("m")
print(plate.Fy.mean(), plate.unit_cop)  # approximately 700 N, m
```

For this example, +Z is vertical before the rotation and +Y afterwards. Rotation
must match your acquisition coordinates. `convert_units("m")` converts lengths
and length-dependent moments; force magnitudes remain in their original unit.

Current force filtering also filters plate positions; high-pass filtering can
therefore erase a static position. Optional one-sample defaults for `Tz` and
position can cause filter failures. `downsample(factor)` currently filters time
and rotation matrices, distorting both; avoid it until the
[resampling fix](../remaining-issues.md) is implemented. Access writable signals
through `plate.force`, `plate.moment`, or `plate.cop`; stacked `plate.data` is a copy.

## Analog and EMG signals

```python
from ibo_biomech import AnalogData, EMGData

emg_time = 2.0 + np.arange(2000) / 1000.0
rng = np.random.default_rng(42)
analog = AnalogData(
    name="EMG_VastusLat", data=rng.normal(0, 0.001, emg_time.size),
    sampling_rate=1000.0, time=emg_time, unit="V", channel=3,
)
emg = EMGData(
    name=analog.name, data=analog.data.copy(), time=analog.time.copy(),
    sampling_rate=analog.sampling_rate, unit=analog.unit, channel=analog.channel,
)
trial.add_emg(emg)
envelope = emg.process_emg()
assert envelope.shape == emg.data.shape
```

The implemented pipeline cleans NaNs, high-pass filters at 30 Hz (order 2),
squares the signal, low-pass filters at 10 Hz (order 2), and normalizes to the
peak. The returned envelope is dimensionless; it is not MVC-normalized.

`trial.parse_EMG_data([3])` also creates EMG channels from analogs, but currently
shares the source array and resets its time origin. Explicit copying preserves
both independence and timing. `processed_data` caches the envelope; subsequent
raw-data edits or filtering do not invalidate that cache. Call `process_emg()`
again for a fresh result after edits.

For plotting, install `matplotlib`, then use `marker.plot()`, `plate.plot()`,
`analog.plot()`, or `emg.plot_processed()`. The marker plot currently labels its
axis as mm even after conversion; check the container's `unit`.
