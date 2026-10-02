# Detect gait events

`GaitAnalyzer` contains a force-peak-based detector adapted for a particular lab
workflow. It assumes +Z is vertical, the lower of two foot markers identifies the
contacting foot, and force samples share the analog sampling rate and time origin.
Run it in acquisition coordinates before a Z-up to Y-up export rotation.

Use uniform, synchronized marker/force clocks with the same first timestamp,
matching force and analog rates, and complete contacts that begin and end within
the recording. Each contact should have one qualifying force peak, and every
plate should be unloaded at the start. Check the marker traces used for foot
assignment before analyzing a recording.

The following synthetic example demonstrates the API, not validation of event
detection for real walking, running, or pathological gait.

## Build a trial with one complete contact

```python
import numpy as np
from ibo_biomech import AnalogData, ForceData, GaitAnalyzer, MarkerData, TrialData

marker_fs = 100.0
force_fs = 1000.0
marker_time = np.arange(200) / marker_fs
force_time = np.arange(2000) / force_fs
fz = np.zeros(force_time.size)
fz[500:1000] = 1000.0 * np.sin(np.linspace(0, np.pi, 500))

markers = {
    "LTOE": MarkerData(
        "LTOE", x=1000.0 * marker_time, y=np.zeros(200), z=np.zeros(200),
        sampling_rate=marker_fs, time=marker_time, unit="mm",
    ),
    "RTOE": MarkerData(
        "RTOE", x=1000.0 * marker_time, y=np.zeros(200), z=np.full(200, 100.0),
        sampling_rate=marker_fs, time=marker_time, unit="mm",
    ),
}
plate = ForceData(
    name="forceplate_0", force=np.vstack([np.zeros(2000), np.zeros(2000), fz]),
    sampling_rate=force_fs, time=force_time,
    metadata={"unit_force": "N", "unit_moment": "Nmm", "unit_position": "mm"},
)
# The current detector takes its rate from trial.analog_rate.
analog = AnalogData("Fz_raw", data=fz.copy(), sampling_rate=force_fs, time=force_time)
trial = TrialData(
    name="synthetic_contact", markers=markers,
    forces={plate.name: plate}, analogs={analog.name: analog},
)
events = GaitAnalyzer.get_plate_contacts(
    trial, bodyweight=70.0 * 9.81, marker_re="RTOE", marker_li="LTOE",
    threshold=20.0, threshold_multiplier=1.2, prominence_multiplier=0.8,
)
```

`bodyweight` is a force in newtons because it is compared with force-peak height;
convert a mass in kg accordingly. The peak multipliers are tunable detection
parameters, not universally appropriate defaults. Plate names in the result are
strings such as `forceplate_0`, not numeric plate indices.

## Inspect contacts and compute durations

```python
if events["feet"] is None:
    print("No contacts detected")
else:
    if not all(np.isfinite(events[key]).all() for key in ("TDa", "TOa")):
        raise ValueError("A contact extends beyond the available recording")
    for key in ("TDv", "TOv"):
        if np.any(events[key] < 0) or np.any(events[key] >= marker_time.size):
            raise ValueError("A contact is outside the available marker samples")
    for name, foot, touchdown, toeoff in zip(
        events["plateNames"], events["feet"], events["TDa"], events["TOa"],
    ):
        time = trial.forces[name].time
        print(foot, time[int(touchdown)], time[int(toeoff)])
    stance_times, step_lengths = GaitAnalyzer.get_stancetimes_steplengths(
        trial.markers, events, aFrq=trial.analog_rate,
    )
    print(stance_times, step_lengths)
```

`TDa`/`TOa` are force/analog array indices and `TDv`/`TOv` are rounded marker
array indices. They are relative to the supplied arrays, unlike `Event.frame`,
which is a source frame number. The returned dictionary does not populate
`trial.events` automatically.
The stance/step helper specifically requires markers named `LTOE` and `RTOE`.
It computes step displacement along X between consecutive touchdown positions,
in the marker length unit. Its first step-length entry is `None`.

No-contact results contain `None`, so the example checks for contacts before
calling the stance helper. Inspect force and marker traces against the detected
events. The X displacement reported here describes consecutive touchdown
positions in the lab frame; choose an appropriate definition for treadmill
step-length analysis.
