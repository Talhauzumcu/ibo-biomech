# Detect gait events

`GaitAnalyzer` contains a force-peak-based detector adapted for a particular lab
workflow. It assumes +Z is vertical, the lower of two foot markers identifies the
contacting foot, and force samples share the analog sampling rate and time origin.
Run it in acquisition coordinates before a Z-up to Y-up export rotation.

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
    if not (np.isfinite(events["TDa"]).all() and np.isfinite(events["TOa"]).all()):
        raise ValueError("A contact extends beyond the available recording")
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

`TDa`/`TOa` are force/analog indices and `TDv`/`TOv` are rounded marker indices.
The stance/step helper specifically requires markers named `LTOE` and `RTOE`.
It computes step displacement along X between consecutive touchdown positions,
in the marker length unit. Its first step-length entry is `None`.

## Current limitations to account for

- A plate loaded above threshold at the very first sample causes every contact
  on that plate to be skipped, including later complete contacts.
- Multiple force peaks within one contact may produce duplicate events.
- The rate-ratio mapping ignores time offsets and can round beyond the last
  marker sample. Cropped, resampled, or asynchronous signals need alignment.
- A missing touchdown/toe-off remains NaN; marker-index conversion can emit
  warnings before this tutorial's result check.
- No-contact results contain `None`; handle that before calling the stance helper.
- Foot assignment uses marker height only; it does not compare foot location
  with plate geometry. The step displacement is unsuitable as a general
  treadmill step-length calculation.

Inspect force and marker traces against the detected events. Proposed repairs
are recorded in [remaining issues](../remaining-issues.md).
