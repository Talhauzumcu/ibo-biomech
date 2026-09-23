# Run scaling, IK, and ID with OpenSim

This tutorial requires your own model, static/dynamic recordings, and OpenSim
setup XML files. Configure marker mappings, tracking weights, and scaling
measurements for your experiment in those setup files.

Install the OpenSim Python bindings in the same environment as ibo-biomech;
follow the [official Python setup guide](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346).
Verify `import opensim` before importing the wrapper. The wrapper is available
from its module, not the top-level package:

```python
from pathlib import Path
from ibo_biomech.handlers.osimHandler import OsimHandler

output = Path("output").resolve()
output.mkdir(exist_ok=True)
model = str(Path("models/generic.osim").resolve())
static_trc = str(Path("data/static.trc").resolve())
walking_trc = str(Path("data/walking.trc").resolve())
```

TRC and ground-reaction MOT files must already use your model's units and
coordinate system. See [exporting OpenSim inputs](opensim-export.md).

## Scale the model

Choose a valid static interval in your recording. Replace the example times and
mass with your participant's values.

```python
scaled_model = OsimHandler.run_scaling(
    model_path=model,
    setup_file=str(Path("setup/scale.xml").resolve()),
    trc_file=static_trc,
    mass=70.0,
    initial_time=0.5,
    final_time=1.0,
    move_markers=True,
    output_file=str(output / "P01_scaled.osim"),
    log_file=str(output / "P01_scale.log"),
)
```

`mass` and `height` are passed to OpenSim's subject settings; omitting them keeps
the setup values. This example leaves height in the setup. Scaling behavior also
depends on the configured measurements and scale factors. `move_markers` controls
the marker-placement stage.

## Run inverse kinematics

```python
ik_file = OsimHandler.run_ik(
    model_path=scaled_model,
    setup_file=str(Path("setup/ik.xml").resolve()),
    trc_file=walking_trc,
    output_file=str(output / "walking_IK.mot"),
    log_file=str(output / "walking_IK.log"),
)
```

When `initial_time` and `final_time` are omitted, the wrapper uses the TRC's
first and last times rather than the setup's time range. Both scaling and IK
also accept `h5_file=...` when `trc_file` is omitted; that route creates a temporary
TRC using the converter's default rotation and unit conversion.

## Map force plates and run inverse dynamics

`build_extloads()` assumes **exactly two plates**, with a fixed right/left
assignment, and model bodies named `calcn_r` and `calcn_l`. `r_idx` uses the
one-based MOT column numbering, so `r_idx=1` selects `ground_force_1_*` for the
right foot and plate 2 for the left. It does not detect which foot contacted a
plate. For other models, more plates, or changing assignments, supply your own
ExternalLoads XML.

```python
from ibo_biomech.utils.utils import build_extloads

external_loads = build_extloads(
    r_idx=1,
    output_file=output / "walking_external_loads.xml",
    mot_file="data/walking_grf.mot",
)
id_file = OsimHandler.run_id(
    model_path=scaled_model,
    setup_file=str(Path("setup/id.xml").resolve()),
    mot_file=ik_file,
    external_loads_file=str(external_loads),
    output_file=str(output / "walking_ID.sto"),
    lowpass_cutoff=-1.0,
    log_file=str(output / "walking_ID.log"),
)
```

For `run_id()`, `mot_file` is the **IK motion**, while ExternalLoads references the
**ground-reaction MOT**. The wrapper excludes muscles. `lowpass_cutoff=-1.0`
disables its coordinate filtering; use a positive value only when appropriate
for your processing pipeline. Missing start/end times are derived from the IK
file. Explicitly restrict them to an interval supported by the external loads.

Inspect generated files and OpenSim logs before interpreting results. The
wrapper currently lacks guaranteed cleanup on exceptions and does not check the
tool's success return value. These examples document the current calls; executing
them requires a compatible OpenSim installation and validated model/setup files.
