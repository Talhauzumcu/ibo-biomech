# ibo-biomech documentation

Use ibo-biomech to load motion-capture trials, process named signals, prepare
OpenSim inputs, and organize results for analysis. Install with
`python -m pip install ibo-biomech`, or `python -m pip install -e .` when working
from this checkout. Plotting additionally needs `matplotlib`.

`C3DHandler` and `H5Handler` return `TrialData`. A trial groups `MarkerData`,
`ForceData`, `AnalogData`, and `EMGData`; it can also hold `IKResults` and
`IDResults`. `Subject` groups multiple trials. Most processing methods mutate
the selected container, so copy a trial before processing if you need its raw data.

Start with loading a recording, or try the synthetic examples in the processing
and gait tutorials without a recording. Examples on each tutorial page run in
order unless marked as an alternative. File-based examples require your own
recordings, models, and setup files; their paths and labels are placeholders.

The package is in alpha. The [remaining issues](remaining-issues.md) page records
verified limitations and proposed fixes, including workflows that are currently
unsuitable for saving or exporting processed data.

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/loading-trials
tutorials/processing
tutorials/opensim-export
tutorials/results-and-subjects
tutorials/opensim-tools
tutorials/gait-events
```

```{toctree}
:caption: API reference
:maxdepth: 1

api/c3dhandler
api/h5handler
api/fileconverter
api/markerdata
api/forcedata
api/analogdata
api/emgdata
api/trialdata
api/subject
api/data
api/motResults
api/ikresults
api/idresults
api/osimHandler
api/gaitanalyzer
api/utils
api/_mixins
```

```{toctree}
:caption: Development
:maxdepth: 1

remaining-issues
```
