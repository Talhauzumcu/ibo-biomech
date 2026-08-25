"""ibo-biomech: load, process and convert biomechanical motion-capture data.

Supports C3D files, the lab's HDF5 format and the OpenSim TRC/MOT formats. The
most commonly used classes are re-exported here for convenient top-level import::

    from ibo_biomech import C3DHandler, H5Handler, FileConverter
    from ibo_biomech import MarkerData, ForceData, AnalogData, TrialData, Subject
"""
from ibo_biomech.biomech_io import FileConverter
from ibo_biomech.containers import AnalogData, ForceData, MarkerData, EMGData, TrialData, Subject, IKResults, IDResults, Data
from ibo_biomech.handlers import C3DHandler, H5Handler
from ibo_biomech.analysis import GaitAnalyzer

__version__ = "0.2.10"
