"""
Bare minimum data container for biomechanical data. Currently used for storing opensim ID results.
"""
from dataclasses import dataclass
from .motResults import MotResults
 
 
@dataclass
class IDResults(MotResults):
    """Inverse dynamics results read from a .mot file.
 
    Carries no quirks of its own beyond :class:`MotResults` -- unlike
    :class:`IKResults`, it has no degrees/radians concept and its
    ``.mot`` writer does not emit an ``inDegrees=`` header line.
    """
    def __post_init__(self):
        super().__post_init__()