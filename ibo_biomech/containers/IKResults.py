"""
Bare minimum data container for biomechanical data. Currently used for storing opensim IK results.
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from .motResults import MotResults
 
_NON_ANGLE_COLUMNS = ['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
 
 
@dataclass
class IKResults(MotResults):
    """Inverse kinematics results read from a .mot file.
 
    Adds degrees/radians conversion (:meth:`to_deg`, :meth:`to_rad`) on top
    of :class:`MotResults`, and its ``.mot`` writer emits an ``inDegrees=``
    header line that :class:`IDResults`' writer does not.
    """
    def __post_init__(self):
        super().__post_init__()
        if self.unit.lower() not in ['deg', 'rad']:
            raise ValueError(f"Unit must be 'deg' or 'rad', got '{self.unit}'")
        
    def _extra_header_lines(self) -> List[str]:
        in_degrees = "yes" if self.unit.lower() == "deg" else "no"
        return [f"inDegrees={in_degrees}\n"]
 
    def to_deg(self):
        """Convert the data to degrees if the unit is in radians."""
        if self.unit.lower() == "rad":
            for column in self.data.values():
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column.name}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column.name in _NON_ANGLE_COLUMNS:
                    continue
                column.data = np.rad2deg(column.data)
                column.unit = 'deg'
            self.unit = "deg"
        else:
            print("Data is already in degrees or unit is not recognized.")
 
    def to_rad(self):
        """Convert the data to radians if the unit is in degrees."""
        if self.unit.lower() == "deg":
            for column in self.data.values():
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column.name}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column.name in _NON_ANGLE_COLUMNS:
                    continue
                column.data = np.deg2rad(column.data)
                column.unit = 'rad'
            self.unit = "rad"
        else:
            print("Data is already in radians or unit is not recognized.")