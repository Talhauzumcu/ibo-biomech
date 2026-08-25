"""
Bare minimum data container for biomechanical data. Currently used for storing opensim ID and IK results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .data import Data
import numpy as np

@dataclass
class IKResults:

    name: str = None
    filepath : str = None
    unit: str = ""
    data: Dict[str, Data] = field(default_factory=dict)
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.filepath is not None:
            self.read(self.filepath)

    def read(self, filepath: str):
        """
        Reads IK results from a .mot file and stores them in the data dictionary.
        """
        from ibo_biomech.utils.utils import read_mot
        mot_data = read_mot(filepath)
        self.time = mot_data['time']
        self.metadata = mot_data.get('metadata', {})
        self.unit = 'deg' if self.metadata.get('inDegrees') == 'yes' else 'rad'
        self.columns = [key for key in mot_data.keys() if (key != 'time' and key != 'metadata')]
        for column in self.columns:
            self.data[column] = Data(name=column, 
                                     data=mot_data[column],
                                     unit = self.unit,
                                     time = self.time)

    def to_deg(self):
        """
        Converts the data to degrees if the unit is in radians.
        """
        non_angles = ['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        if self.unit.lower() == "rad":
            for column in self.columns:
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column in non_angles:
                    continue
                data = self.data[column]
                data.data = np.rad2deg(data.data)
            self.unit = "deg"
        else:
            print("Data is already in degrees or unit is not recognized.")

    def to_rad(self):
        """
        Converts the data to radians if the unit is in degrees.
        """
        non_angles = ['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        if self.unit.lower() == "deg":
            for column in self.columns:
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column in non_angles:
                    continue
                data = self.data[column]
                data.data = np.deg2rad(data.data)
            self.unit = "rad"
        else:
            print("Data is already in radians or unit is not recognized.")
        
    def write(self, filepath: str):
        """
        Writes the IK results to a .mot file.
        """
        num_columns = len(self.columns) + 1  # +1 for time column
        num_samples = len(self.time)
        inDegrees = "yes" if self.unit.lower() == "deg" else "no"
        with open(filepath, 'w') as f:
            # Header
            f.write("Coordinates\n")
            f.write("version=1\n")
            f.write(f"nRows={num_samples}\n")
            f.write(f"nColumns={num_columns}\n")
            f.write(f"inDegrees={inDegrees}\n")
            f.write("endheader\n")
            
            # Column headers
            header = "time"

            for column in self.columns:
                header += f"\t{column}"
            f.write(header + "\n")

            # Data rows
            for i in range(num_samples):
                row = f"{self.time[i]}"
                for column in self.columns:
                    row += f"\t{self.data[column].data[i]}"
                f.write(row + "\n")
        return filepath

    def add_column(self, name: str, data: np.ndarray, unit: str = "", time: Optional[np.ndarray] = None):
            """
            Adds a new column to the IK results.
    
            Args:
                name: Name of the new column.
                data: Signal samples for the new column, shape ``(n_samples,)``.
                unit: Physical unit of the signal (e.g. ``"V"``).
                time: Time vector corresponding to the data samples, shape ``(n_samples,)``.
            """
            if name in self.data:
                raise ValueError(f"Column '{name}' already exists.")
            if len(data) != len(self.time):
                raise ValueError("Data length must match the length of the time vector.")
            
            self.data[name] = Data(name=name, data=data, unit=unit, time=time)
            self.columns.append(name)

    def __repr__(self) -> str:
        """Return a concise summary of the data."""
        columns_str = "\n".join(self.columns)
        return (
            f"IKResults(name={self.name!r}, samples={len(self.time)}, "
            f"unit={self.unit!r}) \n"
            f"Columns: \n{columns_str}"
        )
    
    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

    