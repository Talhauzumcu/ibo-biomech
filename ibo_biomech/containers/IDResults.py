"""
Bare minimum data container for biomechanical data. Currently used for storing opensim ID and IK results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .data import Data
import numpy as np

@dataclass
class IDResults:

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
        Reads ID results from a .mot file and stores them in the data dictionary.
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
        
    def write(self, filepath: str):
        """
        Writes the ID results to a .mot file.
        """
        num_columns = len(self.columns) + 1  # +1 for time column
        num_samples = len(self.time)
        with open(filepath, 'w') as f:
            # Header
            f.write("Coordinates\n")
            f.write("version=1\n")
            f.write(f"nRows={num_samples}\n")
            f.write(f"nColumns={num_columns}\n")
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
            Adds a new column to the ID results.
    
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
        return (
            f"IDResults(name={self.name!r}, samples={len(self.time)}, "
            f"unit={self.unit!r}) \n"
            f"Columns: \n{'\n'.join(self.columns)}"
        )
    
    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

    