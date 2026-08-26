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
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.filepath is not None:
            self.read(self.filepath)
        if len(self.data.keys()) > 0:
            self.columns = list(self.data.keys())

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
            for column in self.data.values():
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column.name in non_angles:
                    continue
                column.data = np.rad2deg(column.data)
                column.unit = 'deg'
            self.unit = "deg"
        else:
            print("Data is already in degrees or unit is not recognized.")

    def to_rad(self):
        """
        Converts the data to radians if the unit is in degrees.
        """
        non_angles = ['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        if self.unit.lower() == "deg":
            for column in self.data.values():
                if column.unit != 'deg' and column.unit != 'rad':
                    print(f"Warning: Column '{column}' has an unrecognized unit '{column.unit}'. Skipping conversion.")
                    continue
                if column.name in non_angles:
                    continue
                column.data = np.deg2rad(column.data)
                column.unit = 'rad'
            self.unit = "rad"
        else:
            print("Data is already in radians or unit is not recognized.")

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase high-pass Butterworth filter to force, moment and CoP.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for data in self.data.values():
            data.highpass_filter(cutoff, order)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase low-pass Butterworth filter to force, moment and CoP.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for data in self.data.values():
            data.lowpass_filter(cutoff, order)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the signal in place to ``[start_idx, end_idx)``.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """

        if start_idx < 0 or end_idx > len(self.time) or start_idx >= end_idx:
            raise ValueError("Invalid crop indices.")
        
        self.time = self.time[start_idx:end_idx]
        for data in self.data.values():
            data.crop(start_idx, end_idx)

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

    def __len__(self):
        """Return the number of samples in the data."""
        return len(self.data.keys())