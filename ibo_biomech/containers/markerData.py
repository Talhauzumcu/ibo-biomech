from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from ibo_biomech.utils.utils import *
from dataclasses import dataclass, field

@dataclass
class MarkerData:
    """Data class for 3D marker trajectories."""
    name: str
    x: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z: np.ndarray = field(default_factory=lambda: np.zeros(1))
    unit: str = 'mm'
    sampling_rate: float = None
    virtual: int = 0 #Whether the marker is virtual or measured.

    def get_trajectory(self) -> np.ndarray:
        """Returns marker trajectory as (3, n_samples) array."""
        return np.column_stack([self.x, self.y, self.z]).T
    
    def get_magnitude(self) -> np.ndarray:
        """Returns magnitude of 3D position."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def get_frame_trajectory(self) -> np.ndarray:
        """Returns trajectory in (4, n_samples) homogeneous coordinates."""
        n_samples = self.x.shape[0]
        ones = np.ones(n_samples)
        return np.vstack([self.x, self.y, self.z, ones])
    
    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to marker trajectories."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)
        self.z = filtfilt(b, a, self.z)
    
    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply high-pass Butterworth filter to marker trajectories."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)
        self.z = filtfilt(b, a, self.z)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Returns a new MarkerData instance with data sliced between start_idx and end_idx."""
        self.x=self.x[start_idx:end_idx]
        self.y=self.y[start_idx:end_idx]
        self.z=self.z[start_idx:end_idx]

    def rotate(self, axis, angle_deg) -> None:
        """Rotate marker trajectory around specified axis by given angle in degrees."""
        rotation_matrix = get_rotation_matrix(axis, angle_deg)
        trajectory = self.get_trajectory()  # (3, n_samples)
        rotated_traj = rotation_matrix @ trajectory  # (3, n_samples)
        self.x = rotated_traj[0, :]
        self.y = rotated_traj[1, :]
        self.z = rotated_traj[2, :]

    def convert_units(self, target_unit: str) -> None:
        """Convert marker trajectory units to target_unit (e.g. 'mm' to 'm')."""
        conversion_factors = {
            ('mm', 'm'): 0.001,
            ('m', 'mm'): 1000,
        }
        if self.unit == target_unit:
            return  # No conversion needed
        key = (self.unit, target_unit)
        if key not in conversion_factors:
            raise ValueError(f"Unsupported unit conversion: {self.unit} to {target_unit}")
        factor = conversion_factors[key]
        self.x *= factor
        self.y *= factor
        self.z *= factor
        self.unit = target_unit

    def plot(self) -> None:
        """Plot marker trajectory in 3D space."""
        import matplotlib.pyplot as plt
        time = np.arange(len(self.x)) / self.sampling_rate if self.sampling_rate else np.arange(len(self.x))
        
        plt.figure(figsize=(12, 8))
        plt.plot(time, self.get_trajectory())
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title(f'Marker: {self.name}')
        plt.legend(['X', 'Y', 'Z'])
        plt.show()

    def __add__(self, other: MarkerData) -> MarkerData:
        """Add two MarkerData instances together (e.g. for averaging)."""
        if self.x.shape != other.x.shape:
            raise ValueError("Cannot add MarkerData with different shapes.")
        
        name = f"{self.name}_plus_{other.name}"
        return MarkerData(
            name=name,
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            sampling_rate=self.sampling_rate,
            virtual = 1
        )
    
    def __truediv__(self, other: MarkerData | float | int) -> MarkerData:
        """Divide MarkerData by another MarkerData or a scalar."""
        if isinstance(other, MarkerData):
            if self.x.shape != other.x.shape:
                raise ValueError("Cannot divide MarkerData with different shapes.")
            return MarkerData(
                name=f"{self.name}_div_{other.name}",
                x=self.x / other.x,
                y=self.y / other.y,
                z=self.z / other.z,
                sampling_rate=self.sampling_rate,
                virtual=1
            )
        elif isinstance(other, (float, int)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide MarkerData by zero.")
            return MarkerData(
                name=f"{self.name}_div_{other}",
                x=self.x / other,
                y=self.y / other,
                z=self.z / other,
                sampling_rate=self.sampling_rate,
                virtual=1
            )
        return NotImplemented