from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class MarkerData:
    """Data class for 3D marker trajectories."""
    name: str
    x: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sampling_rate: float = None
    virtual: int = 0 #Whether the marker is virtual or measured.

    def get_trajectory(self) -> np.ndarray:
        """Returns marker trajectory as (n_samples, 3) array."""
        return np.column_stack([self.x, self.y, self.z])
    
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