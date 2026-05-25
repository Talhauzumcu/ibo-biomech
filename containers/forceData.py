import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class ForceData:
    """Data class for force plate data."""
    name: str
    force: np.ndarray  # (3, n_samples) - Fx, Fy, Fz
    moment: np.ndarray  # (3, n_samples) - Mx, My, Mz
    cop: np.ndarray  # (3, n_samples) - Center of Pressure x, y, z
    Tz: np.ndarray  # (3, n_samples,) - Moment at center of pressure
    metadata: Dict = field(default_factory=dict)
    sampling_rate: float = None
    
    def __post_init__(self):
        # self.transpose_data()
        self.clean_nan()

    def get_force_magnitude(self) -> np.ndarray:
        """Returns magnitude of force vector."""
        return np.sqrt(np.sum(self.force**2, axis=1))

    def transpose_data(self) -> None:
        """Transpose force, moment, and cop data to shape (n_samples, 3)."""
        self.force = self.force.T
        self.moment = self.moment.T
        self.cop = self.cop.T
        self.Tz = self.Tz.T

    def clean_nan(self) -> None:
        """Replace NaN values in force, moment, and cop with zeros."""
        self.force = np.nan_to_num(self.force)
        self.moment = np.nan_to_num(self.moment)
        self.cop = np.nan_to_num(self.cop)
        self.Tz = np.nan_to_num(self.Tz)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to force, moment, and cop data."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        self.force = filtfilt(b, a, self.force, axis=0)
        self.moment = filtfilt(b, a, self.moment, axis=0)
        self.cop = filtfilt(b, a, self.cop, axis=0)
        self.Tz = filtfilt(b, a, self.Tz, axis=0)

    def filter_low_forces(self, threshold: float = 10.0) -> None:
        """Set forces below threshold to zero."""
        force_magnitude = self.get_force_magnitude()
        low_force_indices = force_magnitude < threshold
        self.force[low_force_indices] = 0
        self.moment[low_force_indices] = 0
        self.cop[low_force_indices] = 0
        self.Tz[low_force_indices] = 0

    def downsample(self, factor: int) -> None:
        """Downsample force, moment, cop, and Tz data using scipy's decimate."""
        from scipy.signal import decimate
        self.force = decimate(self.force, factor, axis=0, ftype='fir', zero_phase=True)
        self.moment = decimate(self.moment, factor, axis=0, ftype='fir', zero_phase=True)
        self.cop = decimate(self.cop, factor, axis=0, ftype='fir', zero_phase=True)
        self.Tz = decimate(self.Tz, factor, axis=0, ftype='fir', zero_phase=True)
        if self.sampling_rate:
            self.sampling_rate /= factor

    @property
    def Fx(self) -> np.ndarray:
        return self.force[0, :]
    
    @property
    def Fy(self) -> np.ndarray:
        return self.force[1, :]
    
    @property
    def Fz(self) -> np.ndarray:
        return self.force[2, :]
        
    def plot(self) -> None:
        """Plot force, moment, and cop data."""
        import matplotlib.pyplot as plt
        time = np.arange(self.force.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.force.shape[0])
        
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(time, self.force)
        plt.title(f'{self.name} - Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(3, 1, 2)
        plt.plot(time, self.moment)
        plt.title(f'{self.name} - Moment')
        plt.xlabel('Time (s)')
        plt.ylabel('Moment (Nm)')
        plt.legend(['Mx', 'My', 'Mz'])
        
        plt.subplot(3, 1, 3)
        plt.plot(time, self.cop)
        plt.title(f'{self.name} - Center of Pressure')
        plt.xlabel('Time (s)')
        plt.ylabel('COP (m)')
        plt.legend(['COPx', 'COPy', 'COPz'])
        
        plt.tight_layout()
        plt.show()