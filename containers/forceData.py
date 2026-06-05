import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from utils.utils import * 

@dataclass
class ForceData:
    """Data class for force plate data."""
    name: str
    force: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))  # (3, n_samples) - Fx, Fy, Fz
    moment: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))  # (3, n_samples) - Mx, My, Mz
    cop: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))  # (3, n_samples) - Center of Pressure x, y, z
    location: np.ndarray = field(default_factory=lambda: np.zeros((3, 4, 1))) # (3, 4, n_samples) - Location forceplate corners (4 corners with x, y, z coordinates)
    position: np.ndarray = field(default_factory=lambda: np.zeros((3, 1))) # (3, n_samples) - Position of force plate origin
    rotation: np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 1))) # (3, 3, n_samples) - Rotation matrix of force plate orientation
    offset: np.ndarray = field(default_factory=lambda: np.zeros((3, 1))) # ndarray(3, 1) - forceplate origin offset under corners.  ## Is this relative to the mean position of the corners?
    Tz: np.ndarray = field(default_factory=lambda: np.zeros((1,))) # (n_samples,) - Vertical force component used for gait event detection
    coordinateSystem: int = field(default_factory=lambda: 0) # is forceplate data saved in (1 = global, 0 = local) coordinates
    metadata: Dict = field(default_factory=dict)
    sampling_rate: float = None
    
    def __post_init__(self):
        self.clean_nan()
        self._parse_metadata()
        self.num_samples = self.force.shape[1]

    def _parse_metadata(self) -> None:
        self.unit_force = self.metadata.get('unit_force', 'Unknown')
        self.unit_moment = self.metadata.get('unit_moment', 'Unknown')
        self.unit_cop = self.metadata.get('unit_position', 'Unknown')
    
    def get_force_magnitude(self) -> np.ndarray:
        """Returns magnitude of force vector."""
        return np.sqrt(np.sum(self.force**2, axis=1))

    def clean_nan(self) -> None:
        """Replace NaN values in force, moment, and cop with zeros."""
        self.force = np.nan_to_num(self.force)
        self.moment = np.nan_to_num(self.moment)
        self.cop = np.nan_to_num(self.cop)
        self.location = np.nan_to_num(self.location)
        self.position = np.nan_to_num(self.position)
        self.rotation = np.nan_to_num(self.rotation)
        self.Tz = np.nan_to_num(self.Tz)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to force, moment, and cop data."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        self.force = filtfilt(b, a, self.force, axis=1)
        self.moment = filtfilt(b, a, self.moment, axis=1)
        self.cop = filtfilt(b, a, self.cop, axis=1)

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply high-pass Butterworth filter to force, moment, and cop data."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        self.force = filtfilt(b, a, self.force, axis=1)
        self.moment = filtfilt(b, a, self.moment, axis=1)
        self.cop = filtfilt(b, a, self.cop, axis=1)

    def filter_low_forces(self, threshold: float = 10.0) -> None:
        """Set forces below threshold to zero."""
        force_magnitude = self.get_force_magnitude()
        low_force_indices = force_magnitude < threshold
        self.force[low_force_indices] = 0
        self.moment[low_force_indices] = 0
        self.cop[low_force_indices] = 0

    def downsample(self, factor: int) -> None:
        """Downsample force, moment, cop, and Tz data using scipy's decimate."""
        from scipy.signal import decimate
        self.force = decimate(self.force, factor, axis=0, ftype='fir', zero_phase=True)
        self.moment = decimate(self.moment, factor, axis=0, ftype='fir', zero_phase=True)
        self.cop = decimate(self.cop, factor, axis=0, ftype='fir', zero_phase=True)
        if self.sampling_rate:
            self.sampling_rate /= factor

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop force, moment, cop, and Tz data to specified index range."""
        self.force = self.force[:, start_idx:end_idx]
        self.moment = self.moment[:, start_idx:end_idx]
        self.cop = self.cop[:, start_idx:end_idx]
        self.location = self.location[:, :, start_idx:end_idx]
        self.position = self.position[:, start_idx:end_idx]
        self.rotation = self.rotation[:, :, start_idx:end_idx]

    def rotate(self, axis, angle_deg) -> None:
        rotation_matrix = get_rotation_matrix(axis, angle_deg)
        self.force = rotation_matrix @ self.force
        self.moment = rotation_matrix @ self.moment
        self.cop = rotation_matrix @ self.cop
        self.position = rotation_matrix @ self.position
        self.rotation = rotation_matrix @ self.rotation

    def convert_units(self, target_unit: str) -> None:
        """Convert force units to target_unit (e.g. 'mm' to 'm'). Usually moments are in Nmm and COP is in mm"""
        conversion_factors = {
            ('mm', 'm'): 0.001,
            ('m', 'mm'): 1000,
        }
        if self.unit_cop == target_unit:
            return
        key = (self.unit_cop, target_unit)
        if key not in conversion_factors:
            raise ValueError(f"Unsupported unit conversion: {self.unit_cop} to {target_unit}")
        factor = conversion_factors[key]
        self.moment *= factor
        self.cop *= factor
        self.Tz *= factor
        self.unit_moment = f'N{target_unit}'
        self.unit_cop = target_unit
        
    def plot(self) -> None:
        """Plot force, moment, and cop data."""
        import matplotlib.pyplot as plt
        time = np.arange(self.force.shape[1]) / self.sampling_rate if self.sampling_rate else np.arange(self.force.shape[1])
        
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(time, self.force.T)
        plt.title(f'{self.name} - Force')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Force {self.metadata.get("unit_force", "N")}')
        plt.legend(['Fx', 'Fy', 'Fz'])
        
        plt.subplot(3, 1, 2)
        plt.plot(time, self.moment.T)
        plt.title(f'{self.name} - Moment')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Moment {self.metadata.get("unit_moment", "Nm")}')
        plt.legend(['Mx', 'My', 'Mz'])
        
        plt.subplot(3, 1, 3)
        plt.plot(time, self.cop.T)
        plt.title(f'{self.name} - Center of Pressure')
        plt.xlabel('Time (s)')
        plt.ylabel(f'COP {self.metadata.get("unit_cop", "mm")}')
        plt.legend(['COPx', 'COPy', 'COPz'])
        
        plt.tight_layout()
        plt.show()

    @property
    def Fx(self) -> np.ndarray:
        return self.force[0, :]
    
    @property
    def Fy(self) -> np.ndarray:
        return self.force[1, :]
    
    @property
    def Fz(self) -> np.ndarray:
        return self.force[2, :]

    @property
    def cop_x(self) -> np.ndarray:
        return self.cop[0, :]
    
    @property
    def cop_y(self) -> np.ndarray:
        return self.cop[1, :]

    @property
    def cop_z(self) -> np.ndarray:
        return self.cop[2, :]
    
    @property
    def Mx(self) -> np.ndarray:
        return self.moment[0, :]
    
    @property
    def My(self) -> np.ndarray:
        return self.moment[1, :]
    
    @property
    def Mz(self) -> np.ndarray:
        return self.moment[2, :]
    
    @property
    def x(self) -> np.ndarray:
        return self.position[0, :]
    
    @property
    def y(self) -> np.ndarray:
        return self.position[1, :]
    
    @property
    def z(self) -> np.ndarray:
        return self.position[2, :]
    
    @property
    def unit_position(self) -> str:
        return self.unit_cop