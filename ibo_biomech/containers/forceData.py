"""Force plate data container.

This module defines :class:`ForceData`, which holds the force, moment and
centre-of-pressure signals of a single force plate together with helpers for
filtering, cropping, rotating and unit conversion.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ibo_biomech.utils.utils import *

@dataclass
class ForceData:
    """Signals and geometry for a single force plate.

    Most processing methods operate **in place**. Convenience properties
    (:attr:`Fx`, :attr:`cop_x`, :attr:`Mx`, ...) expose individual rows of the
    stacked arrays.

    Attributes:
        name: Force plate name (e.g. ``"forceplate_0"``).
        force: Force components ``Fx, Fy, Fz``, shape ``(3, n_samples)``.
        moment: Moment components ``Mx, My, Mz``, shape ``(3, n_samples)``.
        cop: Centre of pressure ``x, y, z``, shape ``(3, n_samples)``.
        location: Plate corner coordinates, shape ``(3, 4, n_samples)``.
        position: Plate origin position, shape ``(3, n_samples)``.
        rotation: Plate orientation matrices, shape ``(3, 3, n_samples)``.
        offset: Origin offset relative to the corners, shape ``(3, 1)``.
        Tz: Vertical free moment used for gait-event detection, shape
            ``(n_samples,)``.
        coordinateSystem: ``1`` if data is in global coordinates, ``0`` if local.
        metadata: Raw plate metadata (units, calibration matrix, corners, ...).
        sampling_rate: Sampling frequency in Hz. Required for filtering.
        num_samples: Number of samples; derived in :meth:`__post_init__`.
        unit_force: Force unit, derived from ``metadata``.
        unit_moment: Moment unit, derived from ``metadata``.
        unit_cop: Centre-of-pressure unit, derived from ``metadata``.
    """
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
    time: Optional[np.ndarray] = None

    def __post_init__(self):
        """Clean NaNs, parse unit metadata and cache the sample count."""
        self.clean_nan()
        self._parse_metadata()
        self.num_samples = self.force.shape[1]
        assert self.force.shape[0] == 3, "Force array must have shape (3, n_samples)"

        if self.time is None and self.sampling_rate is not None:
            self.time = np.arange(self.num_samples) / self.sampling_rate
            
    def _parse_metadata(self) -> None:
        """Populate ``unit_force``, ``unit_moment`` and ``unit_cop`` from metadata."""
        self.unit_force = self.metadata.get('unit_force', 'Unknown')
        self.unit_moment = self.metadata.get('unit_moment', 'Unknown')
        self.unit_cop = self.metadata.get('unit_position', 'Unknown')

    def get_force_magnitude(self) -> np.ndarray:
        """Return the magnitude of the force vector per sample.

        Returns:
            Array of shape ``(n_samples,)`` with the Euclidean norm of the force vector at each time point.
        """
        return np.sqrt(np.sum(self.force**2, axis=0))

    def clean_nan(self) -> None:
        """Replace NaN values in all signal and geometry arrays with zeros."""
        self.force = np.nan_to_num(self.force)
        self.moment = np.nan_to_num(self.moment)
        self.cop = np.nan_to_num(self.cop)
        self.location = np.nan_to_num(self.location)
        self.position = np.nan_to_num(self.position)
        self.rotation = np.nan_to_num(self.rotation)
        self.Tz = np.nan_to_num(self.Tz)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase low-pass Butterworth filter to force, moment and CoP.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        self.force = filtfilt(b, a, self.force, axis=1)
        self.moment = filtfilt(b, a, self.moment, axis=1)
        self.cop = filtfilt(b, a, self.cop, axis=1)

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase high-pass Butterworth filter to force, moment and CoP.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        self.force = filtfilt(b, a, self.force, axis=1)
        self.moment = filtfilt(b, a, self.moment, axis=1)
        self.cop = filtfilt(b, a, self.cop, axis=1)

    def filter_low_forces(self, threshold: float = 10.0) -> None:
        """Zero out frames whose force magnitude is below a threshold.

        Force, moment and CoP are set to zero wherever the force magnitude is
        below ``threshold``, removing spurious readings during swing phases.

        Args:
            threshold: Force magnitude below which frames are zeroed, in newtons.
                Defaults to 10.0.
        """
        force_magnitude = self.get_force_magnitude()
        low_force_indices = force_magnitude < threshold
        self.force[low_force_indices] = 0
        self.moment[low_force_indices] = 0
        self.cop[low_force_indices] = 0

    def downsample(self, factor: int) -> None:
        """Downsample force, moment and CoP in place using FIR decimation.

        Updates ``sampling_rate`` accordingly. Uses zero-phase decimation.

        Args:
            factor: Integer downsampling factor.
        """
        from scipy.signal import decimate
        self.force = decimate(self.force, factor, axis=0, ftype='fir', zero_phase=True)
        self.moment = decimate(self.moment, factor, axis=0, ftype='fir', zero_phase=True)
        self.cop = decimate(self.cop, factor, axis=0, ftype='fir', zero_phase=True)
        self._update_num_samples()
        if self.sampling_rate:
            self.sampling_rate /= factor

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop all signal and geometry arrays to ``[start_idx, end_idx)``.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """
        if start_idx < 0 or end_idx > self.num_samples or start_idx >= end_idx:
            raise ValueError("Invalid crop indices.")
        
        self.force = self.force[:, start_idx:end_idx]
        self.moment = self.moment[:, start_idx:end_idx]
        self.cop = self.cop[:, start_idx:end_idx]
        self.location = self.location[:, :, start_idx:end_idx]
        self.position = self.position[:, start_idx:end_idx]
        self.rotation = self.rotation[:, :, start_idx:end_idx]
        self.Tz = self.Tz[start_idx:end_idx]
        self.time = self.time[start_idx:end_idx] if self.time is not None else None
        self._update_num_samples()

    def rotate(self, axis: str, angle_deg: float) -> None:
        """Rotate force, moment, CoP, position and orientation about an axis.

        Args:
            axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``).
            angle_deg: Rotation angle in degrees.
        """
        rotation_matrix = get_rotation_matrix(axis, angle_deg)
        self.force = rotation_matrix @ self.force
        self.moment = rotation_matrix @ self.moment
        self.cop = rotation_matrix @ self.cop
        self.position = rotation_matrix @ self.position
        self.rotation = rotation_matrix @ self.rotation

    def convert_units(self, target_unit: str) -> None:
        """Convert position-derived quantities to a target length unit in place.

        Scales moment, CoP and ``Tz`` and updates the unit labels. Moments are
        typically in ``N·mm`` and CoP in ``mm``. No-op if already in
        ``target_unit``.

        Args:
            target_unit: Desired length unit (``'mm'`` or ``'m'``).

        Raises:
            ValueError: If the requested conversion is not supported.
        """
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
        """Plot force, moment and centre of pressure against time in three subplots."""
        import matplotlib.pyplot as plt
        
        time = self.time if self.time is not None else np.arange(self.force.shape[1]) / self.sampling_rate if \
                                                         self.sampling_rate else np.arange(self.force.shape[1])

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
        """Force along the X axis, shape ``(n_samples,)``."""
        return self.force[0, :]

    @property
    def Fy(self) -> np.ndarray:
        """Force along the Y axis, shape ``(n_samples,)``."""
        return self.force[1, :]

    @property
    def Fz(self) -> np.ndarray:
        """Force along the Z axis, shape ``(n_samples,)``."""
        return self.force[2, :]

    @property
    def cop_x(self) -> np.ndarray:
        """Centre of pressure along the X axis, shape ``(n_samples,)``."""
        return self.cop[0, :]

    @property
    def cop_y(self) -> np.ndarray:
        """Centre of pressure along the Y axis, shape ``(n_samples,)``."""
        return self.cop[1, :]

    @property
    def cop_z(self) -> np.ndarray:
        """Centre of pressure along the Z axis, shape ``(n_samples,)``."""
        return self.cop[2, :]

    @property
    def Mx(self) -> np.ndarray:
        """Moment about the X axis, shape ``(n_samples,)``."""
        return self.moment[0, :]

    @property
    def My(self) -> np.ndarray:
        """Moment about the Y axis, shape ``(n_samples,)``."""
        return self.moment[1, :]

    @property
    def Mz(self) -> np.ndarray:
        """Moment about the Z axis, shape ``(n_samples,)``."""
        return self.moment[2, :]

    @property
    def x(self) -> np.ndarray:
        """Plate origin position along the X axis, shape ``(n_samples,)``."""
        return self.position[0, :]

    @property
    def y(self) -> np.ndarray:
        """Plate origin position along the Y axis, shape ``(n_samples,)``."""
        return self.position[1, :]

    @property
    def z(self) -> np.ndarray:
        """Plate origin position along the Z axis, shape ``(n_samples,)``."""
        return self.position[2, :]

    @property
    def unit_position(self) -> str:
        """Unit of the plate position data (alias of :attr:`unit_cop`)."""
        return self.unit_cop

    @property
    def data(self) -> np.ndarray:
        """Return the force, moment and CoP as a single array of shape ``(9, n_samples)``."""
        return np.vstack((self.force, self.moment, self.cop))

    def _update_num_samples(self) -> None:
        """Update the cached number of samples based on the current data."""
        self.num_samples = self.force.shape[1]
        
    def __repr__(self) -> str:
        """Return a concise summary of the force plate's contents."""
        return (
            f"ForceData(name={self.name!r}, samples={self.num_samples}, "
            f"unit_force={self.unit_force!r}, unit_moment={self.unit_moment!r}, "
            f"unit_cop={self.unit_cop!r}, sampling_rate={self.sampling_rate})"
        )

    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

    def __array__(self):
        """Allow the object to be converted to a NumPy array."""
        return self.data
    
    def __getitem__(self, index):
        """Allow indexing into the data."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Allow setting values in the data."""
        self.data[index] = value

    def __len__(self):
        """Return the number of samples in the data."""
        return len(self.data)

    def __iter__(self):
        """Allow iteration over the data."""
        return iter(self.data)
