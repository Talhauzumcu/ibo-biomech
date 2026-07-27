"""3D marker trajectory container.

This module defines :class:`MarkerData`, the container used throughout the
library to hold a single labelled marker's 3D trajectory together with helpers
for filtering, cropping, rotating, and unit conversion.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from ibo_biomech.utils.utils import *
from dataclasses import dataclass, field

@dataclass
class MarkerData:
    """A single marker's 3D trajectory over time.

    Positions are stored as three separate 1D arrays (one per axis), each of
    length ``n_samples``. All processing methods operate **in place**.

    Attributes:
        name: Marker label (e.g. ``"R_Knee"``).
        x: Position along the X axis, shape ``(n_samples,)``.
        y: Position along the Y axis, shape ``(n_samples,)``.
        z: Position along the Z axis, shape ``(n_samples,)``.
        unit: Length unit of the positions (e.g. ``"mm"`` or ``"m"``).
        sampling_rate: Sampling frequency in Hz. Required for filtering.
        virtual: ``1`` if the marker was computed/derived, ``0`` if measured.
    """
    name: str
    x: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z: np.ndarray = field(default_factory=lambda: np.zeros(1))
    unit: str = 'mm'
    sampling_rate: float = None
    virtual: int = 0 #Whether the marker is virtual or measured.

    def get_trajectory(self) -> np.ndarray:
        """Return the trajectory stacked as a single array.

        Returns:
            Array of shape ``(3, n_samples)`` with rows ``x``, ``y``, ``z``.
        """
        return np.column_stack([self.x, self.y, self.z]).T

    def get_magnitude(self) -> np.ndarray:
        """Return the Euclidean magnitude of the position vector per sample.

        Returns:
            Array of shape ``(n_samples,)`` with ``sqrt(x**2 + y**2 + z**2)``.
        """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def get_frame_trajectory(self) -> np.ndarray:
        """Return the trajectory in homogeneous coordinates.

        Returns:
            Array of shape ``(4, n_samples)`` with rows ``x``, ``y``, ``z`` and a
            row of ones, suitable for multiplication with 4x4 transforms.
        """
        n_samples = self.x.shape[0]
        ones = np.ones(n_samples)
        return np.vstack([self.x, self.y, self.z, ones])

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase low-pass Butterworth filter to all three axes.

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
        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)
        self.z = filtfilt(b, a, self.z)

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase high-pass Butterworth filter to all three axes.

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
        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)
        self.z = filtfilt(b, a, self.z)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the trajectory in place to the half-open range ``[start_idx, end_idx)``.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """
        self.x=self.x[start_idx:end_idx]
        self.y=self.y[start_idx:end_idx]
        self.z=self.z[start_idx:end_idx]

    def rotate(self, axis: str, angle_deg: float) -> None:
        """Rotate the trajectory in place about a coordinate axis.

        Args:
            axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``).
            angle_deg: Rotation angle in degrees.
        """
        rotation_matrix = get_rotation_matrix(axis, angle_deg)
        trajectory = self.get_trajectory()  # (3, n_samples)
        rotated_traj = rotation_matrix @ trajectory  # (3, n_samples)
        self.x = rotated_traj[0, :]
        self.y = rotated_traj[1, :]
        self.z = rotated_traj[2, :]

    def convert_units(self, target_unit: str) -> None:
        """Convert the trajectory to a target length unit in place.

        Supported conversions are between ``'mm'`` and ``'m'``. No-op if the
        data is already in ``target_unit``.

        Args:
            target_unit: Desired unit (``'mm'`` or ``'m'``).

        Raises:
            ValueError: If the requested conversion is not supported.
        """
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
        """Plot the X, Y and Z trajectories against time in a single figure."""
        import matplotlib.pyplot as plt
        time = np.arange(len(self.x)) / self.sampling_rate if self.sampling_rate else np.arange(len(self.x))

        plt.figure(figsize=(12, 8))
        plt.plot(time, self.get_trajectory().T)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title(f'Marker: {self.name}')
        plt.legend(['X', 'Y', 'Z'])
        plt.show()

    @property
    def data(self) -> np.ndarray:
        """Return the trajectory as a single array of shape ``(3, n_samples)``."""
        return self.get_trajectory()
    
    def __add__(self, other: MarkerData) -> MarkerData:
        """Add two markers element-wise, returning a new virtual marker.

        Useful for computing midpoints together with :meth:`__truediv__`.

        Args:
            other: The marker to add. Must have the same number of samples.

        Returns:
            A new :class:`MarkerData` with ``virtual=1``.

        Raises:
            ValueError: If the two markers have different shapes.
        """
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
        """Divide the marker by another marker or a scalar, element-wise.

        Args:
            other: Another :class:`MarkerData` of matching shape, or a non-zero
                scalar.

        Returns:
            A new :class:`MarkerData` with ``virtual=1``, or ``NotImplemented``
            if ``other`` is an unsupported type.

        Raises:
            ValueError: If dividing by a marker of a different shape.
            ZeroDivisionError: If dividing by the scalar zero.
        """
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

    def __repr__(self) -> str:
        """Return a concise summary of the marker's contents."""
        samples = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
        return (
            f"MarkerData(name={self.name!r}, samples={samples}, unit={self.unit!r}, "
            f"sampling_rate={self.sampling_rate}, virtual={self.virtual})"
        )

    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()