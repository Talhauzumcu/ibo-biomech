"""Force plate data container.

This module defines :class:`ForceData`, which holds the force, moment and
centre-of-pressure signals of a single force plate together with helpers for
filtering, cropping, rotating and unit conversion.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ibo_biomech.utils.utils import *
from ._mixins import ArrayLikeMixin
from ._validation import validate_clock

@dataclass
class ForceData(ArrayLikeMixin):
    """Signals and geometry for a single force plate.

    Most processing methods operate **in place**. Convenience properties
    (:attr:`Fx`, :attr:`cop_x`, :attr:`Mx`, ...) expose individual rows of the
    stacked arrays.

    Attributes:
        name: Force plate name (e.g. ``"forceplate_0"``).
        force: Force components ``Fx, Fy, Fz``, shape ``(3, n_samples)``.
        moment: Moment components ``Mx, My, Mz``, shape ``(3, n_samples)``.
        cop: Centre of pressure ``x, y, z``, shape ``(3, n_samples)``.
        corners: Plate corner coordinates, shape ``(3, 4, n_samples)``.
        position: Plate center position in 3D mocap coordinate system, shape ``(3, n_samples)``.
        rotation: Plate orientation matrices, shape ``(3, 3, n_samples)``.
        origin: Origin offset relative to the corners, shape ``(3, 1)``.
        Tz: Full moment-at-CoP vector, shape ``(3, n_samples)``, expressed
            in the same declared frame as force and CoP (not a scalar Z value).
        Missing geometry: Stored as NaN; finite rotation matrices are validated.
        coordinateSystem: ``1`` if data is in global coordinates, ``0`` if local.
        metadata: Raw plate metadata (units, calibration matrix, corners, ...).
        sampling_rate: Sampling frequency in Hz. Required for filtering.
        num_samples: Number of samples; derived in :meth:`__post_init__`.
        unit_force: Force unit, derived from ``metadata``.
        unit_moment: Moment unit, derived from ``metadata``.
        unit_cop: Centre-of-pressure unit, derived from ``metadata``.
    """
    name: str
    force: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    moment: Optional[np.ndarray] = None
    cop: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    origin: Optional[np.ndarray] = None
    Tz: Optional[np.ndarray] = None
    coordinateSystem: int = 1
    metadata: Dict = field(default_factory=dict)
    sampling_rate: Optional[float] = None
    time: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize absent signals at full length and mark unknown geometry NaN.

        Static geometry may be supplied without a time axis or with one sample.
        It is expanded to the signal length. Free moments must always contain
        three components; scalar free moments are not accepted.
        """
        self.force = np.array(self.force, dtype=float, copy=True)
        if self.force.ndim != 2 or self.force.shape[0] != 3 or not self.force.shape[1]:
            raise ValueError('Force array must have shape (3, n_samples), n_samples > 0.')
        self.num_samples = self.force.shape[1]
        self.metadata = dict(self.metadata)
        n = self.num_samples
        for name in ('moment', 'cop', 'Tz'):
            value = getattr(self, name)
            setattr(self, name, np.zeros((3, n)) if value is None else np.array(value, dtype=float, copy=True))
        for name, shape in [('corners', (3, 4)), ('position', (3,)), ('rotation', (3, 3))]:
            value = getattr(self, name)
            if value is None:
                value = np.full((*shape, n), np.nan)
            else:
                value = np.array(value, dtype=float, copy=True)
                if value.shape == shape:
                    value = value[..., None]
                if value.shape == (*shape, 1):
                    value = np.repeat(value, n, axis=-1)
            setattr(self, name, value)
        self.origin = (np.full((3, 1), np.nan) if self.origin is None
                       else np.array(self.origin, dtype=float, copy=True).reshape(3, 1))
        if self.time is not None:
            self.time = np.array(self.time, dtype=float, copy=True)
        elif self.sampling_rate is not None:
            validate_clock(None, n, self.sampling_rate)
            self.time = np.arange(n) / self.sampling_rate
        self.clean_nan()
        self._parse_metadata()
        self.validate()

    def _parse_metadata(self) -> None:
        self.unit_force = self.metadata.get('unit_force', 'Unknown')
        self.unit_moment = self.metadata.get('unit_moment', 'Unknown')
        self.unit_cop = self.metadata.get('unit_position', 'Unknown')

    def validate(self):
        """Reject malformed signals/geometry before a processing operation.

        A wholly NaN geometry sample means unknown, never an identity/zero pose.
        Supplied orientations must be proper orthonormal rotation matrices.
        """
        n = self.num_samples
        for name, shape in [('force', (3, n)), ('moment', (3, n)), ('cop', (3, n)),
                            ('Tz', (3, n)), ('position', (3, n)), ('corners', (3, 4, n)),
                            ('rotation', (3, 3, n)), ('origin', (3, 1))]:
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f'{name} must have shape {shape}, got {value.shape}.')
            if name in ('force', 'moment', 'cop', 'Tz'):
                if not np.all(np.isfinite(value)):
                    raise ValueError(f'{name} contains nonfinite values; clean them before processing.')
            else:
                samples = value.reshape(-1, value.shape[-1])
                valid = np.all(np.isfinite(samples), axis=0) | np.all(np.isnan(samples), axis=0)
                if not np.all(valid):
                    raise ValueError(f'{name} must be finite or wholly unknown (NaN) per sample.')
        rotations = np.moveaxis(self.rotation, -1, 0)
        rotations = rotations[np.isfinite(rotations).all(axis=(1, 2))]
        if len(rotations) and (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6)
                               or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6)):
            raise ValueError('rotation must contain proper orthonormal matrices (determinant +1).')
        if self.coordinateSystem not in (0, 1):
            raise ValueError('coordinateSystem must be 0 (local) or 1 (global).')
        return validate_clock(self.time, n, self.sampling_rate)

    def get_force_magnitude(self) -> np.ndarray:
        """Return the force magnitude at each sample."""
        return np.linalg.norm(self.force, axis=0)

    def clean_nan(self) -> None:
        """Replace nonfinite signal values with zeros; preserve unknown geometry."""
        for name in ('force', 'moment', 'cop', 'Tz'):
            setattr(self, name, np.nan_to_num(getattr(self, name), nan=0., posinf=0., neginf=0.))

    def _filter(self, cutoff, order, btype):
        rate = self.validate()
        values = {name: apply_filter(getattr(self, name), rate, cutoff, order, btype=btype, axis=1)
                  for name in ('force', 'moment', 'cop', 'Tz')}
        for name, value in values.items():
            setattr(self, name, value)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Low-pass force, moment, CoP and free moment; leave geometry unchanged."""
        self._filter(cutoff, order, 'low')

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """High-pass force, moment, CoP and free moment; leave geometry unchanged."""
        self._filter(cutoff, order, 'high')

    def filter_low_forces(self, threshold: float = 10.0) -> None:
        """Zero all signals at unloaded samples, without changing plate geometry."""
        self.validate()
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('Threshold must be finite and nonnegative.')
        mask = self.get_force_magnitude() < threshold
        values = {name: np.where(mask[None, :], 0., getattr(self, name))
                  for name in ('force', 'moment', 'cop', 'Tz')}
        for name, value in values.items():
            setattr(self, name, value)

    def downsample(self, factor: int) -> None:
        """Anti-alias signals; sample clocks and geometry at original indices.

        Uniform timestamps and a consistent positive rate are required. A rate
        may be inferred from a supplied clock. With neither, time stays unknown.
        Rejected operations leave every array and metadata field unchanged.
        """
        if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)) or factor <= 0:
            raise ValueError('Downsampling factor must be a positive integer.')
        rate = self.validate()
        if factor == 1:
            return
        from scipy.signal import decimate
        values = {name: decimate(getattr(self, name), int(factor), axis=1, ftype='fir', zero_phase=True)
                  for name in ('force', 'moment', 'cop', 'Tz')}
        values.update({name: getattr(self, name)[..., ::factor].copy()
                       for name in ('corners', 'position', 'rotation')})
        time = self.time[::factor].copy() if self.time is not None else None
        for name, value in values.items():
            setattr(self, name, value)
        self.time = time
        self.sampling_rate = rate / factor if rate is not None else None
        self._update_num_samples()

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop every sampled array to [start_idx, end_idx)."""
        self.validate()
        validate_crop_range(start_idx, end_idx, self.num_samples)
        for name in ('force', 'moment', 'cop', 'corners', 'position', 'rotation', 'Tz'):
            setattr(self, name, getattr(self, name)[..., start_idx:end_idx].copy())
        self.time = self.time[start_idx:end_idx].copy() if self.time is not None else None
        self._update_num_samples()

    def rotate(self, axis: str, angle_deg: float) -> None:
        """Rotate all vectors/geometry in their declared frame; origin stays local."""
        self.validate()
        if not np.isfinite(angle_deg):
            raise ValueError('Rotation angle must be finite.')
        matrix = get_rotation_matrix(axis, angle_deg)
        values = {name: self._rotate(matrix, getattr(self, name))
                  for name in ('force', 'moment', 'cop', 'position', 'rotation', 'Tz', 'corners')}
        for name, value in values.items():
            setattr(self, name, value)

    def _rotate(self, matrix, data):
        return np.einsum('ij,j...->i...', matrix, data)

    def convert_units(self, target_unit: str) -> None:
        """Convert all length-derived quantities, including geometry/free moment."""
        self.validate()
        if self.unit_cop == target_unit:
            return
        factors = {('mm', 'm'): .001, ('m', 'mm'): 1000.}
        if (self.unit_cop, target_unit) not in factors:
            raise ValueError(f'Unsupported unit conversion: {self.unit_cop} to {target_unit}')
        factor = factors[self.unit_cop, target_unit]
        values = {name: getattr(self, name) * factor
                  for name in ('moment', 'cop', 'Tz', 'position', 'corners', 'origin')}
        for name, value in values.items():
            setattr(self, name, value)
        self.unit_moment, self.unit_cop = f'N{target_unit}', target_unit
        self.metadata.update(unit_moment=self.unit_moment, unit_position=self.unit_cop)

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
