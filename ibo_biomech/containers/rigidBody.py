"""Rigid-body position and orientation trajectories."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ibo_biomech.utils.utils import get_rotation_matrix, validate_crop_range
from ._validation import validate_clock


@dataclass
class RigidBody:
    """Pose of a rigid body, with the same sample-last layout as ForceData.

    Attributes:
        name: Body name.
        markers: List of names identifying the markers on the body.
        position: Global position, shape ``(3, n_samples)``.
        rotation: Local-to-global orientation, shape ``(3, 3, n_samples)``.
            A static ``(3, 3)`` matrix or a single sample is broadcast in time.
            Omitted orientations are unknown (NaN).
        num_samples: Sample count, derived from position.
        RPY: Read-only roll, pitch, yaw in degrees, shape ``(3, n_samples)``.
            Uses extrinsic XYZ: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
        unit: Position unit, default ``'mm'``.
        sampling_rate: Optional sampling frequency in Hz.
        time: Optional sample timestamps in seconds.
    """

    name: str
    markers: List[str] = field(default_factory=list)
    position: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    rotation: Optional[np.ndarray] = None
    unit: str = 'mm'
    sampling_rate: Optional[float] = None
    time: Optional[np.ndarray] = None
    num_samples: int = field(init=False)

    def __post_init__(self):
        self.markers = list(self.markers)
        self._initialize_position()
        self.set_rotation(self.rotation)
        self._initialize_time()

    def _initialize_position(self) -> None:
        """Copy the position trajectory and derive its sample count."""
        self.position = np.array(self.position, dtype=float, copy=True)
        if self.position.ndim != 2 or self.position.shape[0] != 3 or not self.position.shape[1]:
            raise ValueError('position must have shape (3, n_samples), n_samples > 0.')
        self.num_samples = self.position.shape[1]
        self._validate_pose_array('position', self.position, (3, self.num_samples))

    def _initialize_time(self) -> None:
        """Copy or generate timestamps and validate the sample clock."""
        if self.time is not None:
            self.time = np.array(self.time, dtype=float, copy=True)
        elif self.sampling_rate is not None:
            validate_clock(None, self.num_samples, self.sampling_rate)
            self.time = np.arange(self.num_samples) / self.sampling_rate
        validate_clock(self.time, self.num_samples, self.sampling_rate)

    def set_rotation(self, rotation: Optional[np.ndarray]) -> None:
        """Copy and validate new orientations before replacing the current ones.

        Accepts ``(3, 3)``, ``(3, 3, 1)`` or ``(3, 3, num_samples)``;
        single matrices are broadcast to the existing sample count. ``None``
        resets all orientations to unknown (NaN). Invalid input leaves the
        current rotation unchanged. RPY reflects the new rotation on access.
        """
        if rotation is None:
            rotation = np.full((3, 3, self.num_samples), np.nan)
        else:
            rotation = np.array(rotation, dtype=float, copy=True)
            if rotation.shape == (3, 3):
                rotation = rotation[..., None]
            if rotation.shape == (3, 3, 1):
                rotation = np.repeat(rotation, self.num_samples, axis=-1)
        self._validate_rotation(rotation)
        self.rotation = rotation

    @staticmethod
    def _validate_pose_array(name: str, value: np.ndarray, shape: tuple) -> None:
        """Require the expected shape and finite or wholly unknown samples."""
        value = np.asarray(value)
        if value.shape != shape:
            raise ValueError(f'{name} must have shape {shape}, got {value.shape}.')
        samples = value.reshape(-1, shape[-1])
        if not np.all(np.isfinite(samples).all(axis=0) | np.isnan(samples).all(axis=0)):
            raise ValueError(f'{name} must be finite or wholly unknown (NaN) per sample.')

    def _validate_rotation(self, rotation: np.ndarray) -> None:
        """Check orientation shape, missing samples and proper rotations."""
        self._validate_pose_array('rotation', rotation, (3, 3, self.num_samples))
        rotations = np.moveaxis(rotation, -1, 0)
        rotations = rotations[np.isfinite(rotations).all(axis=(1, 2))]
        if len(rotations) and (
            not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6)
        ):
            raise ValueError('rotation must contain proper orthonormal matrices (determinant +1).')

    def validate(self):
        """Validate pose shapes, proper rotations and the optional clock.

        Each position/orientation sample must be finite or entirely NaN.
        """
        self._validate_pose_array('position', self.position, (3, self.num_samples))
        self._validate_rotation(self.rotation)
        return validate_clock(self.time, self.num_samples, self.sampling_rate)

    @property
    def RPY(self) -> np.ndarray:
        """Derive roll/pitch/yaw in degrees from the current rotation matrices.

        Unknown orientations remain NaN. At gimbal lock SciPy warns and sets
        yaw to zero, returning angles that still represent the same rotation.
        The returned array is read-only; edit ``rotation`` to change the pose.
        """
        self.validate()
        rotations = np.moveaxis(self.rotation, -1, 0)
        valid = np.isfinite(rotations).all(axis=(1, 2))
        angles = np.full((3, self.num_samples), np.nan)
        if valid.any():
            angles[:, valid] = Rotation.from_matrix(rotations[valid]).as_euler('xyz', degrees=True).T
        angles.setflags(write=False)
        return angles

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the pose and clock in place to ``[start_idx, end_idx)``."""
        self.validate()
        validate_crop_range(start_idx, end_idx, self.num_samples)
        self.position = self.position[:, start_idx:end_idx].copy()
        self.rotation = self.rotation[..., start_idx:end_idx].copy()
        self.time = self.time[start_idx:end_idx].copy() if self.time is not None else None
        self.num_samples = self.position.shape[1]

    def rotate(self, axis: str, angle_deg: float) -> None:
        """Rotate position and orientation about a global coordinate axis."""
        self.validate()
        if not np.isfinite(angle_deg):
            raise ValueError('Rotation angle must be finite.')
        matrix = get_rotation_matrix(axis, angle_deg)
        self.position = matrix @ self.position
        self.rotation = np.einsum('ij,jkn->ikn', matrix, self.rotation)

    def convert_units(self, target_unit: str) -> None:
        """Convert position between millimetres and metres in place."""
        self.validate()
        if self.unit == target_unit:
            return
        factors = {('mm', 'm'): .001, ('m', 'mm'): 1000.}
        if (self.unit, target_unit) not in factors:
            raise ValueError(f'Unsupported unit conversion: {self.unit} to {target_unit}')
        self.position *= factors[self.unit, target_unit]
        self.unit = target_unit


    def plot(self) -> None:
        """Plot the position and orientation trajectories in three subplots."""
        import matplotlib.pyplot as plt

        self.validate()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(self.time, self.position.T)
        axes[0].set_ylabel(f'Position ({self.unit})')
        axes[0].legend(['X', 'Y', 'Z'])
        axes[1].plot(self.time, self.RPY.T)
        axes[1].set_ylabel('Orientation (deg)')
        axes[1].legend(['Roll', 'Pitch', 'Yaw'])
        plt.tight_layout()
        plt.show()

    @property
    def roll(self) -> np.ndarray:
        """Read-only roll in degrees, shape ``(n_samples,)``."""
        return self.RPY[0]

    @property
    def pitch(self) -> np.ndarray:
        """Read-only pitch in degrees, shape ``(n_samples,)``."""
        return self.RPY[1]

    @property
    def yaw(self) -> np.ndarray:
        """Read-only yaw in degrees, shape ``(n_samples,)``."""
        return self.RPY[2]

    def __repr__(self) -> str:
        return (f'RigidBody(name={self.name!r}, markers={len(self.markers)}, '
                f'samples={self.num_samples}, unit={self.unit!r}, '
                f'sampling_rate={self.sampling_rate})')

    def __str__(self) -> str:
        return self.__repr__()


rigidBody = RigidBody
