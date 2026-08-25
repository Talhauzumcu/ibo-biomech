"""
Bare minimum data container for biomechanical data. Currently used for storing opensim ID and IK results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Data:
    """Processing methods operate **in place**.
    
    Attributes:
        name: Name of the data container.
        data: Signal samples, shape ``(n_samples,)``.
        unit: Physical unit of the signal (e.g. ``"V"``).
        time: Time vector corresponding to the data samples, shape ``(n_samples,)``.
        metadata: Information from the header section of .sto or .mot files, stored as a dictionary.
    """

    name: str
    data: np.ndarray
    unit: str = ""
    time: Optional[np.ndarray] = None
    sampling_rate: Optional[float] = None
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.time is not None:
            if len(self.time) != len(self.data):
                raise ValueError("Time vector length must match data length.")

            if self._is_uniformly_sampled(self.time):
                self.sampling_rate = 1 / np.mean(np.diff(self.time))
            else:
                self.sampling_rate = None

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the signal in place to ``[start_idx, end_idx)``.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """
        self.data = self.data[start_idx:end_idx]

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase low-pass Butterworth filter in place.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            if self.time is not None:
                if not self._is_uniformly_sampled(self.time):
                    raise ValueError("Time vector is not uniformly sampled. Cannot compute sampling rate.")
                self.sampling_rate = 1 / np.mean(np.diff(self.time))
            else:
                raise ValueError("Sampling rate or a uniform time vector must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        self.data = filtfilt(b, a, self.data, axis=0)

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase high-pass Butterworth filter in place.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            if self.time is not None:
                if not self._is_uniformly_sampled(self.time):
                    raise ValueError("Time vector is not uniformly sampled. Cannot compute sampling rate.")
                self.sampling_rate = 1 / np.mean(np.diff(self.time))
            else:
                raise ValueError("Sampling rate or a uniform time vector must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        self.data = filtfilt(b, a, self.data, axis=0)

    def plot(self) -> None:
        """Plot the signal against time."""
        import matplotlib.pyplot as plt

        if self.time is not None:
            time = self.time
        else:
            time = np.arange(self.data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.data.shape[0])

        plt.figure(figsize=(12, 6))
        plt.plot(time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Signal ({self.unit})')
        plt.title(f'Analog Signal: {self.name}')
        plt.show()

    def _is_uniformly_sampled(self, time: np.ndarray) -> bool:
        """Check if the time vector is uniformly sampled."""
        dt = np.diff(time)
        return np.allclose(dt, dt[0], atol=1e-6)
    
    def __repr__(self) -> str:
        """Return a concise summary of the data."""
        return (
            f"Data(name={self.name!r}, samples={self.data.size}, "
            f"unit={self.unit!r})"
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

    def __add__(self, other):
        """Allow addition with another Data object or a scalar."""
        if isinstance(other, Data):
            return Data(
                name=self.name,
                data=self.data + other.data,
                unit=self.unit,
                time=self.time
            )
        else:
            return Data(
                name=self.name,
                data=self.data + other,
                unit=self.unit,
                time=self.time
            )

    def __sub__(self, other):
        """Allow subtraction with another Data object or a scalar."""
        if isinstance(other, Data):
            return Data(
                name=self.name,
                data=self.data - other.data,
                unit=self.unit,
                time=self.time
            )
        else:
            return Data(
                name=self.name,
                data=self.data - other,
                unit=self.unit,
                time=self.time
            )

    def __mul__(self, other):
        """Allow multiplication with another Data object or a scalar."""
        if isinstance(other, Data):
            return Data(
                name=self.name,
                data=self.data * other.data,
                unit=self.unit,
                time=self.time
            )
        else:
            return Data(
                name=self.name,
                data=self.data * other,
                unit=self.unit,
                time=self.time
            )

    def __truediv__(self, other):
        """Allow division with another Data object or a scalar."""
        if isinstance(other, Data):
            return Data(
                name=self.name,
                data=self.data / other.data,
                unit=self.unit,
                time=self.time
            )
        else:
            return Data(
                name=self.name,
                data=self.data / other,
                unit=self.unit,
                time=self.time
            )
