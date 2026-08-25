"""Generic analog signal container.

This module defines :class:`AnalogData`, a lightweight container for a single
analog channel (EMG, raw force-plate voltage, etc.) sampled at a fixed rate.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class AnalogData:
    """A single analog channel sampled at a fixed rate.

    Processing methods operate **in place**.

    Attributes:
        name: Channel label.
        data: Signal samples, shape ``(n_samples,)``.
        sampling_rate: Sampling frequency in Hz. Required for filtering.
        unit: Physical unit of the signal (e.g. ``"V"``).
        channel: Hardware channel index this signal came from, if known.
    """
    name: str
    data: np.ndarray
    sampling_rate: float = None
    unit: str = ""
    channel: Optional[int] = None

    def get_data(self) -> np.ndarray:
        """Return the raw signal samples.

        Returns:
            The underlying ``data`` array.
        """
        return self.data

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
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
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
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        self.data = filtfilt(b, a, self.data, axis=0)

    def plot(self) -> None:
        """Plot the signal against time."""
        import matplotlib.pyplot as plt
        time = np.arange(self.data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.data.shape[0])

        plt.figure(figsize=(12, 6))
        plt.plot(time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Signal ({self.unit})')
        plt.title(f'Analog Signal: {self.name}')
        plt.show()

    def __repr__(self) -> str:
        """Return a concise summary of the analog channel."""
        return (
            f"AnalogData(name={self.name!r}, samples={self.data.size}, "
            f"sampling_rate={self.sampling_rate}, unit={self.unit!r}, "
            f"channel={self.channel})"
        )

    def __array__(self):
        """Allow the object to be converted to a NumPy array."""
        return self.data
    
    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

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
