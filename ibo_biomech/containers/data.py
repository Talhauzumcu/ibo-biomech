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
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)
    
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

    def __repr__(self) -> str:
        """Return a concise summary of the data."""
        return (
            f"Data(name={self.name!r}, samples={self.data.size}, "
            f"unit={self.unit!r})"
        )
    
    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

    