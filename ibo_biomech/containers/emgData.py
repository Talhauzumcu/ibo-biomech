"""EMG signal container.

This module defines :class:`EMGData`, a container for a single electromyography
channel with a standard processing pipeline (high-pass filter, rectify,
low-pass envelope, normalise) exposed via :meth:`EMGData.process_emg`.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class EMGData:
    """A single EMG channel and its processing pipeline.

    Unlike most containers in this library, the filtering helpers here **return**
    new arrays rather than mutating ``data`` in place, so they can be chained
    inside :meth:`process_emg`. The processed envelope is computed lazily and
    cached via the :attr:`processed_data` property.

    Attributes:
        name: Channel label (e.g. ``"EMG_VastusLat"``).
        data: Raw EMG samples, shape ``(n_samples,)``.
        sampling_rate: Sampling frequency in Hz. Required for filtering.
        unit: Physical unit of the signal.
        channel: Hardware channel index this signal came from, if known.
    """
    name: str
    data: np.ndarray
    sampling_rate: float = None
    unit: str = "unknown"
    channel: Optional[int] = None

    def get_raw_data(self) -> np.ndarray:
        """Return the raw, unprocessed EMG samples.

        Returns:
            The underlying ``data`` array.
        """
        return self.data

    def clean_nan(self) -> None:
        """Replace NaN values in the raw signal with zeros, in place."""
        self.data = np.nan_to_num(self.data)

    def process_emg(self) -> np.ndarray:
        """Run the standard EMG processing pipeline.

        Steps: clean NaNs, 2nd-order high-pass at 30 Hz, square (rectify),
        2nd-order low-pass envelope at 10 Hz, then normalise to the peak.

        Returns:
            The processed, peak-normalised EMG envelope, shape ``(n_samples,)``.
        """
        self.clean_nan()
        filtered_signal = self._highpass_filter(cutoff=30, order=2)
        rectified_signal = np.power(filtered_signal, 2)
        enveloped_signal = self._envelope_signal(rectified_signal, cutoff=10, order=2)
        normalized_signal = self._normalize_emg(enveloped_signal)
        return normalized_signal

    def _envelope_signal(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Low-pass filter a signal to extract its envelope.

        Args:
            signal: Rectified signal to envelope.
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Returns:
            The enveloped (low-pass filtered) signal.

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        return filtfilt(b, a, signal, axis=0)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> np.ndarray:
        """Lowpass filter the EMG data in place. 

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

    def highpass_filter(self, cutoff: float, order: int = 4) -> np.ndarray:
        """highpass filter the EMG data in place. 

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

    def _highpass_filter(self, cutoff: float, order: int = 4) -> np.ndarray:
        """Return a zero-phase high-pass Butterworth filtered copy of the signal.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.

        Returns:
            The filtered signal (the container's ``data`` is left unchanged).

        Raises:
            ValueError: If ``sampling_rate`` is not set.
        """
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        return filtfilt(b, a, self.data, axis=0)

    @staticmethod
    def _normalize_emg(emg_signal):
        """Normalise a signal to its peak amplitude.

        Args:
            emg_signal: Signal to normalise.

        Returns:
            The signal divided by its maximum value, or the signal unchanged if
            the maximum is zero.
        """
        max_amplitude = np.max(emg_signal)
        normalized_signal = emg_signal / max_amplitude if max_amplitude != 0 else emg_signal

        return normalized_signal

    def plot(self) -> None:
        """Plot the raw EMG signal against time."""
        import matplotlib.pyplot as plt
        time = np.arange(self.data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.data.shape[0])

        plt.figure(figsize=(12, 6))
        plt.plot(time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Signal ({self.unit})')
        plt.title(f'Analog Signal: {self.name}')
        plt.show()

    def plot_processed(self) -> None:
        """Plot the processed EMG envelope against time."""
        import matplotlib.pyplot as plt
        time = np.arange(self.processed_data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.processed_data.shape[0])

        plt.figure(figsize=(12, 6))
        plt.plot(time, self.processed_data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Processed Signal ({self.unit})')
        plt.title(f'Processed EMG Signal: {self.name}')
        plt.show()

    def crop(self, start_idx: int, end_idx: int) -> None:
            """Crop the signal in place to ``[start_idx, end_idx)``.
    
            Args:
                start_idx: First sample index to keep.
                end_idx: First sample index to drop (exclusive).
            """
            self.data = self.data[start_idx:end_idx]
            
    @property
    def processed_data(self) -> np.ndarray:
        """The processed EMG envelope, computed once and cached.

        On first access this runs :meth:`process_emg` and caches the result for
        subsequent accesses.

        Returns:
            The processed, peak-normalised EMG envelope.
        """
        if not hasattr(self, '_processed_data'):
            self._processed_data = self.process_emg()
        return self._processed_data

    def __repr__(self) -> str:
        """Return a concise summary of the EMG channel's contents."""
        return (
            f"EMGData(name={self.name!r}, samples={self.data.size}, "
            f"sampling_rate={self.sampling_rate}, unit={self.unit!r}, "
            f"channel={self.channel})"
        )

    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()