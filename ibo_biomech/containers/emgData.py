import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class EMGData:
    """Data class for EMG signals."""
    name: str
    data: np.ndarray
    sampling_rate: float = None
    unit: str = "unknown"
    channel: Optional[int] = None

    def get_raw_data(self) -> np.ndarray:
        """Returns the EMG signal data."""
        return self.data
    
    def clean_nan(self) -> None:
        """Replace NaN values in EMG data with zeros."""
        self.data = np.nan_to_num(self.data)

    def process_emg(self) -> np.ndarray:
        self.clean_nan()
        # filtered_signal = self.highpass_filter(self.data, cutoff=30, order=2)
        rectified_signal = np.power(self.data, 2)
        enveloped_signal = self._envelope_signal(rectified_signal, cutoff=6, order=2)
        normalized_signal = self._normalize_emg(enveloped_signal)
        return normalized_signal
    
    def _envelope_signal(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply low-pass Butterworth filter to the analog signal."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        return filtfilt(b, a, signal, axis=0)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply low-pass Butterworth filter to the analog signal."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply low-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='low')
        return filtfilt(b, a, self.data, axis=0)
    
    def highpass_filter(self, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply high-pass Butterworth filter to the analog signal."""
        from scipy.signal import butter, filtfilt
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set to apply high-pass filter.")
        b, a = butter(order, cutoff / (0.5 * self.sampling_rate), btype='high')
        return filtfilt(b, a, self.data, axis=0)
    
    @staticmethod
    def _normalize_emg(emg_signal):
        max_amplitude = np.max(emg_signal)
        normalized_signal = emg_signal / max_amplitude if max_amplitude != 0 else emg_signal

        return normalized_signal
    
    def plot(self) -> None:
        """Plot the analog signal."""
        import matplotlib.pyplot as plt
        time = np.arange(self.data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.data.shape[0])
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Signal ({self.unit})')
        plt.title(f'Analog Signal: {self.name}')
        plt.show()

    def plot_processed(self) -> None:
        """Plot the processed EMG signal."""
        import matplotlib.pyplot as plt
        time = np.arange(self.processed_data.shape[0]) / self.sampling_rate if self.sampling_rate else np.arange(self.processed_data.shape[0])
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, self.processed_data)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Processed Signal ({self.unit})')
        plt.title(f'Processed EMG Signal: {self.name}')
        plt.show()

    @property
    def processed_data(self) -> np.ndarray:
        """Returns the processed EMG signal data."""
        if not hasattr(self, '_processed_data'):
            self._processed_data = self.process_emg()
        return self._processed_data