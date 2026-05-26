import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class AnalogData:
    """Data class for analog signals (EMG, force plates, etc.)."""
    name: str
    data: np.ndarray
    sampling_rate: float = None
    unit: str = ""
    channel: Optional[int] = None

    def get_data(self) -> np.ndarray:
        """Returns the analog signal data."""
        return self.data

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the analog data to the specified index range."""
        self.data = self.data[start_idx:end_idx]
        
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