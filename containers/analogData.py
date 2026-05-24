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
