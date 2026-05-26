import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .forceData import ForceData
from .markerData import MarkerData
from .analogData import AnalogData

"""This module defines the TrialData container class for storing trial-level data including markers, analogs, and forces."""

@dataclass
class TrialData:
    """Data class for a single trial."""
    trial_name: str
    markers: Dict[str, MarkerData] = field(default_factory=dict)
    analogs: Dict[str, AnalogData] = field(default_factory=dict)
    forces: Dict[str, ForceData] = field(default_factory=dict)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop all marker, analog, and force data to the specified index range."""
        for marker in self.markers.values():
            marker.crop(start_idx, end_idx)
        for analog in self.analogs.values():
            analog.crop(start_idx, end_idx)
        for force in self.forces.values():
            force.crop(start_idx, end_idx)
    
    def lowpass_filter(self, cutoff_marker: float, cutoff_analog: float, cutoff_force: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to all marker, analog and force data. 
        Cutoff frequencies can be specified separately for markers, analogs, and forces."""
        for marker in self.markers.values():
            marker.lowpass_filter(cutoff_marker, order)
        for analog in self.analogs.values():
            analog.lowpass_filter(cutoff_analog, order)
        for force in self.forces.values():
            force.lowpass_filter(cutoff_force, order)

    def get_marker_names(self) -> List[str]:
        """Returns list of marker names."""
        return list(self.markers.keys())
    
    def get_analog_names(self) -> List[str]:
        """Returns list of analog channel names."""
        return list(self.analogs.keys())
    
    def get_force_names(self) -> List[str]:
        """Returns list of force plate names."""
        return list(self.forces.keys())
    
    def get_marker(self, name: str) -> Optional[MarkerData]:
        """Get marker data by name."""
        return self.markers.get(name)
    
    def get_analog(self, name: str) -> Optional[AnalogData]:
        """Get analog data by name."""
        return self.analogs.get(name)
    
    def get_analog_by_channel(self, channel: int) -> Optional[AnalogData]:
        """Get analog data by channel number."""
        for analog in self.analogs.values():
            if analog.channel == channel:
                return analog
        return None
    
    def get_force(self, name: str) -> Optional[ForceData]:
        """Get force data by name."""
        return self.forces.get(name)
    
    def get_event(self, name: str) -> Optional[int]:
        """Get event time by name."""
        return self.events.get(name)
    
    def __repr__(self) -> str:
        return f"TrialData(trial_name={self.trial_name}, markers={list(self.markers.keys())}, analogs={list(self.analogs.keys())}, forces={list(self.forces.keys())})"