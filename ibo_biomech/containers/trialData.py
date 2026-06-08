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
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculates some easy access attributes after initialization."""
        self.marker_labels = list(self.markers.keys())
        self.analog_labels = list(self.analogs.keys())
        self.marker_rate = next(iter(self.markers.values())).sampling_rate if self.markers else None
        self.analog_rate = next(iter(self.analogs.values())).sampling_rate if self.analogs else None
    
    def rotate_markers(self, axis, angle_deg) -> None:
        """Rotate all marker trajectories around specified axis by given angle in degrees."""
        for marker in self.markers.values():
            marker.rotate(axis, angle_deg)
    
    def convert_marker_units(self, target_unit: str) -> None:
        """Convert all marker trajectories to the specified unit.
            Only converts markers"""
        for marker in self.markers.values():
            marker.convert_units(target_unit)
    
    def convert_force_units(self, target_unit: str) -> None:
        """Convert all force plate data to the specified unit in terms of position (e.g. mm to m)."""
        for force in self.forces.values():
            force.convert_units(target_unit)
    
    def convert_units(self, target_unit: str) -> None:
        """Convert all marker, analog, and force data to the specified unit."""
        self.convert_marker_units(target_unit)
        self.convert_force_units(target_unit)

    def rotate_forces(self, axis, angle_deg) -> None:
        """Rotate all force plate data around specified axis by given angle in degrees."""
        for force in self.forces.values():
            force.rotate(axis, angle_deg)

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
        self.lowpass_filter_markers(cutoff_marker, order)
        self.lowpass_filter_analogs(cutoff_analog, order)
        self.lowpass_filter_forces(cutoff_force, order)

    def lowpass_filter_markers(self, cutoff_freq: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to all marker data."""
        for marker in self.markers.values():
            marker.lowpass_filter(cutoff_freq, order)
    
    def lowpass_filter_analogs(self, cutoff_freq: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to all analog data."""
        for analog in self.analogs.values():
            analog.lowpass_filter(cutoff_freq, order)

    def lowpass_filter_forces(self, cutoff_freq: float, order: int = 4) -> None:
        """Apply low-pass Butterworth filter to all force plate data."""
        for force in self.forces.values():
            force.lowpass_filter(cutoff_freq, order)

    def add_marker(self, marker_data: MarkerData) -> None:
        """Add a marker to the trial."""
        self.markers[marker_data.name] = marker_data
        
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
        return f"TrialData \n {'-'*50}\n trial_name={self.trial_name}\n{'-'*50}\nmetadata={self.metadata}\n{'-'*50}\nmarkers={list(self.markers.keys())}\n {'-'*50}\nanalogs={list(self.analogs.keys())}\n{'-'*50}\nforces={list(self.forces.keys())})"