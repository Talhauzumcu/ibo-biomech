"""Trial-level data container.

This module defines :class:`TrialData`, the central container returned by the
file handlers. It groups all markers, analog channels, force plates and EMG
signals of a single trial and offers trial-wide processing helpers.
"""
import numpy as np
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ibo_biomech.utils.utils import read_sto, read_mot, time_normalize
from .forceData import ForceData
from .markerData import MarkerData
from .analogData import AnalogData
from .emgData import EMGData
from .IKResults import IKResults
from .IDResults import IDResults
import pandas as pd

@dataclass
class TrialData:
    """All data belonging to a single recorded trial.

    Trial-wide methods fan out to the per-channel containers, so cropping,
    filtering, rotating and unit conversion all operate **in place**.

    Attributes:
        name: Name of the trial.
        markers: Mapping of marker label to :class:`MarkerData`.
        analogs: Mapping of channel label to :class:`AnalogData`.
        forces: Mapping of plate name to :class:`ForceData`.
        emgs: Mapping of channel name to :class:`EMGData`.
        metadata: Free-form trial metadata.
        marker_labels: Cached list of marker labels (set in ``__post_init__``).
        analog_labels: Cached list of analog labels (set in ``__post_init__``).
        marker_rate: Marker sampling rate in Hz (set in ``__post_init__``).
        analog_rate: Analog sampling rate in Hz (set in ``__post_init__``).
    """
    name: str = "Unknown"
    markers: Dict[str, MarkerData] = field(default_factory=dict)
    analogs: Dict[str, AnalogData] = field(default_factory=dict)
    forces: Dict[str, ForceData] = field(default_factory=dict)
    emgs: Dict[str, EMGData] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    ik_results: IKResults = field(default=None, repr=False)
    id_results: IDResults = field(default=None, repr=False)

    def __post_init__(self):
        """Cache convenience attributes (labels and sampling rates)."""
        self.marker_labels = list(self.markers.keys())
        self.analog_labels = list(self.analogs.keys())
        self.marker_rate = next(iter(self.markers.values())).sampling_rate if self.markers else None
        self.analog_rate = next(iter(self.analogs.values())).sampling_rate if self.analogs else None

    def parse_EMG_data(self, EMGChannels: List[int]) -> None:
        """Build :class:`EMGData` entries from selected analog channels.

        For each channel number, the matching analog signal is copied into a new
        :class:`EMGData` and stored in :attr:`emgs` under the analog's name.

        Args:
            EMGChannels: Hardware channel indices that carry EMG signals.
        """
        for channel in EMGChannels:
            analog = self.get_analog_by_channel(channel)
            if analog is not None:
                emg = EMGData(
                    name=analog.name,
                    data=analog.data,
                    sampling_rate=analog.sampling_rate,
                    unit=analog.unit,
                    channel=analog.channel
                )
                self.emgs[analog.name] = emg

    def attach_IK_results(self, ikresults_file: str) -> None:
        """Attach inverse kinematics results to the trial.

        Args:
            ik_results_file: Path to the IK results file.
        """
        self.ik_results = IKResults(filepath=ikresults_file)

    def attach_ID_results(self, idresults_file: str) -> None:
            """Attach inverse dynamics results to the trial.

            Args:
                id_results_file: Path to the ID results file.
            """
            self.id_results = IDResults(filepath=idresults_file)

    def rotate_markers(self, axis: str, angle_deg: float) -> None:
        """Rotate every marker about an axis in place.

        Args:
            axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``).
            angle_deg: Rotation angle in degrees.
        """
        for marker in self.markers.values():
            marker.rotate(axis, angle_deg)

    def convert_marker_units(self, target_unit: str) -> None:
        """Convert every marker to a target length unit in place.

        Args:
            target_unit: Desired unit (``'mm'`` or ``'m'``).
        """
        for marker in self.markers.values():
            marker.convert_units(target_unit)

    def convert_force_units(self, target_unit: str) -> None:
        """Convert position-derived quantities of every force plate in place.

        Args:
            target_unit: Desired length unit (``'mm'`` or ``'m'``).
        """
        for force in self.forces.values():
            force.convert_units(target_unit)

    def convert_units(self, target_unit: str) -> None:
        """Convert both markers and force plates to a target length unit.

        Args:
            target_unit: Desired unit (``'mm'`` or ``'m'``).
        """
        self.convert_marker_units(target_unit)
        self.convert_force_units(target_unit)

    def rotate_forces(self, axis: str, angle_deg: float) -> None:
        """Rotate every force plate about an axis in place.

        Args:
            axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``).
            angle_deg: Rotation angle in degrees.
        """
        for force in self.forces.values():
            force.rotate(axis, angle_deg)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop every marker, analog and force channel to the same index range.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """
        for marker in self.markers.values():
            marker.crop(start_idx, end_idx)
        for analog in self.analogs.values():
            analog.crop(start_idx, end_idx)
        for force in self.forces.values():
            force.crop(start_idx, end_idx)
        for emg in self.emgs.values():
            emg.crop(start_idx, end_idx)

    def lowpass_filter(self, cutoff_marker: float, cutoff_analog: float, cutoff_force: float, order: int = 4) -> None:
        """Low-pass filter markers, analogs and forces with separate cutoffs.

        Args:
            cutoff_marker: Cutoff for markers in Hz.
            cutoff_analog: Cutoff for analog channels in Hz.
            cutoff_force: Cutoff for force plates in Hz.
            order: Filter order applied to all. Defaults to 4.
        """
        self.lowpass_filter_markers(cutoff_marker, order)
        self.lowpass_filter_analogs(cutoff_analog, order)
        self.lowpass_filter_forces(cutoff_force, order)

    def lowpass_filter_markers(self, cutoff_freq: float, order: int = 4) -> None:
        """Low-pass filter every marker in place.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for marker in self.markers.values():
            marker.lowpass_filter(cutoff_freq, order)

    def lowpass_filter_analogs(self, cutoff_freq: float, order: int = 4) -> None:
        """Low-pass filter every analog channel in place.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for analog in self.analogs.values():
            analog.lowpass_filter(cutoff_freq, order)

    def lowpass_filter_forces(self, cutoff_freq: float, order: int = 4) -> None:
        """Low-pass filter every force plate in place.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for force in self.forces.values():
            force.lowpass_filter(cutoff_freq, order)

    def lowpass_filter_emgs(self, cutoff_freq: float, order: int = 4) -> None:
        """Low-pass filter every EMG channel in place.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for emg in self.emgs.values():
            emg.lowpass_filter(cutoff_freq, order)
            
    def add_marker(self, marker_data: MarkerData) -> None:
        """Add (or replace) a marker, keyed by its name.

        Args:
            marker_data: The marker to store.
        """
        self.markers[marker_data.name] = marker_data

    def add_force(self, force_data: ForceData) -> None:
        """Add (or replace) a force plate, keyed by its name.

        Args:
            force_data: The force plate to store.
        """
        self.forces[force_data.name] = force_data

    def add_analog(self, analog_data: AnalogData) -> None:
        """Add (or replace) an analog channel, keyed by its name.

        Args:
            analog_data: The analog channel to store.
        """
        self.analogs[analog_data.name] = analog_data

    def add_emg(self, emg_data: EMGData) -> None:
        """Add (or replace) an EMG channel, keyed by its name.

        Args:
            emg_data: The EMG channel to store.
        """
        self.emgs[emg_data.name] = emg_data

    def as_df(self, data_dict: Dict[str, Any], time_normalize=False) -> pd.DataFrame:
        """Convert a dictionary of data objects to a pandas DataFrame. Columns will also include
        the metadata for easy plotting and analysis.
        
        Args:
            data_dict: Dictionary of data objects (MarkerData, ForceData, AnalogData, etc.)
        Returns:
            A pandas DataFrame containing the data and metadata.
        """
        df = pd.DataFrame()
        for key, value in data_dict.items():
            if isinstance(value, MarkerData):
                df[f'{key}_x'] = value.x
                df[f'{key}_y'] = value.y
                df[f'{key}_z'] = value.z
            elif isinstance(value, ForceData):
                df[f'{key}_Fx'] = value.Fx
                df[f'{key}_Fy'] = value.Fy
                df[f'{key}_Fz'] = value.Fz
                df[f'{key}_Mx'] = value.Mx
                df[f'{key}_My'] = value.My
                df[f'{key}_Mz'] = value.Mz
                df[f'{key}_CoPx'] = value.CoPx
                df[f'{key}_CoPy'] = value.CoPy
            else:
                df[key] = value.data
        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                #Skip if the metadata is an array, as it will not fit into a column of the dataframe
                continue
            df[key] = value
        return df

    def get_marker_names(self) -> List[str]:
        """Return the list of marker labels.

        Returns:
            Marker names in insertion order.
        """
        return list(self.markers.keys())

    def get_analog_names(self) -> List[str]:
        """Return the list of analog channel labels.

        Returns:
            Analog channel names in insertion order.
        """
        return list(self.analogs.keys())

    def get_force_names(self) -> List[str]:
        """Return the list of force plate names.

        Returns:
            Force plate names in insertion order.
        """
        return list(self.forces.keys())

    def get_marker(self, name: str) -> Optional[MarkerData]:
        """Look up a marker by name.

        Args:
            name: Marker label.

        Returns:
            The matching :class:`MarkerData`, or ``None`` if not found.
        """
        return self.markers.get(name)

    def get_markers(self, names: List[str]) -> Dict[str, MarkerData]:
        """Look up several markers by name.

        Args:
            names: Marker labels to fetch.

        Returns:
            A list of :class:`MarkerData` (``None`` for any name not found), in
            the order requested.
        """
        return [self.markers.get(name) for name in names]

    def get_analog(self, name: str) -> Optional[AnalogData]:
        """Look up an analog channel by name.

        Args:
            name: Analog channel label.

        Returns:
            The matching :class:`AnalogData`, or ``None`` if not found.
        """
        return self.analogs.get(name)

    def get_analog_by_channel(self, channel: int) -> Optional[AnalogData]:
        """Look up an analog channel by its hardware channel index.

        Args:
            channel: Hardware channel index.

        Returns:
            The matching :class:`AnalogData`, or ``None`` if not found.
        """
        for analog in self.analogs.values():
            if analog.channel == channel:
                return analog
        return None

    def get_force(self, name: str) -> Optional[ForceData]:
        """Look up a force plate by name.

        Args:
            name: Force plate name.

        Returns:
            The matching :class:`ForceData`, or ``None`` if not found.
        """
        return self.forces.get(name)
    
    def _get_right_foot_forceplate_index(self) -> int:
        """Return the index of the force plate that right foot hits. It is used for automatic external loads file generation"""
        ##TODO
        pass        

    def __repr__(self) -> str:
        """Return a concise summary of the trial's contents."""
        emgs = len(self.emgs) if self.emgs is not None else 0
        ik_results = len(self.ik_results.data) if self.ik_results is not None else 0
        id_results = len(self.id_results.data) if self.id_results is not None else 0
        return (
            f"TrialData(name={self.name!r}, markers={len(self.markers)}, "
            f"analogs={len(self.analogs)}, forces={len(self.forces)}, "
            f"emgs={emgs}, IK Results={ik_results}," 
            f"ID Results={id_results})"
        )

    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()
