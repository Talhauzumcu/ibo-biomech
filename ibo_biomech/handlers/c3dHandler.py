from ezc3d import c3d
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union
from ibo_biomech.containers import AnalogData, ForceData, MarkerData, TrialData
import numpy as np
from copy import deepcopy
from .h5Handler import H5Handler
from ibo_biomech.containers._validation import collection_clock
from ibo_biomech.utils.utils import *

class C3DHandler:
    """Load, manipulate and export C3D motion-capture files.

    Wraps the :mod:`ezc3d` reader. After :meth:`load_data`, parsed channels are
    available as the :attr:`markers`, :attr:`analogs` and :attr:`forces`
    dictionaries shared with the returned TrialData. The raw source structure
    remains separate and is written only by :meth:`write_raw_c3d`.

    Attributes:
        filepath: Path to the source C3D file.
        c3d_data: Underlying ezc3d data structure (``None`` until loaded).
        markers: Mapping of marker label to :class:`~ibo_biomech.containers.MarkerData`.
        analogs: Mapping of channel label to :class:`~ibo_biomech.containers.AnalogData`.
        forces: Mapping of plate name to :class:`~ibo_biomech.containers.ForceData`.
    """

    def __init__(self, filepath: Optional[str] = None):
        """Initialize the handler.

        Args:
            filepath: Path to the C3D file to load. The file is not read until
                :meth:`load_data` is called.
        """
        self.filepath = filepath
        self.trial_name = os.path.splitext(os.path.basename(filepath))[0] if filepath else None
        self.c3d_data = None
        self.markers: Dict[str, MarkerData] = {}
        self.analogs: Dict[str, AnalogData] = {}
        self.forces: Dict[str, ForceData] = {}
        self.trial = None
        self._source_forces = {}

    def load_data(self) -> TrialData:
        """Read the C3D file and parse it into container objects. Also creates and returns a TrialData instance same as h5handler.

        Populates :attr:`markers`, :attr:`analogs` and :attr:`forces` and stores
        the raw structure on :attr:`c3d_data`.

        Returns:
            A :class:`~ibo_biomech.containers.TrialData` instance containing the
            parsed data.
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the C3D file cannot be read or parsed.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"C3D file not found: {self.filepath}")
        
        try:
            self.c3d_data = c3d(os.fspath(self.filepath), extract_forceplat_data=True)
            self._parse_markers()
            self._parse_analogs()
            self._parse_force_plates()     
            self.is_loaded = True

        except Exception as e:
            raise Exception(f"Error loading C3D file: {str(e)}")
    
        self.trial = TrialData(name=self.trial_name, markers=self.markers,
                               analogs=self.analogs, forces=self.forces)
        self._source_forces = deepcopy(self.forces)
        return self.trial

    @staticmethod
    def _unique_label(label, existing, reserved=()):
        base = label.strip() or 'unnamed'
        candidate, suffix = base, 2
        while candidate in existing:
            candidate = f'{base}_{suffix}'
            suffix += 1
            while candidate in reserved:
                candidate = f'{base}_{suffix}'
                suffix += 1
        return candidate

    def _parse_markers(self) -> None:
        """Parse marker data from C3D file into MarkerData containers."""
        self.markers = {}
        try:
            points = self.c3d_data['data']['points']
            marker_labels = self.c3d_data['parameters']['POINT']['LABELS']['value']
            point_rate = self.c3d_data['parameters']['POINT']['RATE']['value'][0]
            unit = self.c3d_data['parameters']['POINT']['UNITS']['value'][0]
            first_frame = int(self.c3d_data['header']['points']['first_frame'])
            meta = self.c3d_data['data'].get('meta_points', {})
            virtual = self.c3d_data['parameters']['POINT'].get('VIRTUAL', {}).get('value', [])
            for i, label in enumerate(marker_labels):
                label = self._unique_label(label, self.markers, {v.strip() for v in marker_labels})
                x = points[0, i, :].copy()
                y = points[1, i, :].copy()
                z = points[2, i, :].copy()
                
                marker = MarkerData(
                    name=label.strip(),
                    x=x,
                    y=y,
                    z=z,
                    unit=unit.strip(),
                    sampling_rate=point_rate,
                    time=(first_frame + np.arange(points.shape[-1])) / point_rate,
                    first_frame=first_frame,
                    residuals=meta['residuals'][0, i].copy() if 'residuals' in meta else None,
                    camera_masks=meta['camera_masks'][:, i].copy() if 'camera_masks' in meta else None,
                    virtual=int(virtual[i]) if i < len(virtual) else 0
                )
                
                self.markers[label.strip()] = marker
                
        except KeyError as e:
            print(f"Warning: Could not parse marker data: {str(e)}")
    
    def _parse_analogs(self) -> None:
        """Parse analog data from C3D file into AnalogData containers."""
        self.analogs = {}
        try:
            analogs = self.c3d_data['data']['analogs']
            analog_labels = self.c3d_data['parameters']['ANALOG']['LABELS']['value']
            analog_rate = self.c3d_data['parameters']['ANALOG']['RATE']['value'][0]

            try:
                units = self.c3d_data['parameters']['ANALOG']['UNITS']['value']
            except KeyError:
                units = [""] * len(analog_labels)
            
            for i, label in enumerate(analog_labels):
                analog_signal = analogs[0, i, :]

                label = self._unique_label(label, self.analogs, {v.strip() for v in analog_labels})
                analog = AnalogData(
                    name=label.strip(),
                    data=analog_signal.copy(),
                    sampling_rate=analog_rate,
                    unit=units[i].strip() if i < len(units) else "",
                    channel=i,
                    time=self.c3d_data['header']['points']['first_frame'] /
                         self.c3d_data['parameters']['POINT']['RATE']['value'][0] +
                         np.arange(len(analog_signal)) / analog_rate
                )
                
                self.analogs[label.strip()] = analog
                
        except KeyError as e:
            print(f"Warning: Could not parse analog data: {str(e)}")
    
    def _parse_force_plates(self) -> None:
        """Parse force plate data from C3D file into ForceData containers."""
        self.forces = {}
        force_plates = self.c3d_data['data']['platform']

        for i, plate in enumerate(force_plates):

            metadata = {
                'unit_force': plate['unit_force'],
                'unit_moment': plate['unit_moment'],
                'unit_position': plate['unit_position'], #Center of pressure units
                'calibration_matrix': plate['cal_matrix'],
                'corners': plate['corners'],
                'origin': plate['origin']
            }

            fp_rotation, position = get_fp_cs(plate['corners'])

            fp_rotation = np.repeat(fp_rotation[:, :, np.newaxis], plate['force'].shape[1], axis=2) #c3d has these as static values, to be consistent with future moving fp implementations repeat these.
            position = np.repeat(position[:, np.newaxis], plate['force'].shape[1], axis=1)
            corners = np.repeat(plate['corners'][:, :, np.newaxis], plate['force'].shape[1], axis=2)

            self.forces[f"forceplate_{i}"] = ForceData(
                name=f"forceplate_{i}",
                force=plate['force'],
                moment=plate['moment'],
                cop=plate['center_of_pressure'],
                Tz = plate['Tz'],
                rotation=fp_rotation,
                position=position,
                corners=corners,
                origin=plate['origin'],
                metadata=metadata,
                sampling_rate=self.c3d_data['parameters']['ANALOG']['RATE']['value'][0],
                time=self.c3d_data['header']['points']['first_frame'] /
                     self.c3d_data['parameters']['POINT']['RATE']['value'][0] +
                     np.arange(plate['force'].shape[1]) / self.c3d_data['parameters']['ANALOG']['RATE']['value'][0]
            )

    def add_marker(self, marker: MarkerData) -> None:
        """Add a processed marker; write_c3d serializes this shared trial state."""
        if self.trial is None:
            raise ValueError('Load a trial before adding markers.')
        self.trial.add_marker(marker)

    @staticmethod
    def _trim_events(raw, start, end):
        """Retain timestamped C3D events within the half-open recording interval."""
        events = raw['parameters'].get('EVENT')
        if not events or 'TIMES' not in events:
            return
        times = np.asarray(events['TIMES']['value'])
        if times.ndim != 2 or times.shape[0] != 2:
            raise ValueError('EVENT:TIMES must have shape (2, events).')
        seconds = times[0] * 60. + times[1]
        keep = (seconds >= start - 1e-9) & (seconds < end - 1e-9)
        for name, parameter in events.items():
            if name in ('__METADATA__', 'USED'):
                continue
            value = parameter['value']
            array = np.asarray(value)
            if array.ndim and array.shape[-1] == len(seconds):
                selected = array[..., keep]
                parameter['value'] = selected.tolist() if isinstance(value, list) else selected
        events['USED']['value'] = np.array([int(keep.sum())])

    def _check_derived_forces(self, trial):
        """C3D platforms are reconstructed from analogs, never from cached vectors."""
        if set(trial.forces) != set(self._source_forces):
            raise ValueError('C3D force-plate addition/removal is unsupported; save processed forces to HDF5/MOT.')
        for name, plate in trial.forces.items():
            original = self._source_forces[name]
            plate.validate()
            if plate.time is None:
                raise ValueError('C3D force export requires a source-aligned clock.')
            indices = np.searchsorted(original.time, plate.time)
            if np.any(indices >= len(original.time)) or not np.allclose(original.time[indices], plate.time, atol=1e-9, rtol=0):
                raise ValueError('C3D force clock must select original samples; use HDF5/MOT for resampled forces.')
            for field in ('force', 'moment', 'cop', 'Tz', 'corners', 'position', 'rotation'):
                if not np.allclose(getattr(plate, field), getattr(original, field)[..., indices], equal_nan=True):
                    raise ValueError('Processed force vectors cannot be reconstructed as calibrated C3D analogs; '
                                     'save to HDF5/MOT, or process the source analog channels instead.')
            if (not np.allclose(plate.origin, original.origin, equal_nan=True) or
                (plate.unit_force, plate.unit_moment, plate.unit_cop, plate.coordinateSystem) !=
                (original.unit_force, original.unit_moment, original.unit_cop, original.coordinateSystem)):
                raise ValueError('C3D force geometry/units changed; use HDF5/MOT for processed forces.')

    def _processed_c3d(self, trial):
        H5Handler._validate_trial(trial)
        if not trial.markers:
            raise ValueError('Processed C3D export requires at least one marker.')
        if trial.emgs or trial.ik_results is not None or trial.id_results is not None:
            raise ValueError('C3D export cannot preserve separate EMG/IK/ID results; use HDF5.')
        if getattr(trial, '_unloaded_collections', set()):
            raise ValueError('C3D export requires a fully loaded trial.')
        self._check_derived_forces(trial)
        n, time, rate = collection_clock(trial.markers, markers=True)
        if rate is None or time is None:
            raise ValueError('C3D markers require a sampling rate and clock.')
        first = next(iter(trial.markers.values()))
        frame = int(round(time[0] * rate))
        if frame < 0 or not np.isclose(frame, time[0] * rate, atol=1e-6):
            raise ValueError('C3D time origin must lie on a nonnegative marker frame.')
        if trial.forces and first.unit != self.c3d_data['parameters']['POINT']['UNITS']['value'][0].strip():
            raise ValueError('Changing C3D point units also changes force-platform calibration; use HDF5/MOT.')
        raw = deepcopy(self.c3d_data)
        points = raw['parameters']['POINT']
        points['LABELS']['value'] = list(trial.markers)
        for key in list(points):
            if key.startswith('LABELS') and key != 'LABELS':
                del points[key]
        points['RATE']['value'], points['UNITS']['value'] = [rate], [first.unit]
        points['FRAMES']['value'] = [n]
        raw['header']['points']['first_frame'] = frame
        raw['data']['points'] = np.stack([marker.get_frame_trajectory() for marker in trial.markers.values()], axis=1)
        # C3D has no unknown-residual state: unknown/unobserved points are invalid (-1).
        raw['data']['meta_points'] = {
            'residuals': np.stack([np.nan_to_num(marker.residuals, nan=-1.) if marker.residuals is not None
                                   else np.full(n, -1.) for marker in trial.markers.values()])[None, ...],
            'camera_masks': np.stack([marker.camera_masks if marker.camera_masks is not None
                                      else np.zeros((7, n), dtype=bool)
                                      for marker in trial.markers.values()], axis=1).astype(bool)}
        raw.add_parameter('POINT', 'VIRTUAL', [int(m.virtual) for m in trial.markers.values()])
        analogs = raw['parameters']['ANALOG']
        if trial.analogs:
            na, analog_time, analog_rate = collection_clock(trial.analogs)
            if analog_rate is None or analog_time is None:
                raise ValueError('C3D analogs require a sampling rate and clock.')
            ratio = analog_rate / rate
            if not np.isclose(ratio, round(ratio)) or na != n * round(ratio) or not np.isclose(analog_time[0], time[0], atol=1e-9):
                raise ValueError('C3D analogs must cover the same interval at an integer multiple of the point rate.')
            channel_map = {channel.channel: i + 1 for i, channel in enumerate(trial.analogs.values())}
            if len(channel_map) != len(trial.analogs):
                raise ValueError('C3D analog source channel identifiers must be unique.')
            if trial.forces:
                for plate in trial.forces.values():
                    if plate.time.shape != analog_time.shape or not np.allclose(plate.time, analog_time, rtol=0, atol=1e-9):
                        raise ValueError('Crop C3D force and analog containers to the same interval.')
                mapping = np.asarray(raw['parameters']['FORCE_PLATFORM']['CHANNEL']['value']).copy()
                for index in np.ndindex(mapping.shape):
                    if mapping[index] > 0:
                        source_channel = int(mapping[index]) - 1
                        if source_channel not in channel_map:
                            raise ValueError('Cannot remove an analog channel referenced by a force plate.')
                        mapping[index] = channel_map[source_channel]
                raw['parameters']['FORCE_PLATFORM']['CHANNEL']['value'] = mapping
            raw['data']['analogs'] = np.stack([a.data for a in trial.analogs.values()])[None, ...]
            analogs['RATE']['value'] = [analog_rate]
        else:
            if trial.forces:
                raise ValueError('Force platforms require their original analog channels.')
            raw['data']['analogs'] = np.empty((1, 0, 0))
            analogs['RATE']['value'] = [0.]
        analogs['LABELS']['value'] = list(trial.analogs)
        analogs['UNITS']['value'] = [a.unit for a in trial.analogs.values()]
        analogs['SCALE']['value'] = np.ones(len(trial.analogs))
        analogs['OFFSET']['value'] = np.zeros(len(trial.analogs), dtype=int)
        analogs['GEN_SCALE']['value'] = [1.]
        for key in list(analogs):
            if key.startswith('LABELS') and key != 'LABELS':
                del analogs[key]
        self._trim_events(raw, time[0], time[-1] + 1. / rate)
        # The old derived platform cache is deliberately not used by ezc3d.write.
        if 'platform' in raw['data']:
            del raw['data']['platform']
        return raw

    @staticmethod
    def _write_atomic(raw, output_filepath):
        path = os.fspath(output_filepath)
        if not path.lower().endswith('.c3d'):
            path += '.c3d'
        fd, temporary = tempfile.mkstemp(prefix='.c3d-', suffix='.c3d', dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        try:
            raw.write(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def write_c3d(self, output_filepath: str, trial: Optional[TrialData] = None) -> str:
        """Write processed markers and analogs from the supplied/shared TrialData.

        Force plates are derived from calibrated analogs by C3D readers. Direct
        edits to cached force vectors/geometry are rejected; export those to
        HDF5/MOT. Frame offsets, residuals, camera masks and in-range events are
        preserved. Unsupported data is rejected before touching the destination.
        Use write_raw_c3d explicitly to serialize the untouched raw structure.
        """
        if self.c3d_data is None:
            raise ValueError('No C3D data loaded to write.')
        return self._write_atomic(self._processed_c3d(self.trial if trial is None else trial), output_filepath)

    def write_raw_c3d(self, output_filepath: str) -> str:
        """Explicitly write the raw source structure, ignoring processed containers."""
        if self.c3d_data is None:
            raise ValueError('No C3D data loaded to write.')
        return self._write_atomic(deepcopy(self.c3d_data), output_filepath)

    def slice_c3d(self, start_frame: int, end_frame: int):
        """Crop the shared trial in place; indices are zero-based, end inclusive.

        The raw source stays available to write_raw_c3d. write_c3d reconstructs
        processed samples and retains only in-range events. Returned TrialData
        references observe the crop, with analog and force clocks kept aligned.
        """
        if self.trial is None:
            raise ValueError('No C3D data loaded to slice.')
        first = next(iter(self.trial.markers.values()))
        validate_crop_range(start_frame, end_frame + 1, len(first.x))
        trial = deepcopy(self.trial)
        start = first.time[start_frame]
        end = first.time[end_frame] + 1. / first.sampling_rate
        for marker in trial.markers.values():
            marker.crop(start_frame, end_frame + 1)
        for collection in (trial.analogs, trial.forces, trial.emgs):
            for channel in collection.values():
                left = int(np.searchsorted(channel.time, start - 1e-9))
                right = int(np.searchsorted(channel.time, end - 1e-9))
                channel.crop(left, right)
        raw = self._processed_c3d(trial)  # validate everything before changing shared state
        for name in ('markers', 'analogs', 'forces', 'emgs'):
            mapping = getattr(self.trial, name)
            mapping.clear()
            mapping.update(getattr(trial, name))
        self.trial.__post_init__()
        return raw
