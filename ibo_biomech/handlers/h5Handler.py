from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

import h5py
import numpy as np

from ibo_biomech.containers import AnalogData, ForceData, MarkerData, EMGData, TrialData, Subject, IKResults, IDResults, Data
from ibo_biomech.containers import RigidBody, Event
from ibo_biomech.containers._validation import collection_clock, frame_at_time, validate_clock
from ._force_schema import read_plate, write_plate


class H5Handler:
    """Read and write the lab's HDF5 trial format.

    Reading returns a fully populated :class:`~ibo_biomech.containers.TrialData`.
    Writing validates the trial and atomically replaces loaded collections.
    On clock changes, opaque annotations move to ``SourceData`` so they remain
    explicitly scoped to the original recording. Only the current force schema
    is accepted; older HDF5 files must be regenerated from source recordings.

    Metadata fields (``SubjectID``, ``Condition``, ``BodyMass``, ...) are
    intentionally outside the scope of this handler. To update them after
    saving, use h5py directly::

        with h5py.File("trial_processed.h5", "r+") as f:
            f["MetaData"].attrs["SubjectID"] = "P01"
            f["MetaData"].attrs["Condition"] = "walking"

    Attributes:
        h5_path: Path to the source HDF5 file.
        trial_name: Trial name derived from the file name.
    """

    def __init__(self, h5_path: str) -> None:
        """Initialize the handler.

        Args:
            h5_path: Path to an existing HDF5 file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        self.h5_path = h5_path
        self.trial_name = os.path.splitext(os.path.basename(h5_path))[0]

    def load_data(self, 
                  load_markers: bool = True,
                  load_analogs: bool = True,
                  load_forces: bool = True,
                  load_emgs: bool = True,
                  load_ik_results: bool = True,
                  load_id_results: bool = True,
                  load_rigid_bodies: bool = True,
                  load_events: bool = True ) -> TrialData:
        """Read the HDF5 file and return a fully populated trial. Accepts flags to selectively load specific data types for faster loading times

        Args:
            load_markers: Whether to load marker data.
            load_analogs: Whether to load analog data.
            load_forces: Whether to load force plate data.
            load_emgs: Whether to load EMG data.
            load_ik_results: Whether to load inverse kinematics results.
            load_id_results: Whether to load inverse dynamics results.
            load_rigid_bodies: Whether to load rigid-body poses and marker names.
            load_events: Whether to load named events, including repeated names.
        Returns:
            A :class:`~ibo_biomech.containers.TrialData` with markers, analogs,
            forces, rigid bodies, events and metadata.
        """

        with h5py.File(self.h5_path, "r") as h5f:
            markers = self._load_markers(h5f) if load_markers else {}
            analogs = self._load_analogs(h5f) if load_analogs else {}
            forces = self._load_forces(h5f) if load_forces else {}
            rigid_bodies = self._load_rigid_bodies(h5f) if load_rigid_bodies else {}
            events = self._load_events(h5f) if load_events else []
            emgs = self._load_emgs(h5f) if load_emgs else {}
            ik_results = self._load_ik_results(h5f) if load_ik_results else None
            id_results = self._load_id_results(h5f) if load_id_results else None
            metadata = dict(h5f["MetaData"].attrs)

        trial = TrialData(
            name=self.trial_name,
            markers=markers,
            analogs=analogs,
            forces=forces,
            rigid_bodies=rigid_bodies,
            events=events,
            emgs=emgs,
            ik_results=ik_results,
            id_results=id_results,
            metadata=metadata,
        )

        trial._unloaded_collections = {name for name, loaded in (
            ('markers', load_markers), ('analogs', load_analogs), ('forces', load_forces),
            ('emgs', load_emgs), ('ik_results', load_ik_results), ('id_results', load_id_results),
            ('rigid_bodies', load_rigid_bodies), ('events', load_events)) if not loaded}
        return trial

    def load_subject_data(self) -> Subject:
        """Read the file into a subject populated from its metadata.

        Builds a :class:`~ibo_biomech.containers.Subject` from the file's
        ``MetaData`` attributes and attaches the loaded trial to it.

        Returns:
            A :class:`~ibo_biomech.containers.Subject` containing the trial.
        """
        trial_data = self.load_data()
        with h5py.File(self.h5_path, "r") as h5f:
            meta = h5f["MetaData"].attrs
            subject_id = meta.get("SubjectID", "")
            condition = meta.get("Condition", "")
            body_mass = meta.get("BodyMass", None)
            body_height = meta.get("BodyHeight", None)
            age = meta.get("Age", None)

        subject = Subject(id=subject_id, condition=condition, body_mass=body_mass, body_height=body_height, age=age)
        subject.add_trial(trial_name=trial_data.name, trial_data=trial_data)
        return subject

    @staticmethod
    def _replace_dataset(group, name, value):
        if name in group:
            del group[name]
        if value is not None:
            group.create_dataset(name, data=value, compression="gzip")

    @staticmethod
    def _validate_trial(trial):
        skipped = getattr(trial, '_unloaded_collections', set())
        for name in ('markers', 'analogs', 'emgs'):
            channels = getattr(trial, name)
            if name in skipped or not channels:
                continue
            collection_clock(channels, markers=name == 'markers')
            if name == 'markers':
                first = next(iter(channels.values()))
                H5Handler._validate_marker_units(channels, first.unit)
                for marker in channels.values():
                    marker.validate_sample_metadata()
                    if marker.first_frame != first.first_frame:
                        raise ValueError('Markers must share a first_frame.')
            else:
                for channel in channels.values():
                    if channel.channel is not None and not isinstance(channel.channel, (int, np.integer)):
                        raise ValueError('Channel identifiers must be integers or None.')
        if 'forces' not in skipped:
            for name, plate in trial.forces.items():
                if name != plate.name:
                    raise ValueError('Force mapping keys must match plate names.')
                plate.validate()
        if 'rigid_bodies' not in skipped:
            for name, body in trial.rigid_bodies.items():
                if not isinstance(body.name, str) or name != body.name:
                    raise ValueError('Rigid body mapping keys must match body names (strings).')
                if not isinstance(body.markers, list) or not all(isinstance(marker, str) for marker in body.markers):
                    raise ValueError('Rigid body markers must be a list of strings.')
                body.validate()
        if 'events' not in skipped:
            for event in trial.events:
                if not isinstance(event, Event):
                    raise ValueError('Trial events must contain Event instances.')
                event.validate()
        for name in ('ik_results', 'id_results'):
            result = getattr(trial, name)
            if name not in skipped and result is not None:
                for column in result.data.values():
                    if len(column.data) != len(result.time):
                        raise ValueError(f'{name}: column lengths must match the result clock.')

    def _atomic_update(self, out_path, update):
        destination = Path(out_path)
        fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.h5', dir=destination.parent)
        os.close(fd)
        try:
            shutil.copy2(self.h5_path, temporary)
            with h5py.File(temporary, 'r+') as h5f:
                update(h5f)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(out_path)

    def save_data(self, trial: TrialData, out_path: str) -> str:
        """Validate then atomically replace loaded collections from a processed trial.

        Removed channels are removed on disk. Collections deliberately skipped
        during load are preserved. Typed events retain source frames and times;
        crop them explicitly with TrialData.crop_events if desired.
        If clocks change, opaque legacy events/rigid bodies
        and unlabeled trajectories are moved under SourceData, where they remain
        explicitly scoped to the source recording rather than the cropped trial.
        """
        self._validate_trial(trial)
        skipped = getattr(trial, '_unloaded_collections', set())
        def update(h5f):
            self._archive_source_annotations(h5f, trial, skipped)
            for name in ('markers', 'analogs', 'forces', 'emgs', 'ik_results', 'id_results', 'rigid_bodies', 'events'):
                if name not in skipped:
                    getattr(self, f'_save_{name}')(h5f, trial)
            h5f.require_group('MetaData').attrs['LastUpdate'] = str(datetime.now())
        return self._atomic_update(out_path, update)

    @staticmethod
    def _archive_source_annotations(h5f, trial, skipped):
        changed = False
        for name, path in [('markers', 'Trajectories/Labeled'), ('analogs', 'Analog'),
                           ('emgs', 'EMG')]:
            if name in skipped or path not in h5f:
                continue
            channels = getattr(trial, name)
            original = h5f[path]
            if not channels:
                changed |= 'Data' in original and original['Data'].size > 0
                continue
            first = next(iter(channels.values()))
            count = len(first.x if name == 'markers' else first.data)
            changed |= 'Data' in original and original['Data'].shape[-1] != count
            if 'Time' in original:
                changed |= first.time is None or not np.array_equal(original['Time'][:], first.time)
        if 'forces' not in skipped and 'ForcePlates' in h5f:
            for group in h5f['ForcePlates'].values():
                plate = trial.forces.get(group.attrs.get('Name'))
                changed |= plate is None or group['Force'].shape[-1] != plate.num_samples
                if plate is not None and 'Time' in group:
                    changed |= plate.time is None or not np.array_equal(group['Time'][:], plate.time)
        if ('rigid_bodies' not in skipped and 'RigidBodies' in h5f
                and 'SchemaVersion' in h5f['RigidBodies'].attrs):
            for group in h5f['RigidBodies'].values():
                body = trial.rigid_bodies.get(group.attrs['Name'])
                changed |= body is None or group['Position'].shape[-1] != body.num_samples
                if body is not None and 'Time' in group:
                    changed |= body.time is None or not np.array_equal(group['Time'][:], body.time)
        if not changed:
            return
        paths = ['Trajectories/Unlabeled']
        if ('events' not in skipped and 'Events' in h5f
                and H5Handler._opaque_events(h5f['Events'])):
            paths.append('Events')
        if ('rigid_bodies' not in skipped and 'RigidBodies' in h5f
                and 'SchemaVersion' not in h5f['RigidBodies'].attrs):
            paths.append('RigidBodies')
        for path in paths:
            if path in h5f:
                target = 'SourceData/' + path
                if target in h5f:
                    raise ValueError(f'Cannot archive {path}: {target} already exists.')
                h5f.require_group(target.rsplit('/', 1)[0])
                h5f.move(path, target)
        source = h5f.require_group('SourceData')
        source.attrs['Scope'] = 'Original recording; not aligned to the processed trial clock'

    def modify_metadata(self, updates: Dict[str, Any], out_path: Optional[str] = None) -> H5Handler:
        """Atomically update metadata, including same-path saves."""
        out_path = self.h5_path if out_path is None else out_path
        def update(h5f):
            h5f.require_group('MetaData').attrs.update(updates)
        self._atomic_update(out_path, update)
        return H5Handler(out_path)

    def _save_markers(self, h5f: h5py.File, trial: TrialData) -> None:
        traj = h5f.require_group('Trajectories')
        labeled = traj.require_group('Labeled')
        markers = list(trial.markers.values())
        if markers:
            n, time, rate = collection_clock(trial.markers, markers=True)
            unit, start = markers[0].unit, markers[0].first_frame
            self._validate_marker_units(trial.markers, unit)
            data = np.stack([marker.get_frame_trajectory() for marker in markers])
            types = np.stack([np.full(n, 2) if marker.virtual else
                              marker.sample_types if marker.sample_types is not None
                              else np.ones(n, dtype=np.int8) for marker in markers])
            residuals = np.stack([marker.residuals if marker.residuals is not None
                                  else np.full(n, np.nan) for marker in markers])
            masks = np.stack([marker.camera_masks if marker.camera_masks is not None
                              else np.zeros((7, n), dtype=bool) for marker in markers])
        else:
            n, time, rate, unit, start = 0, None, None, '', 0
            data, types = np.empty((0, 4, 0)), np.empty((0, 0), dtype=np.int8)
            residuals, masks = np.empty((0, 0)), np.empty((0, 7, 0), dtype=bool)
        for name, value in [('Data', data), ('Type', types), ('Time', time),
                            ('Residuals', residuals), ('CameraMasks', masks),
                            ('CameraMasksKnown', np.array([m.camera_masks is not None for m in markers])),
                            ('Virtual', np.array([bool(m.virtual) for m in markers]))]:
            self._replace_dataset(labeled, name, value)
        labeled.attrs.update(Labels=list(trial.markers), Unit=unit, NumLabeled=len(markers),
                             ResidualStatus='NaN means unavailable; negative means invalid')
        traj.attrs.update(NumFrames=n, StartFrame=start, EndFrame=start + n - 1)
        if 'SamplingFrequency' in traj.attrs:
            del traj.attrs['SamplingFrequency']
        if rate is not None:
            traj.attrs['SamplingFrequency'] = rate

    def _save_channels(self, h5f, channels, group_name):
        group = h5f.require_group(group_name)
        if channels:
            n, time, rate = collection_clock(channels)
            data = np.stack([channel.data for channel in channels.values()])
        else:
            n, time, rate, data = 0, None, None, np.empty((0, 0))
        self._replace_dataset(group, 'Data', data)
        self._replace_dataset(group, 'Time', time)
        group.attrs.update(Labels=list(channels), NumSamples=n,
                           Units=[channel.unit or '' for channel in channels.values()],
                           Channels=[channel.channel if channel.channel is not None else -1
                                     for channel in channels.values()])
        for key in ('SamplingFrequency', 'SamplingFactor'):
            if key in group.attrs:
                del group.attrs[key]
        if rate is not None:
            group.attrs['SamplingFrequency'] = rate

    def _save_analogs(self, h5f: h5py.File, trial: TrialData) -> None:
        self._save_channels(h5f, trial.analogs, 'Analog')

    def _save_emgs(self, h5f: h5py.File, trial: TrialData) -> None:
        self._save_channels(h5f, trial.emgs, 'EMG')

    def _save_ik_results(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite IK results datasets from trial.ik_results."""
        if not trial.ik_results:
            return

        if 'IKResults' not in h5f.keys():
            ik_group = h5f.create_group("IKResults")
            ik_group.attrs["Labels"] = ['time']
            ik_group.attrs["NumSamples"] = 0
        else:
            ik_group = h5f["IKResults"]

        existing_labels = self._decode_labels(ik_group.attrs.get("Labels", []))
        ordered = existing_labels + [l for l in trial.ik_results.data if l not in existing_labels]
        sample_ik = next(iter(trial.ik_results.data.values()))
        n_samples = len(sample_ik.data)
        time = sample_ik.time
        metadata = trial.ik_results.metadata

        data = np.zeros((len(ordered), n_samples), dtype=np.float64)
        data[0] = time  # First row is time
        for i, label in enumerate(ordered[1:], start=1):  # Start from index 1 to skip time
            if label in trial.ik_results.data.keys():
                data[i] = trial.ik_results.data[label].data

        if "Data" in ik_group:
            del ik_group["Data"]
        if "Time" in ik_group:
            del ik_group["Time"]
        ik_group.create_dataset("Data", data=data, compression="gzip")
        ik_group.create_dataset("Time", data=time, compression="gzip") if time is not None else None
        ik_group.attrs["Labels"] = ordered
        ik_group.attrs["NumSamples"] = n_samples
        ik_group.attrs["Metadata"] = str(metadata)  # Store metadata as a string representation

    def _save_id_results(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite ID results datasets from trial.id_results."""
        if not trial.id_results:
            return

        if 'IDResults' not in h5f.keys():
            id_group = h5f.create_group("IDResults")
            id_group.attrs["Labels"] = ['time']
            id_group.attrs["NumSamples"] = 0
        else:
            id_group = h5f["IDResults"]

        existing_labels = self._decode_labels(id_group.attrs.get("Labels", []))
        ordered = existing_labels + [l for l in trial.id_results.data if l not in existing_labels]
        sample_id = next(iter(trial.id_results.data.values()))
        n_samples = len(sample_id.data)
        time = sample_id.time
        
        data = np.zeros((len(ordered), n_samples), dtype=np.float64)
        data[0] = time  # First row is time
        for i, label in enumerate(ordered[1:], start=1):  # Start from index 1 to skip time
            if label in trial.id_results.data:
                data[i] = trial.id_results.data[label].data

        if "Data" in id_group:
            del id_group["Data"]
        if "Time" in id_group:
            del id_group["Time"]
        id_group.create_dataset("Data", data=data, compression="gzip")
        id_group.create_dataset("Time", data=time, compression="gzip") if time is not None else None
        id_group.attrs["Labels"] = ordered
        id_group.attrs["NumSamples"] = n_samples
        id_group.attrs["Metadata"] = str(sample_id.metadata)  # Store metadata as a string representation

    def _save_forces(self, h5f: h5py.File, trial: TrialData) -> None:
        group = h5f.require_group('ForcePlates')
        existing = {plate.attrs.get('Name', f'ForcePlate_{key}'): key for key, plate in group.items()}
        for name, key in list(existing.items()):
            if name not in trial.forces:
                del group[key]
        for name, plate in trial.forces.items():
            if name in existing and existing[name] in group:
                target = group[existing[name]]
            else:
                key = 0
                while str(key) in group:
                    key += 1
                target = group.create_group(str(key))
            write_plate(target, plate)

    def _load_markers(self, h5f: h5py.File) -> Dict[str, MarkerData]:
        if 'Trajectories/Labeled' not in h5f:
            return {}
        traj, group = h5f['Trajectories'], h5f['Trajectories/Labeled']
        labels = self._decode_labels(group.attrs.get('Labels', []))
        if not labels:
            return {}
        data = group['Data'][:]
        if data.ndim != 3 or data.shape[:2] != (len(labels), 4):
            raise ValueError('Marker Data must have shape (markers, 4, frames).')
        time = group['Time'][:] if 'Time' in group else None
        residuals = group['Residuals'][:] if 'Residuals' in group else None
        if residuals is not None and residuals.shape != (len(labels), data.shape[-1]):
            raise ValueError('Residuals do not match marker/frame counts.')
        types = group['Type'][:] if 'Type' in group else None
        virtual = group['Virtual'][:] if 'Virtual' in group else np.zeros(len(labels), dtype=bool)
        masks = group['CameraMasks'][:] if 'CameraMasks' in group else None
        masks_known = group['CameraMasksKnown'][:] if 'CameraMasksKnown' in group else np.zeros(len(labels), dtype=bool)
        result = {}
        for i, label in enumerate(labels):
            result[label] = MarkerData(name=label, x=data[i, 0], y=data[i, 1], z=data[i, 2],
                unit=group.attrs.get('Unit', 'mm'), sampling_rate=traj.attrs.get('SamplingFrequency'),
                time=time, first_frame=int(traj.attrs.get('StartFrame', 0)), virtual=int(virtual[i]),
                residuals=residuals[i] if residuals is not None else None,
                sample_types=types[i] if types is not None else None,
                camera_masks=masks[i] if masks is not None and masks_known[i] else None)
        return result

    def _load_channels(self, h5f, name, cls):
        if name not in h5f:
            return {}
        group = h5f[name]
        labels = self._decode_labels(group.attrs.get('Labels', []))
        if not labels:
            return {}
        units = self._decode_labels(group.attrs.get('Units', ['unknown'] * len(labels)))
        channels = group.attrs.get('Channels', np.arange(len(labels)))
        if len(units) != len(labels) or len(channels) != len(labels):
            raise ValueError(f'{name}: Units and Channels must match Labels.')
        data = group['Data'][:]
        if data.ndim != 2 or data.shape[0] != len(labels):
            raise ValueError(f'{name}: Data must have shape (channels, samples).')
        time = group['Time'][:] if 'Time' in group else None
        return {label: cls(name=label, data=data[i], time=time, unit=units[i],
                           sampling_rate=group.attrs.get('SamplingFrequency'),
                           channel=int(channels[i]) if channels[i] >= 0 else None)
                for i, label in enumerate(labels)}

    def _load_analogs(self, h5f: h5py.File) -> Dict[str, AnalogData]:
        return self._load_channels(h5f, 'Analog', AnalogData)

    def _load_emgs(self, h5f: h5py.File) -> Dict[str, EMGData]:
        return self._load_channels(h5f, 'EMG', EMGData)

    def _load_ik_results(self, h5f: h5py.File) -> Dict[str, Data]:
        """Parse and load IK results to container."""
        import ast 
        ik_group = h5f.get("IKResults")
        if ik_group is None:
            return None

        labels = self._decode_labels(ik_group.attrs.get("Labels", []))
        if not labels:
            return None

        data = ik_group["Data"][:]  # shape: (n_channels, n_samples)
        time = ik_group["Time"][:] if "Time" in ik_group else None  # Optional time dataset
        metadata = ast.literal_eval(ik_group.attrs.get("Metadata", "{}"))  # Convert string representation back to dictionary
        inDegrees = metadata.get("inDegrees", None)
        data_dict = {}
        for i, label in enumerate(labels):
            if label == 'time':
                continue  # Skip the time label, as it's already stored separately
            data_dict[label] = Data(
                name=label,
                data=data[i],
                unit = 'deg' if inDegrees == 'yes' else 'rad',
                time = time,
                )

        return IKResults(name=self.trial_name, time=time, data=data_dict, metadata=metadata, unit='deg' if inDegrees == 'yes' else 'rad')

    def _load_id_results(self, h5f: h5py.File) -> Dict[str, Data]:
        """Parse and load ID results to container."""
        import ast 
        id_group = h5f.get("IDResults")
        if id_group is None:
            return None

        labels = self._decode_labels(id_group.attrs.get("Labels", []))
        if not labels:
            return None

        data = id_group["Data"][:]  # shape: (n_channels, n_samples)
        time = id_group["Time"][:] if "Time" in id_group else None  # Optional time dataset
        metadata = ast.literal_eval(id_group.attrs.get("Metadata", "{}"))  # Convert string representation back to dictionary
        unit = metadata.get("unit", None)
        data_dict = {}
        for i, label in enumerate(labels):
            if label == 'time':
                continue  # Skip the time label, as it's already stored separately
            data_dict[label] = Data(
                name=label,
                data=data[i],
                unit=unit,
                time = time,
                metadata=metadata,
            )

        return IDResults(name=self.trial_name, time=time, data=data_dict, metadata=metadata, unit=unit)

    def _load_forces(self, h5f: h5py.File) -> Dict[str, ForceData]:
        plates = {}
        for group in h5f.get('ForcePlates', {}).values():
            plate = read_plate(group)
            if plate.name in plates:
                raise ValueError(f'Duplicate force plate name: {plate.name}')
            plates[plate.name] = plate
        return plates

    def _save_rigid_bodies(self, h5f: h5py.File, trial: TrialData) -> None:
        """Replace the typed collection; retain legacy opaque data in SourceData.

        Schema 1 stores numbered body groups with Position, Rotation, Markers
        (UTF-8 strings) and optional Time datasets. RPY is derived, not stored.
        Unversioned legacy data is left untouched when no bodies are supplied.
        """
        if 'RigidBodies' in h5f:
            group = h5f['RigidBodies']
            if 'SchemaVersion' not in group.attrs and len(group):
                if not trial.rigid_bodies:
                    return
                target = 'SourceData/RigidBodies'
                if target in h5f:
                    raise ValueError(f'Cannot archive RigidBodies: {target} already exists.')
                h5f.require_group('SourceData').attrs['Scope'] = 'Original recording; not aligned to the processed trial clock'
                h5f.move('RigidBodies', target)
            else:
                del h5f['RigidBodies']
        if not trial.rigid_bodies:
            return
        group = h5f.create_group('RigidBodies', track_order=True)
        group.attrs['SchemaVersion'] = 1
        for index, body in enumerate(trial.rigid_bodies.values()):
            target = group.create_group(str(index))
            target.attrs.update(Name=body.name, NumSamples=body.num_samples, Unit=body.unit)
            if body.sampling_rate is not None:
                target.attrs['SamplingFrequency'] = body.sampling_rate
            for key, value in [('Position', body.position), ('Rotation', body.rotation), ('Time', body.time)]:
                self._replace_dataset(target, key, value)
            target.create_dataset('Markers', data=body.markers, dtype=h5py.string_dtype('utf-8'))

    def _load_rigid_bodies(self, h5f: h5py.File) -> Dict[str, RigidBody]:
        """Load schema 1 poses; leave unversioned legacy payloads opaque."""
        if 'RigidBodies' not in h5f:
            return {}
        collection = h5f['RigidBodies']
        if 'SchemaVersion' not in collection.attrs:
            return {}
        if collection.attrs['SchemaVersion'] != 1:
            raise ValueError('Only rigid body schema 1 is supported.')
        bodies = {}
        for group in collection.values():
            if not isinstance(group, h5py.Group):
                raise ValueError('RigidBodies entries must be body groups.')
            missing = {'Position', 'Rotation', 'Markers'} - set(group)
            if missing:
                raise ValueError(f'{group.name}: missing required datasets: {sorted(missing)}.')
            if not {'Name', 'NumSamples', 'Unit'} <= set(group.attrs):
                raise ValueError(f'{group.name}: Name, NumSamples and Unit attributes are required.')
            name = group.attrs['Name']
            name = name.decode() if isinstance(name, bytes) else name
            if not isinstance(name, str) or name in bodies:
                raise ValueError(f'Invalid or duplicate rigid body name: {name!r}')
            position, rotation = group['Position'][:], group['Rotation'][:]
            if position.ndim != 2 or position.shape[0] != 3 or position.shape[1] == 0:
                raise ValueError(f'{group.name}: Position must have shape (3, n_samples), n_samples > 0.')
            n = position.shape[1]
            if group.attrs['NumSamples'] != n or rotation.shape != (3, 3, n):
                raise ValueError(f'{group.name}: NumSamples and Rotation must match Position.')
            markers = group['Markers']
            if markers.ndim != 1 or h5py.check_string_dtype(markers.dtype) is None:
                raise ValueError(f'{group.name}: Markers must be a one-dimensional string dataset.')
            unit = group.attrs['Unit']
            unit = unit.decode() if isinstance(unit, bytes) else unit
            bodies[name] = RigidBody(name=name, markers=self._decode_labels(markers[:]),
                position=position, rotation=rotation, unit=unit,
                sampling_rate=group.attrs.get('SamplingFrequency'),
                time=group['Time'][:] if 'Time' in group else None)
        return bodies

    @staticmethod
    def _opaque_events(group) -> bool:
        """Identify legacy annotations without enough data to build Event objects."""
        return 'SchemaVersion' not in group.attrs and 'LABELS' not in group.attrs

    def _save_events(self, h5f: h5py.File, trial: TrialData) -> None:
        """Replace events, preserving order, duplicates and source-frame clocks.

        Events are not implicitly cropped when another collection changes.
        Use TrialData.crop_events to explicitly select the desired frame range.
        """
        if 'Events' in h5f:
            group = h5f['Events']
            if self._opaque_events(group) and len(group):
                if not trial.events:
                    return
                target = 'SourceData/Events'
                if target in h5f:
                    raise ValueError(f'Cannot archive Events: {target} already exists.')
                h5f.require_group('SourceData').attrs['Scope'] = 'Original recording; not aligned to the processed trial clock'
                h5f.move('Events', target)
            else:
                del h5f['Events']
        if not trial.events:
            return
        group = h5f.create_group('Events')
        group.attrs.update(SchemaVersion=1, Scope='Trial clock; zero-based source point frames; seconds')
        for key, field in [('Name', 'name'), ('Description', 'description')]:
            group.create_dataset(key, data=[getattr(event, field) for event in trial.events],
                                 dtype=h5py.string_dtype('utf-8'))
        for key, field, dtype in [('Frame', 'frame', np.int64), ('Time', 'time', float)]:
            self._replace_dataset(group, key, np.array([getattr(event, field) for event in trial.events], dtype=dtype))

    def _load_events(self, h5f: h5py.File) -> List[Event]:
        """Load typed events or the converter's legacy Time/LABELS format."""
        if 'Events' not in h5f:
            return []
        group = h5f['Events']
        if self._opaque_events(group):
            return []
        if 'SchemaVersion' not in group.attrs:
            labels = self._decode_labels(group.attrs['LABELS'])
            if not labels:
                return []
            if 'Time' not in group or group['Time'].shape != (len(labels),):
                raise ValueError('Legacy event Time must match LABELS.')
            rate = h5f['Trajectories'].attrs.get('SamplingFrequency') if 'Trajectories' in h5f else None
            if rate is None or not np.isfinite(rate) or rate <= 0:
                raise ValueError('Legacy event frames require a positive trajectory sampling rate.')
            first_frame = int(h5f['Trajectories'].attrs.get('StartFrame', 0))
            first_time = first_frame / rate
            if 'Trajectories/Labeled/Time' in h5f:
                clock = h5f['Trajectories/Labeled/Time'][:]
                if clock.size:
                    validate_clock(clock, len(clock), rate)
                    first_time = float(clock[0])
            descriptions = self._decode_labels(group.attrs.get('DESCRIPTIONS', [''] * len(labels)))
            if len(descriptions) != len(labels):
                raise ValueError('Legacy event annotations must match LABELS.')
            times = group['Time'][:]
            if not np.isfinite(times).all() or (times < 0).any():
                raise ValueError('Event time must be finite and nonnegative.')
            return [Event(name, frame_at_time(time, rate, first_frame, first_time), float(time), description=description)
                    for name, time, description in zip(labels, times, descriptions)]
        if group.attrs['SchemaVersion'] != 1:
            raise ValueError('Only event schema 1 is supported.')
        required = {'Name', 'Frame', 'Time'}
        if not required <= set(group):
            raise ValueError('Events requires Name, Frame and Time datasets.')
        if group['Name'].ndim != 1:
            raise ValueError('Event Name must be a one-dimensional string dataset.')
        count = len(group['Name'])
        values = {}
        for key, field in [('Name', 'name'), ('Frame', 'frame'), ('Time', 'time'),
                           ('Description', 'description')]:
            if key not in group:
                continue
            dataset = group[key]
            if dataset.shape != (count,):
                raise ValueError(f'Event {key} must match the event count.')
            if field in ('name', 'description'):
                if h5py.check_string_dtype(dataset.dtype) is None:
                    raise ValueError(f'Event {key} must contain strings.')
                values[field] = self._decode_labels(dataset[:])
            else:
                values[field] = dataset[:]
        return [Event(**{field: value[index] for field, value in values.items()}) for index in range(count)]

    @staticmethod
    def _validate_marker_units(markers: Dict[str, MarkerData], expected_unit: str) -> None:
        """Validate that all markers have the expected unit.

        Args:
            markers: Dictionary of marker labels to MarkerData.
            expected_unit: The expected unit for all markers.

        Raises:
            ValueError: If any marker has a different unit than expected.
        """
        for label, marker in markers.items():
            if marker.unit != expected_unit:
                raise ValueError(f"Marker '{label}' has unit '{marker.unit}', "
                                 f"expected '{expected_unit}'. This exception occurs when the marker units' \
                                 'between different markers don't match. Please ensure that ' \
                                 'all markers have the same unit before saving to HDF5.")
            
    @staticmethod
    def _decode_labels(raw) -> List[str]:
        """Decode HDF5 label arrays (bytes or str) to plain Python strings."""
        labels = []
        for item in raw:
            if isinstance(item, bytes):
                labels.append(item.decode())
            else:
                labels.append(str(item))
        return labels

    @classmethod
    def from_path(cls, path: str) -> H5Handler:
        """Create an H5Handler from a file path."""
        return cls(path)
