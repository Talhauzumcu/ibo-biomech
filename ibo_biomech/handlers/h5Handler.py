from __future__ import annotations
import os
import shutil
from datetime import datetime
from typing import Dict, Optional, Any, List

import h5py
import numpy as np

from ibo_biomech.containers import AnalogData, ForceData, MarkerData, EMGData, TrialData, Subject, IKResults, IDResults, Data


class H5Handler:
    """Read and write the lab's HDF5 trial format.

    Reading returns a fully populated :class:`~ibo_biomech.containers.TrialData`.
    Writing uses the original file as a template and overwrites only the datasets
    that :class:`TrialData` owns, preserving all other groups.

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
                  load_id_results: bool = True ) -> TrialData:
        """Read the HDF5 file and return a fully populated trial. Accepts flags to selectively load specific data types for faster loading times

        Args:
            load_markers: Whether to load marker data.
            load_analogs: Whether to load analog data.
            load_forces: Whether to load force plate data.
            load_emgs: Whether to load EMG data.
            load_ik_results: Whether to load inverse kinematics results.
            load_id_results: Whether to load inverse dynamics results.
        Returns:
            A :class:`~ibo_biomech.containers.TrialData` with markers, analogs,
            forces and metadata.
        """

        with h5py.File(self.h5_path, "r") as h5f:
            markers = self._load_markers(h5f) if load_markers else None
            analogs = self._load_analogs(h5f) if load_analogs else None
            forces = self._load_forces(h5f) if load_forces else None
            emgs = self._load_emgs(h5f) if load_emgs else None
            ik_results = self._load_ik_results(h5f) if load_ik_results else None
            id_results = self._load_id_results(h5f) if load_id_results else None
            metadata = dict(h5f["MetaData"].attrs)

        return TrialData(
            name=self.trial_name,
            markers=markers,
            analogs=analogs,
            forces=forces,
            emgs=emgs,
            ik_results=ik_results,
            id_results=id_results,
            metadata=metadata,
        )

    def load_subject_data(self) -> Subject:
        """Read the file into a subject populated from its metadata.

        Builds a :class:`~ibo_biomech.containers.Subject` from the file's
        ``MetaData`` attributes and attaches the loaded trial to it.

        Returns:
            A :class:`~ibo_biomech.containers.Subject` containing the trial.
        """
        trial_data = self.load_trial_data()
        with h5py.File(self.h5_path, "r") as h5f:
            meta = h5f["MetaData"].attrs
            subject_id = meta.get("SubjectID", "")
            condition = meta.get("Condition", "")
            body_mass = meta.get("BodyMass", None)
            body_height = meta.get("BodyHeight", None)
            age = meta.get("Age", None)

        subject = Subject(id=subject_id, condition=condition, body_mass=body_mass, body_height=body_height, age=age)
        subject.add_trial(trial_name=trial_data.trial_name, trial_data=trial_data)
        return subject

    def save_data(self, trial: TrialData, out_path: str) -> str:
        """Save a (processed) TrialData back to the lab HDF5 format.

        The original file is used as a template: all groups and attributes are
        copied, then the datasets that TrialData owns (trajectories,
        analog, force plates) are overwritten with current values.  Groups that
        TrialData has no knowledge of (RigidBodies, Events, CustomFields, …)
        are preserved untouched.

        Args:
            trial:    The TrialData object whose data should be written.
            out_path: Destination path for the new HDF5 file.  May be the same
                      as self.h5_path to overwrite in-place.
        Returns:
            The path to the saved file (out_path).
        """

        shutil.copy2(self.h5_path, out_path)

        with h5py.File(out_path, "r+") as h5f:
            self._save_markers(h5f, trial)
            self._save_analogs(h5f, trial)
            self._save_forces(h5f, trial)
            self._save_emgs(h5f, trial)
            self._save_ik_results(h5f, trial)
            self._save_id_results(h5f, trial)
            h5f["MetaData"].attrs["LastUpdate"] = str(datetime.now())

        print(f"Saved processed trial to {out_path}")
        return out_path

    def modify_metadata(self, updates: Dict[str, Any], out_path: Optional[str] = None) -> H5Handler:
        """Modify metadata attributes in the HDF5 file.

        Args:
            updates: A dictionary of metadata attributes to update.
            out_path: Optional path to save the modified file. If None, the original file is overwritten.
        Returns:
            H5Handler: A new H5Handler instance pointing to the modified file.
        """
        if out_path is None:
            out_path = self.h5_path

        shutil.copy2(self.h5_path, out_path)

        with h5py.File(out_path, "r+") as h5f:
            for key, value in updates.items():
                h5f["MetaData"].attrs[key] = value

        print(f"Modified metadata in {out_path}")
        return self.from_path(out_path)

    def _save_markers(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite labeled trajectory datasets from trial.markers."""
        labeled = h5f["Trajectories/Labeled"]
        existing_labels = self._decode_labels(labeled.attrs.get("Labels", []))

        # Rebuild the data array in the same label order as the original file
        # so that the Labels attribute and the Data rows stay in sync.
        # Unknown labels (added in memory) are appended at the end.
        ordered = existing_labels + [l for l in trial.markers if l not in existing_labels]

        n_markers = len(ordered)
        if n_markers == 0:
            return

        sample_marker = next(iter(trial.markers.values()))
        n_frames = len(sample_marker.x)

        data = np.zeros((n_markers, 4, n_frames), dtype=np.float64)
        type_arr = np.ones((n_markers, n_frames), dtype=np.int8)
        time = sample_marker.time if sample_marker.time is not None else np.zeros(n_frames, dtype=np.float64)
        for i, label in enumerate(ordered):
            if label in trial.markers:
                data[i] = trial.markers[label].get_frame_trajectory()

        del labeled["Data"]
        del labeled["Type"]
        del labeled["Time"]
        labeled.create_dataset("Data", data=data, compression="gzip")
        labeled.create_dataset("Type", data=type_arr, compression="gzip")
        labeled.create_dataset("Time", data=time, compression="gzip") if time is not None else None
        labeled.attrs["Labels"] = ordered

        traj = h5f["Trajectories"]
        traj.attrs["NumFrames"] = n_frames
        traj.attrs["SamplingFrequency"] = sample_marker.sampling_rate

    def _save_analogs(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite analog datasets from trial.analogs."""
        if not trial.analogs:
            return

        analog_group = h5f["Analog"]
        existing_labels = self._decode_labels(analog_group.attrs.get("Labels", []))
        ordered = existing_labels + [l for l in trial.analogs if l not in existing_labels]

        sample_analog = next(iter(trial.analogs.values()))
        n_samples = len(sample_analog.data)

        data = np.zeros((len(ordered), n_samples), dtype=np.float64)
        time = sample_analog.time if sample_analog.time is not None else np.zeros(n_samples, dtype=np.float64)
        for i, label in enumerate(ordered):
            if label in trial.analogs:
                data[i] = trial.analogs[label].data

        del analog_group["Data"]
        del analog_group['Time']
        analog_group.create_dataset("Data", data=data, compression="gzip")
        analog_group.create_dataset("Time", data=time, compression="gzip") if time is not None else None
        analog_group.attrs["Labels"] = ordered
        analog_group.attrs["NumSamples"] = n_samples
        analog_group.attrs["SamplingFrequency"] = sample_analog.sampling_rate

    def _save_emgs(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite EMG datasets from trial.emgs."""
        if not trial.emgs:
            return

        if 'EMG' not in h5f.keys():
            emg_group = h5f.create_group("EMG")
            emg_group.attrs["Labels"] = []
            emg_group.attrs["NumSamples"] = 0
            emg_group.attrs["SamplingFrequency"] = 0.0

        existing_labels = self._decode_labels(emg_group.attrs.get("Labels", []))
        ordered = existing_labels + [l for l in trial.emgs if l not in existing_labels]
        sample_emg = next(iter(trial.emgs.values()))
        n_samples = len(sample_emg.data)

        data = np.zeros((len(ordered), n_samples), dtype=np.float64)
        time = sample_emg.time if sample_emg.time is not None else np.zeros(n_samples, dtype=np.float64)
        for i, label in enumerate(ordered):
            if label in trial.emgs:
                data[i] = trial.emgs[label].data

        if "Data" in emg_group:
            del emg_group["Data"]
        if "Time" in emg_group:
            del emg_group["Time"]
            
        emg_group.create_dataset("Data", data=data, compression="gzip")
        emg_group.create_dataset("Time", data=time, compression="gzip") if time is not None else None
        emg_group.attrs["Labels"] = ordered
        emg_group.attrs["NumSamples"] = n_samples
        emg_group.attrs["SamplingFrequency"] = sample_emg.sampling_rate

    def _save_ik_results(self, h5f: h5py.File, trial: TrialData) -> None:
        """Overwrite IK results datasets from trial.ik_results."""
        if not trial.ik_results:
            return

        if 'IKResults' not in h5f.keys():
            ik_group = h5f.create_group("IKResults")
            ik_group.attrs["Labels"] = ['time']
            ik_group.attrs["NumSamples"] = 0

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
        """Overwrite force plate datasets from trial.forces."""
        if not trial.forces:
            return

        fp_group = h5f["ForcePlates"]

        # Map existing subgroups by their Name attribute for lookup
        existing_by_name = {}
        for key in fp_group:
            name = fp_group[key].attrs.get("Name", "")
            if isinstance(name, bytes):
                name = name.decode()
            existing_by_name[name] = key

        for fp_name, fp_data in trial.forces.items():
            if fp_name not in existing_by_name:
                # New plate added in memory, append at the next integer index
                next_key = str(len(fp_group))
                plate = fp_group.create_group(next_key)
                plate.attrs["Name"] = fp_name
                plate.attrs["CoordinateSystem"] = 0
                plate.attrs["Filter"] = "none"
                meta = fp_data.metadata or {}
                plate.attrs["unit_force"]    = meta.get("unit_force", "")
                plate.attrs["unit_moment"]   = meta.get("unit_moment", "")
                plate.attrs["unit_position"] = meta.get("unit_position", "")
                plate.attrs["origin"]        = meta.get("origin", 0.0)
            else:
                plate = fp_group[existing_by_name[fp_name]]

            n_samples = fp_data.force.shape[0]
            plate.attrs["NumSamples"]       = n_samples
            plate.attrs["SamplingFrequency"] = fp_data.sampling_rate

            for ds_name, arr in [("Force", fp_data.force),
                                  ("Moment", fp_data.moment),
                                  ("COP", fp_data.cop),
                                  ("Location", fp_data.location),
                                  ("Position", fp_data.position),
                                  ("Rotation", fp_data.rotation),
                                  ("Offset", fp_data.offset),
                                  ("Tz", fp_data.Tz),
                                  ("Time", fp_data.time)
                                  ]:
                if ds_name in plate:
                    del plate[ds_name]
                plate.create_dataset(ds_name, data=arr, compression="gzip")

    def _load_markers(self, h5f: h5py.File) -> Dict[str, MarkerData]:
        """Parse labeled trajectories into {label: MarkerData}."""
        markers: Dict[str, MarkerData] = {}

        traj_group = h5f.get("Trajectories")
        if traj_group is None:
            return markers

        sampling_rate = traj_group.attrs.get("SamplingFrequency")
        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)

        labeled_group = traj_group.get("Labeled")
        if labeled_group is None:
            return markers

        labels = self._decode_labels(labeled_group.attrs.get("Labels", []))
        data = labeled_group["Data"][:]  # shape: (n_markers, 4, n_frames)
        time = labeled_group['Time'][:] if 'Time' in labeled_group else None  # Optional time dataset
        for i, label in enumerate(labels):
            trajectory = data[i]  # (4, n_frames)
            markers[label] = MarkerData(
                name=label,
                x=trajectory[0],
                y=trajectory[1],
                z=trajectory[2],
                sampling_rate=sampling_rate,
                time=time
            )

        return markers

    def _load_analogs(self, h5f: h5py.File) -> Dict[str, AnalogData]:
        """Parse analog channels into {label: AnalogData}."""
        analogs: Dict[str, AnalogData] = {}

        analog_group = h5f.get("Analog")
        if analog_group is None:
            return analogs

        sampling_rate = analog_group.attrs.get("SamplingFrequency")
        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)

        labels = self._decode_labels(analog_group.attrs.get("Labels", []))
        if not labels:
            return analogs

        data = analog_group["Data"][:]  # shape: (n_channels, n_samples)
        time = analog_group['Time'][:] if 'Time' in analog_group else None  # Optional time dataset
        for i, label in enumerate(labels):
            analogs[label] = AnalogData(
                name=label,
                data=data[i],
                sampling_rate=sampling_rate,
                channel=i,
                time=time
            )

        return analogs

    def _load_emgs(self, h5f: h5py.File) -> Dict[str, EMGData]:
        """Parse EMG channels into {label: EMGData}."""
        emgs: Dict[str, EMGData] = {}

        emg_group = h5f.get("EMG")
        if emg_group is None:
            return emgs

        sampling_rate = emg_group.attrs.get("SamplingFrequency")
        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)

        labels = self._decode_labels(emg_group.attrs.get("Labels", []))
        if not labels:
            return emgs

        data = emg_group["Data"][:]  # shape: (n_channels, n_samples)
        time = emg_group['Time'][:] if 'Time' in emg_group else None  # Optional time dataset

        for i, label in enumerate(labels):
            emgs[label] = EMGData(
                name=label,
                data=data[i],
                sampling_rate=sampling_rate,
                channel=i,
                time=time
            )

        return emgs

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
        """Parse force plate groups into {name: ForceData}."""
        forces: Dict[str, ForceData] = {}
        fp_group = h5f.get("ForcePlates")
        if fp_group is None:
            return forces

        for key in fp_group:
            plate = fp_group[key]
            name = plate.attrs.get("Name", f"ForcePlate_{key}")
            if isinstance(name, bytes):
                name = name.decode()

            force = plate["Force"][:]
            n_samples = force.shape[1]
            time = plate["Time"][:] if "Time" in plate else None
            moment = plate["Moment"][:] if "Moment" in plate else np.zeros_like(force)
            cop = plate["COP"][:] if "COP" in plate else np.zeros_like(force)
            location = (plate["Location"][:]
                        if "Location" in plate
                        else np.zeros((3, 4, n_samples)))
            position = (plate["Position"][:]
                        if "Position" in plate
                        else np.zeros((3, n_samples)))
            rotation = (plate["Rotation"][:]
                        if "Rotation" in plate
                        else np.zeros((3, 3, n_samples)))
            offset = (plate["Offset"][:]
                      if "Offset" in plate
                      else np.zeros((3, 1)))
            Tz = (plate["Tz"][:] if "Tz" in plate else np.zeros(n_samples))

            sampling_rate = plate.attrs.get("SamplingFrequency")
            if sampling_rate is not None:
                sampling_rate = float(sampling_rate)

            metadata = {
                "unit_force":    plate.attrs.get("unit_force", ""),
                "unit_moment":   plate.attrs.get("unit_moment", ""),
                "unit_position": plate.attrs.get("unit_position", ""),
                "origin":        plate.attrs.get("origin", None),
            }

            forces[name] = ForceData(
                name=name,
                force=force,
                moment=moment,
                cop=cop,
                location=location,
                position=position,
                rotation=rotation,
                offset=offset,
                metadata=metadata,
                Tz = Tz,
                sampling_rate=sampling_rate,
                time=time
            )

        return forces

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