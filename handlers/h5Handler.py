import os
import shutil
from datetime import datetime
from typing import Dict, Optional, Any, List

import h5py
import numpy as np

from containers import AnalogData, ForceData, MarkerData, TrialData, Subject


class H5Handler:
    """Handler for reading and writing lab HDF5 files.

    Metadata fields (SubjectID, Condition, BodyMass, etc.) are intentionally
    outside the scope of this handler. To update them after saving, use h5py
    directly:

        with h5py.File("trial_processed.h5", "r+") as f:
            f["MetaData"].attrs["SubjectID"] = "P01"
            f["MetaData"].attrs["Condition"] = "walking"
    """

    def __init__(self, h5_path: str) -> None:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        self.h5_path = h5_path
        self.trial_name = os.path.splitext(os.path.basename(h5_path))[0]

    def load_data(self) -> TrialData:
        """Read the HDF5 file and return a fully populated TrialData object."""
        with h5py.File(self.h5_path, "r") as h5f:
            markers = self._load_markers(h5f)
            analogs = self._load_analogs(h5f)
            forces = self._load_forces(h5f)

        return TrialData(
            trial_name=self.trial_name,
            markers=markers,
            analogs=analogs,
            forces=forces,
        )

    def load_subject_data(self) -> Subject:
        """Loads trial data and creates a subject instance with the metadata. Created subject instance has the trialdata added."""
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

    def save_data(self, trial: TrialData, out_path: str) -> None:
        """Save a (processed) TrialData back to the lab HDF5 format.

        The original file is used as a template: all groups and attributes are
        copied, then the datasets that TrialData owns (trajectories,
        analog, force plates) are overwritten with current values.  Groups that
        TrialData has no knowledge of (RigidBodies, Events, CustomFields, …)
        are preserved untouched.

        Args:
            trial:    The TrialData object whose data should be written.
            out_path: Destination path for the new HDF5 file.  May be the same
                      as self.h5_path to overwrite in-place (the copy is done
                      before any writes, so the original is safe).
        """

        shutil.copy2(self.h5_path, out_path)

        with h5py.File(out_path, "r+") as h5f:
            self._save_markers(h5f, trial)
            self._save_analogs(h5f, trial)
            self._save_forces(h5f, trial)
            h5f["MetaData"].attrs["LastUpdate"] = str(datetime.now())

        print(f"Saved processed trial to {out_path}")

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

        for i, label in enumerate(ordered):
            if label in trial.markers:
                data[i] = trial.markers[label].get_frame_trajectory()

        del labeled["Data"]
        del labeled["Type"]
        labeled.create_dataset("Data", data=data, compression="gzip")
        labeled.create_dataset("Type", data=type_arr, compression="gzip")
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
        for i, label in enumerate(ordered):
            if label in trial.analogs:
                data[i] = trial.analogs[label].data

        del analog_group["Data"]
        analog_group.create_dataset("Data", data=data, compression="gzip")
        analog_group.attrs["Labels"] = ordered
        analog_group.attrs["NumSamples"] = n_samples
        analog_group.attrs["SamplingFrequency"] = sample_analog.sampling_rate

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
                                  ("Tz", fp_data.Tz)
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

        for i, label in enumerate(labels):
            trajectory = data[i]  # (4, n_frames)
            markers[label] = MarkerData(
                name=label,
                x=trajectory[0],
                y=trajectory[1],
                z=trajectory[2],
                sampling_rate=sampling_rate,
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

        for i, label in enumerate(labels):
            analogs[label] = AnalogData(
                name=label,
                data=data[i],
                sampling_rate=sampling_rate,
                channel=i,
            )

        return analogs

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

            force  = plate["Force"][:]   # (n_samples, 3)
            moment = plate["Moment"][:]  # (n_samples, 3)
            cop    = plate["COP"][:]     # (n_samples, 3)
            location = plate["Location"][:]
            position = plate["Position"][:]
            rotation = plate["Rotation"][:]
            offset = plate["Offset"][:]
            Tz = plate["Tz"][:]

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