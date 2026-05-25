import h5py
import numpy as np
from handlers.c3dHandler import C3DHandler
from datetime import datetime, timezone

class FileConverter:
    """Stateless utility class to convert between biomechanical file formats."""
    
    def c3d_to_h5(self, c3d_path: str, h5_path: str) -> None:
        """Converts a C3D file directly to the lab's HDF5 architecture."""

        handler = C3DHandler(c3d_path)
        handler.load_data()
        labels = list(handler.markers.keys())
        num_labeled = len(labels)
        marker_rate = None
        num_frames = 0
        if num_labeled > 0:
            first_marker = handler.markers[labels[0]]
            marker_rate = first_marker.sampling_rate
            num_frames = len(first_marker.x)

        start_frame = handler.c3d_data['parameters']['POINT']['DATA_START']['value'][0]
        end_frame = start_frame + num_frames - 1 if num_frames > 0 else None

        if num_labeled > 0:
            labeled_data = np.zeros((num_labeled, 4, num_frames), dtype=np.float64)
            for i, label in enumerate(labels):
                marker = handler.markers[label]
                labeled_data[i, :, :] = marker.get_frame_trajectory()
            
            #Not sure if c3d has measured/virtual information. Setting everyting to 1
            labeled_type = np.ones((num_labeled, num_frames), dtype=np.int8)
        else:
            labeled_data = np.empty((0, 4, num_frames), dtype=np.float64)
            labeled_type = np.empty((0, num_frames), dtype=np.int8)

        # C3DHandler currently does not handle unlabeled trajectories.
        unlabeled_data = np.empty((0, 4, num_frames), dtype=np.float64)
        unlabeled_type = np.empty((0, num_frames), dtype=np.int8)

        analog_labels = list(handler.analogs.keys())
        num_channels = len(analog_labels)
        analog_rate = None
        num_samples = 0
        if num_channels > 0:
            first_analog = handler.analogs[analog_labels[0]]
            analog_rate = first_analog.sampling_rate
            num_samples = len(first_analog.data)
            analog_data = np.zeros((num_channels, num_samples), dtype=np.float64)
            for i, label in enumerate(analog_labels):
                analog_data[i, :] = handler.analogs[label].data
        else:
            analog_data = np.empty((0, 0), dtype=np.float64)

        with h5py.File(h5_path, "w") as h5f:
            meta_group = h5f.create_group("MetaData")
            meta_group.attrs["PathFile"] = c3d_path
            meta_group.attrs["OriginalFiles"] = [c3d_path]
            meta_group.attrs["Project"] = ""
            meta_group.attrs["ProjectPI"] = ""
            meta_group.attrs["SubjectID"] = ""
            meta_group.attrs["Condition"] = ""
            meta_group.attrs["BodyMass"] = "Unknown"
            meta_group.attrs["BodyHeight"] = 1.85
            meta_group.attrs["Sex"] = ""
            meta_group.attrs["Age"] = 45
            meta_group.attrs["FileCreationLocal"] = str(datetime.now())
            meta_group.attrs["FileCreationUTC"] = str(datetime.now(timezone.utc))
            meta_group.attrs["LastUpdate"] = str(datetime.now())
            location_group = meta_group.create_group("Location")
            location_group.attrs["Lat"] = "Unknown"
            location_group.attrs["Lon"] = "Unknown"

            traj_group = h5f.create_group("Trajectories")
            traj_group.attrs["SamplingFrequency"] = marker_rate
            traj_group.attrs["NumFrames"] = num_frames if num_frames > 0 else "Unknown"
            traj_group.attrs["StartFrame"] = start_frame
            traj_group.attrs["EndFrame"] = end_frame
            traj_group.attrs["GlobalCoordinateSystem"] = ""

            labeled_group = traj_group.create_group("Labeled")
            labeled_group.attrs["Labels"] = labels
            labeled_group.create_dataset("Data", data=labeled_data, compression="gzip")
            labeled_group.create_dataset("Type", data=labeled_type, compression="gzip")

            unlabeled_group = traj_group.create_group("Unlabeled")
            unlabeled_group.attrs["NumUnlabeled"] = 0
            unlabeled_group.create_dataset("Data", data=unlabeled_data, compression="gzip")
            unlabeled_group.create_dataset("Type", data=unlabeled_type, compression="gzip")

            analog_group = h5f.create_group("Analog")
            analog_group.attrs["BoardName"] = ""
            analog_group.attrs["SamplingFrequency"] = analog_rate
            sampling_factor = None
            if analog_rate is not None and marker_rate is not None:
                sampling_factor = analog_rate / marker_rate
            analog_group.attrs["SamplingFactor"] = sampling_factor
            analog_group.attrs["NumSamples"] = num_samples if num_channels > 0 else None
            analog_group.create_dataset("Data", data=analog_data, compression="gzip")
            analog_group.attrs["Labels"] = analog_labels

            fp_group = h5f.create_group("ForcePlates")
            for i, (name, fp_data) in enumerate(handler.forces.items()):
                plate_group = fp_group.create_group(str(i))
                plate_group.attrs['unit_force'] = fp_data.metadata.get("unit_force") if fp_data.metadata else "Unknown"
                plate_group.attrs['unit_moment'] = fp_data.metadata.get("unit_moment") if fp_data.metadata else "Unknown"
                plate_group.attrs['unit_position'] = fp_data.metadata.get("unit_position") if fp_data.metadata else "Unknown"
                plate_group.attrs['origin'] = fp_data.metadata.get("origin") if fp_data.metadata else 0.0
                plate_group.attrs['Name'] = name
                numSamples = fp_data.force.shape[0]
                plate_group.attrs['NumSamples'] = num_samples
                plate_group.attrs['SamplingFrequency'] = fp_data.sampling_rate
                fp_sampling_factor = None
                if fp_data.sampling_rate is not None and marker_rate is not None:
                    fp_sampling_factor = fp_data.sampling_rate / marker_rate
                plate_group.attrs['SamplingFactor'] = fp_sampling_factor
                plate_group.attrs['Filter'] = "none"
                plate_group.create_dataset("COP", data=fp_data.cop, compression="gzip")
                plate_group.create_dataset("Force", data=fp_data.force, compression="gzip")
                plate_group.create_dataset("Moment", data=fp_data.moment, compression="gzip")

                corners = fp_data.metadata.get("corners") if fp_data.metadata else None
                if corners is not None and num_frames > 0:
                    location = np.repeat(corners[:, :, np.newaxis], num_frames, axis=2)
                else:
                    location = None
                plate_group.create_dataset("Location", data=location, compression="gzip")

                origin = fp_data.metadata.get("origin") if fp_data.metadata else None
                if origin is not None and num_frames > 0:
                    position = np.zeros((4, num_frames), dtype=np.float64)
                    position[0:3, :] = np.asarray(origin, dtype=np.float64).reshape(3, 1)
                else:
                    position = None
                plate_group.create_dataset("Position", data=position, compression="gzip")
                plate_group.create_dataset("Rotation", data=np.zeros((3,3, numSamples)), compression="gzip")
                plate_group.create_dataset("Offset", data=np.zeros((3,)), compression="gzip")
                plate_group.attrs["CoordinateSystem"] = 1

            h5f.create_group("RigidBodies")
            h5f.create_group("Events")
            h5f.create_group("CustomFields")

        print(f"Successfully converted {c3d_path} to {h5_path}")
