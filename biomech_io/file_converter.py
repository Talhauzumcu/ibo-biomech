import h5py
import numpy as np
from utils.utils import *
from handlers import *
from datetime import datetime, timezone

class FileConverter:
    """Stateless utility class to convert between biomechanical file formats."""
    
    @staticmethod
    def c3d_to_h5(c3d_path: str, h5_path: str) -> None:
        """Converts a C3D file directly to the lab's HDF5 architecture."""

        handler = C3DHandler(c3d_path)
        handler.load_data()
        labels = list(handler.markers.keys())
        num_labeled = len(labels)
        marker_rate = handler.c3d_data['header']['points']['frame_rate']
        start_frame = handler.c3d_data['header']['points']['first_frame']
        end_frame = handler.c3d_data['header']['points']['last_frame']
        num_frames = end_frame - start_frame + 1 

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

        labeled_residuals = handler.c3d_data['data']['meta_points']['residuals']
        # C3DHandler currently does not handle unlabeled trajectories.
        unlabeled_data = np.empty((0, 4, num_frames), dtype=np.float64)
        unlabeled_type = np.empty((0, num_frames), dtype=np.int8)

        analog_labels = list(handler.analogs.keys())
        num_channels = len(analog_labels)
        analog_rate = handler.c3d_data['header']['analogs']['frame_rate'] if num_channels > 0 else None
        num_samples = 0
        if num_channels > 0:
            first_analog = handler.analogs[analog_labels[0]]
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
            meta_group.attrs["Project"] = "Unknown"
            meta_group.attrs["ProjectPI"] = "Unknown"
            meta_group.attrs["SubjectID"] = "Unknown"
            meta_group.attrs["Condition"] = "Unknown"
            meta_group.attrs["BodyMass"] = "Unknown"
            meta_group.attrs["BodyHeight"] = "Unknown"
            meta_group.attrs["Sex"] = "Unknown"
            meta_group.attrs["Age"] = "Unknown"
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
            labeled_group.attrs["NumLabeled"] = len(labels)
            labeled_group.attrs['Unit'] = handler.c3d_data['parameters']['POINT']['UNITS']['value'][0]
            labeled_group.create_dataset("Data", data=labeled_data, compression="gzip")
            labeled_group.create_dataset("Type", data=labeled_type, compression="gzip")
            labeled_group.create_dataset("Residuals", data=labeled_residuals, compression="gzip")

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
                plate_group.create_dataset('Tz', data=fp_data.Tz, compression="gzip")
                corners = fp_data.metadata.get("corners") if fp_data.metadata else None
                if corners is not None and num_frames > 0:
                    location = np.repeat(corners[:, :, np.newaxis], num_frames, axis=2)
                else:
                    location = None
                plate_group.create_dataset("Location", data=location, compression="gzip")

                origin = fp_data.metadata.get("origin") if fp_data.metadata else None
                if origin is not None and num_frames > 0:
                    position = np.zeros((3, num_frames), dtype=np.float64)
                    position[0:3, :] = np.asarray(origin, dtype=np.float64).reshape(3, 1)
                else:
                    position = None
                
                #Since c3d files don't hold these information, initial fileconversion uses static values. 
                #These can be updated directly from the h5 file and resaved.
                plate_group.create_dataset("Position", data=position, compression="gzip")
                plate_group.create_dataset("Rotation", data=np.zeros((3,3, numSamples)), compression="gzip")
                plate_group.create_dataset("Offset", data=np.zeros((3,)), compression="gzip")
                plate_group.attrs["CoordinateSystem"] = 0 # Is this information available in c3d files? If not where to get it? or what is default?
            
            h5f.create_group("RigidBodies") #C3d doesn't hold these? have to be custom created
            events_group = h5f.create_group("Events")
            h5f.create_group("CustomFields")

        print(f"Successfully converted {c3d_path} to {h5_path}")

    @staticmethod
    def h5_to_trc(h5_path: str, trc_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Converts and h5 file to TRC format. Default values are set to convert from lab's h5 architecture to OpenSim TRC format."""
        h5h = H5Handler(h5_path)
        trial = h5h.load_data()
        trial.rotate_markers(axis, angle)

        if convert_to_meters:
            trial.convert_marker_units('m')
        header_dict = {
        'data_rate': trial.marker_rate,
        'camera_rate': trial.marker_rate,
        'num_frames': len(trial.markers[trial.marker_labels[0]].x),
        'num_markers': len(trial.markers),
        'units': trial.markers[trial.marker_labels[0]].unit if trial.marker_labels else 'M',
        'orig_data_rate': trial.marker_rate,
        'orig_data_start_frame': 0,
        'orig_num_frames': len(trial.markers[trial.marker_labels[0]].x),
        'marker_labels': trial.marker_labels
        }
                
        write_trc(trc_path, header_dict, trial.markers)


    @staticmethod
    def c3d_to_trc(c3d_path: str, trc_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Convenience method to convert directly from c3d to trc."""
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_trc(temp_h5_path, trc_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)

    @staticmethod
    def h5_to_mot(h5_path: str, mot_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Converts an h5 file to OpenSim MOT format. This will include only GRFs."""
        h5h = H5Handler(h5_path)
        trial = h5h.load_data()
        trial.rotate_forces(axis, angle)
        if convert_to_meters:
            trial.convert_force_units('m') #c3d data is often saved as mm and Nmm.

        write_mot(mot_path, trial.forces)

    @staticmethod
    def h5_to_opensim(h5_path: str, mot_path: str, trc_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Convenience method to convert an h5 file to OpenSim TRC and MOT formats."""
        FileConverter.h5_to_trc(h5_path, trc_path, axis, angle, convert_to_meters)
        FileConverter.h5_to_mot(h5_path, mot_path, axis, angle, convert_to_meters)

    @staticmethod
    def c3d_to_trc(c3d_path: str, trc_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Convenience method to convert directly from c3d to trc."""
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_trc(temp_h5_path, trc_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)
    
    @staticmethod
    def c3d_to_mot(c3d_path: str, mot_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Convenience method to convert directly from c3d to mot."""
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_mot(temp_h5_path, mot_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)
        
    @staticmethod
    def c3d_to_opensim(c3d_path: str, mot_path: str, trc_path: str, axis='x', angle=-90, convert_to_meters: bool = True) -> None:
        """Convenience method to convert directly from c3d to OpenSim TRC and MOT formats."""
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_opensim(temp_h5_path, mot_path, trc_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)