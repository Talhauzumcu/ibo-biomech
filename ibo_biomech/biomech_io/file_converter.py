"""Conversions between biomechanical file formats.

This module defines :class:`FileConverter`, a stateless collection of static
methods to convert between C3D, the lab's HDF5 format, and the OpenSim TRC/MOT
formats. Conversions targeting OpenSim apply, by default, a -90° rotation about
the X axis and convert positions to metres to match OpenSim's coordinate system.
"""
from typing import List

import h5py
import numpy as np
from ibo_biomech.utils.utils import *
from ibo_biomech.handlers import *
from datetime import datetime, timezone

class FileConverter:
    """Stateless converters between biomechanical file formats.

    All methods are static; no instance is required.
    """

    @staticmethod
    def c3d_to_h5(c3d_path: str, 
                  h5_path: str, 
                  **metadata) -> str:
        """Convert a C3D file to the lab's HDF5 format.

        Builds the full HDF5 group structure (metadata, trajectories, analog,
        force plates, ...) from the parsed C3D data. Fields that C3D does not
        carry are filled with placeholder values to be edited later.

        Args:
            c3d_path: Path to the source C3D file.
            h5_path: Destination path for the HDF5 file.
            metadata: Optional metadata fields to include in the HDF5 file. 
            Keys can include ``project``, ``project_pi``, ``subject_id``, 
            ``condition``, ``body_mass``, ``body_height``, ``sex``, and ``age``. 
            Any missing fields will be filled with ``"Unknown"``.
        Returns:
            The path to the created HDF5 file.
        """

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

        #Create a time vector for all data types. 
        analog_time = np.arange(num_samples) / analog_rate if num_channels > 0 else np.empty((0,), dtype=np.float64)
        marker_time = np.arange(num_frames) / marker_rate

        with h5py.File(h5_path, "w") as h5f:
            meta_group = h5f.create_group("MetaData")
            meta_group.attrs["PathFile"] = c3d_path
            meta_group.attrs["OriginalFiles"] = [c3d_path]
            meta_group.attrs["Project"] = metadata.get("project", "Unknown")
            meta_group.attrs["ProjectPI"] = metadata.get("project_pi", "Unknown")
            meta_group.attrs["SubjectID"] = metadata.get("subject_id", "Unknown")
            meta_group.attrs["Condition"] = metadata.get("condition", "Unknown")
            meta_group.attrs["BodyMass"] = metadata.get("body_mass", "Unknown")
            meta_group.attrs["BodyHeight"] = metadata.get("body_height", "Unknown")
            meta_group.attrs["Sex"] = metadata.get("sex", "Unknown")
            meta_group.attrs["Age"] = metadata.get("age", "Unknown")
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
            labeled_group.create_dataset("Time", data=marker_time, compression="gzip")

            unlabeled_group = traj_group.create_group("Unlabeled")
            unlabeled_group.attrs["NumUnlabeled"] = 0
            unlabeled_group.create_dataset("Data", data=unlabeled_data, compression="gzip")
            unlabeled_group.create_dataset("Type", data=unlabeled_type, compression="gzip")
            unlabeled_group.create_dataset("Time", data=marker_time, compression="gzip")

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
            analog_group.create_dataset("Time", data=analog_time, compression="gzip")

            fp_group = h5f.create_group("ForcePlates")
            for i, (name, fp_data) in enumerate(handler.forces.items()):
                plate_group = fp_group.create_group(str(i))
                
                plate_group.attrs['unit_force'] = fp_data.metadata.get("unit_force") if fp_data.metadata else "Unknown"
                plate_group.attrs['unit_moment'] = fp_data.metadata.get("unit_moment") if fp_data.metadata else "Unknown"
                plate_group.attrs['unit_position'] = fp_data.metadata.get("unit_position") if fp_data.metadata else "Unknown"
                plate_group.attrs['origin'] = fp_data.metadata.get("origin") if fp_data.metadata else 0.0
                plate_group.attrs['Name'] = name
                numSamples = fp_data.force.shape[1]
                plate_group.attrs['NumSamples'] = numSamples
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
                plate_group.create_dataset('Time', data=analog_time, compression="gzip")
                corners = fp_data.metadata.get("corners") if fp_data.metadata else None
                if corners is not None and num_frames > 0:
                    location = np.repeat(corners[:, :, np.newaxis], numSamples, axis=2)
                else:
                    location = np.zeros((4, 3, numSamples), dtype=np.float64)
                plate_group.create_dataset("Location", data=location, compression="gzip")

                origin = fp_data.metadata.get("origin") if fp_data.metadata else None
                if origin is not None and num_frames > 0:
                    position = np.zeros((3, numSamples), dtype=np.float64)
                    position[0:3, :] = np.asarray(origin, dtype=np.float64).reshape(3, 1)
                else:
                    position = np.zeros((3, numSamples), dtype=np.float64)
                
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
        return h5_path

    @staticmethod
    def h5_to_trc(h5_path: str, trc_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> str:
        """Convert an HDF5 file's markers to an OpenSim TRC file.

        Args:
            h5_path: Path to the source HDF5 file.
            trc_path: Destination path for the TRC file.
            axis: Axis to rotate markers about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert marker units to metres.
                Defaults to ``True``.
        Returns:
            The path to the created TRC file.
        """
        h5h = H5Handler(h5_path)
        trial = h5h.load_data()
        trial.rotate_markers(axis, angle)
        time = next(iter(trial.markers.values())).time if trial.markers else None
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
                
        write_trc(trc_path, header_dict, trial.markers, time=time)
        return trc_path

    @staticmethod
    def c3d_to_trc(c3d_path: str, trc_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> str:
        """Convert a C3D file's markers directly to an OpenSim TRC file.

        Goes through a temporary HDF5 file which is deleted afterwards.

        Args:
            c3d_path: Path to the source C3D file.
            trc_path: Destination path for the TRC file.
            axis: Axis to rotate markers about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert marker units to metres.
                Defaults to ``True``.
        Returns:
            The path to the created TRC file.
        """
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_trc(temp_h5_path, trc_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)
        return trc_path

    @staticmethod
    def h5_to_mot(h5_path: str, mot_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> str:
        """Convert an HDF5 file's force plates to an OpenSim MOT file.

        Only ground reaction forces (and moments/CoP) are written.

        Args:
            h5_path: Path to the source HDF5 file.
            mot_path: Destination path for the MOT file.
            axis: Axis to rotate forces about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert force-plate position units to
                metres. Defaults to ``True``.
        Returns:
            The path to the created MOT file.
        """
        h5h = H5Handler(h5_path)
        trial = h5h.load_data()
        trial.rotate_forces(axis, angle)
        time = next(iter(trial.forces.values())).time if trial.forces else None
        if convert_to_meters:
            trial.convert_force_units('m') #c3d data is often saved as mm and Nmm.

        write_mot(mot_path, trial.forces, time=time)
        return mot_path
    
    @staticmethod
    def h5_to_opensim(h5_path: str, mot_path: str, trc_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> List[str]:
        """Convert an HDF5 file to both OpenSim TRC and MOT files.

        Args:
            h5_path: Path to the source HDF5 file.
            mot_path: Destination path for the MOT (forces) file.
            trc_path: Destination path for the TRC (markers) file.
            axis: Axis to rotate data about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert units to metres. Defaults to
                ``True``.
        Returns:
            A list containing the paths to the created MOT and TRC files.
        """
        FileConverter.h5_to_trc(h5_path, trc_path, axis, angle, convert_to_meters)
        FileConverter.h5_to_mot(h5_path, mot_path, axis, angle, convert_to_meters)
        return [mot_path, trc_path]

    @staticmethod
    def c3d_to_mot(c3d_path: str, mot_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> str:
        """Convert a C3D file's force plates directly to an OpenSim MOT file.

        Goes through a temporary HDF5 file which is deleted afterwards.

        Args:
            c3d_path: Path to the source C3D file.
            mot_path: Destination path for the MOT file.
            axis: Axis to rotate forces about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert position units to metres.
                Defaults to ``True``.
        Returns:
            The path to the created MOT file.
        """
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        FileConverter.h5_to_mot(temp_h5_path, mot_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)
        return mot_path
    
    @staticmethod
    def c3d_to_opensim(c3d_path: str, mot_path: str, trc_path: str, axis: str = 'x', angle: float = -90, convert_to_meters: bool = True) -> List[str]:
        """Convert a C3D file directly to both OpenSim TRC and MOT files.

        Goes through a temporary HDF5 file which is deleted afterwards.

        Args:
            c3d_path: Path to the source C3D file.
            mot_path: Destination path for the MOT (forces) file.
            trc_path: Destination path for the TRC (markers) file.
            axis: Axis to rotate data about before export. Defaults to ``'x'``.
            angle: Rotation angle in degrees. Defaults to ``-90``.
            convert_to_meters: Whether to convert units to metres. Defaults to
                ``True``.
        Returns:
            A list containing the paths to the created MOT and TRC files.
        """
        temp_h5_path = ".temp_conversion.h5"
        FileConverter.c3d_to_h5(c3d_path, temp_h5_path)
        result = FileConverter.h5_to_opensim(temp_h5_path, mot_path, trc_path, axis, angle, convert_to_meters)
        os.remove(temp_h5_path)
        return result

