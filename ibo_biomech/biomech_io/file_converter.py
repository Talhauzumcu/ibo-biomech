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

        c3d_path, h5_path = os.fspath(c3d_path), os.fspath(h5_path)
        handler = C3DHandler(c3d_path)
        trial = handler.load_data()
        with h5py.File(h5_path, "w") as h5f:
            meta = h5f.create_group('MetaData')
            meta.attrs.update(PathFile=c3d_path, OriginalFiles=[c3d_path],
                              FileCreationLocal=str(datetime.now()),
                              FileCreationUTC=str(datetime.now(timezone.utc)),
                              LastUpdate=str(datetime.now()))
            for field, argument in [('Project', 'project'), ('ProjectPI', 'project_pi'),
                                    ('SubjectID', 'subject_id'), ('Condition', 'condition'),
                                    ('BodyMass', 'body_mass'), ('BodyHeight', 'body_height'),
                                    ('Sex', 'sex'), ('Age', 'age')]:
                meta.attrs[field] = metadata.get(argument, 'Unknown')
            meta.create_group('Location').attrs.update(Lat='Unknown', Lon='Unknown')
            # The same serializers are used for conversion and processed saves.
            writer = H5Handler(h5_path)
            writer._validate_trial(trial)
            writer._save_markers(h5f, trial)
            writer._save_analogs(h5f, trial)
            writer._save_forces(h5f, trial)
            h5f.create_group('RigidBodies')
            events = h5f.create_group('Events')
            events.attrs['Scope'] = 'Source recording; absolute seconds'
            source_events = handler.c3d_data['parameters'].get('EVENT', {})
            if 'TIMES' in source_events:
                times = np.asarray(source_events['TIMES']['value'])
                events.create_dataset('Time', data=times[0] * 60. + times[1])
                for key in ('LABELS', 'CONTEXTS', 'DESCRIPTIONS'):
                    if key in source_events:
                        events.attrs[key] = source_events[key]['value']
            h5f.create_group('CustomFields')
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
