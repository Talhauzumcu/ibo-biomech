from ezc3d import c3d
import os
from typing import Dict, List, Optional, Tuple, Union
from ibo_biomech.containers import AnalogData, ForceData, MarkerData
from ibo_biomech.utils.utils import get_rotation_matrix
import numpy as np
from copy import deepcopy

class C3DHandler:
    """Load, manipulate and export C3D motion-capture files.

    Wraps the :mod:`ezc3d` reader. After :meth:`load_data`, parsed channels are
    available as the :attr:`markers`, :attr:`analogs` and :attr:`forces`
    dictionaries, and the raw ezc3d structure is kept on :attr:`c3d_data` so the
    file can be modified and re-written.

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
        self.c3d_data = None
        self.markers: Dict[str, MarkerData] = {}
        self.analogs: Dict[str, AnalogData] = {}
        self.forces: Dict[str, ForceData] = {}
    
    #DEPRECATED
    # @staticmethod
    # def load_trial(filepath: str, trial_name: Optional[str] = None) -> TrialData:
    #     """
    #     Static method to load C3D file directly into a TrialData object.
    #     This method doesn't keep the handler instance, making it memory efficient.
        
    #     Args:
    #         filepath: Path to the C3D file
    #         trial_name: Name for the trial (defaults to filename without extension)
            
    #     Returns:
    #         TrialData object populated with markers, analogs, and forces
            
    #     Example:
    #         trial_1 = C3DHandler.load_trial("path/to/file.c3d")
    #         # Access data: trial_1.markers, trial_1.analogs, trial_1.forces
    #     """
    #     # Create temporary handler instance
    #     handler = C3DHandler(filepath)
        
    #     # Determine trial name
    #     if trial_name is None:
    #         trial_name = os.path.splitext(os.path.basename(filepath))[0]
        
    #     # Create TrialData object with the parsed data
    #     trial = TrialData(
    #         trial_name=trial_name,
    #         markers=handler.markers,
    #         analogs=handler.analogs,
    #         forces=handler.forces
    #     )
        
    #     # Handler goes out of scope and gets garbage collected
    #     return trial
    

    def load_data(self) -> None:
        """Read the C3D file and parse it into container objects.

        Populates :attr:`markers`, :attr:`analogs` and :attr:`forces` and stores
        the raw structure on :attr:`c3d_data`.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the C3D file cannot be read or parsed.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"C3D file not found: {self.filepath}")
        
        try:
            self.c3d_data = c3d(self.filepath, extract_forceplat_data=True)
            self._parse_markers()
            self._parse_analogs()
            self._parse_force_plates()     
            self.is_loaded = True

        except Exception as e:
            raise Exception(f"Error loading C3D file: {str(e)}")
    
    def _parse_markers(self) -> None:
        """Parse marker data from C3D file into MarkerData containers."""
        try:
            points = self.c3d_data['data']['points']
            marker_labels = self.c3d_data['parameters']['POINT']['LABELS']['value']
            point_rate = self.c3d_data['parameters']['POINT']['RATE']['value'][0]
            unit = self.c3d_data['parameters']['POINT']['UNITS']['value'][0]
            for i, label in enumerate(marker_labels):
                x = points[0, i, :]
                y = points[1, i, :]
                z = points[2, i, :]
                
                marker = MarkerData(
                    name=label.strip(),
                    x=x,
                    y=y,
                    z=z,
                    unit=unit.strip(),
                    sampling_rate=point_rate
                )
                
                self.markers[label.strip()] = marker
                
        except KeyError as e:
            print(f"Warning: Could not parse marker data: {str(e)}")
    
    def _parse_analogs(self) -> None:
        """Parse analog data from C3D file into AnalogData containers."""
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

                #Don't overwrite existing labels
                j = 2
                while True:
                    if label.strip() in self.analogs.keys():
                        # print(f"Warning: Duplicate analog label found: {label.strip()}. adding increment to the label.")
                        label = f"{label.strip()}_{j}"
                        j += 1
                    else:   
                        break
                
                analog = AnalogData(
                    name=label.strip(),
                    data=analog_signal,
                    sampling_rate=analog_rate,
                    unit=units[i].strip() if i < len(units) else "",
                    channel=i
                )
                
                self.analogs[label.strip()] = analog
                
        except KeyError as e:
            print(f"Warning: Could not parse analog data: {str(e)}")
    
    def _parse_force_plates(self) -> None:
        """Parse force plate data from C3D file into ForceData containers."""
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

            self.forces[f"forceplate_{i}"] = ForceData(
                name=f"forceplate_{i}",
                force=plate['force'],
                moment=plate['moment'],
                cop=plate['center_of_pressure'],
                Tz = plate['Tz'][2,:],
                metadata=metadata,
                sampling_rate=self.c3d_data['parameters']['ANALOG']['RATE']['value'][0]
            )

    def add_marker(self, marker: MarkerData) -> None:
        """Add a marker to both the handler and the underlying C3D structure.

        Args:
            marker: The marker to add.
        """
        self.markers[marker.name] = marker
        self._add_marker_to_c3d(marker)

    def _add_marker_to_c3d(self, marker: MarkerData) -> None:
        """Append a marker to the raw ezc3d point data and labels.

        Args:
            marker: The marker to add. Skipped if its name already exists.

        Raises:
            Exception: If no C3D data has been loaded.
        """
        if self.c3d_data is None:
            raise Exception("No C3D data loaded to add marker to.")

        # self.c3d_data.add_parameter('POINT', marker.name, [1,2,3])

        if marker.name not in self.c3d_data['parameters']['POINT']['LABELS']['value']:
            self.c3d_data['parameters']['POINT']['LABELS']['value'].append(marker.name)
            marker_frame = marker.get_frame_trajectory().reshape(4,1,marker.x.shape[0])
            self.c3d_data['data']['points'] = np.concatenate((self.c3d_data['data']['points'], marker_frame), axis=1)
        else:
            print(f"Marker {marker.name} already exists in C3D data. Skipping addition to C3D structure.")

    def write_c3d(self, output_filepath: str) -> None:
        """Write the current C3D structure to disk.

        Deletes the cached ``meta_points`` so ezc3d regenerates them for any
        markers added in memory.

        Args:
            output_filepath: Destination path for the C3D file.

        Returns:
            The path that was written.

        Raises:
            Exception: If no C3D data has been loaded.
        """
        #Check if all marker data exist in the c3d structure
        if self.c3d_data is None:
            raise Exception("No C3D data loaded to write.")

        #Delete meta points to let c3d create new ones
        del self.c3d_data['data']['meta_points']
        self.c3d_data.write(output_filepath)
        return output_filepath
    
    def rotate_data(self, axis: str, angle_deg: float) -> None:
        """
        Rotate all marker and force data around a specified axis by a given angle.
        
        Args:
            axis: Axis to rotate around ('x', 'y', or 'z')
            angle_deg: Rotation angle in degrees
        """
        rotation_matrix = get_rotation_matrix(axis, angle_deg)
        
        # Rotate marker data
        for marker_name, marker in self.markers.items():
            coords = np.column_stack([marker.x, marker.y, marker.z])  # (n_samples, 3)
            rotated = coords @ rotation_matrix.T  # Apply rotation
            marker.x = rotated[:, 0]
            marker.y = rotated[:, 1]
            marker.z = rotated[:, 2]
        
        # Rotate force data
        for force_name, force in self.forces.items():
            # Rotate force vectors (n_samples, 3)
            rotated_force = force.force @ rotation_matrix.T
            rotated_force = np.nan_to_num(rotated_force)  # Handle NaN values by setting them to 0
            force.force = rotated_force
            
            # Rotate moment vectors (n_samples, 3)
            rotated_moment = force.moment @ rotation_matrix.T
            rotated_moment = np.nan_to_num(rotated_moment)  # Handle NaN values by setting them to 0
            force.moment = rotated_moment
            
            # Rotate center of pressure (n_samples, 3)
            rotated_cop = force.cop @ rotation_matrix.T
            rotated_cop = np.nan_to_num(rotated_cop)  # Handle NaN values by setting them to 0
            force.cop = rotated_cop
            
            # Rotate corners if they exist in metadata
            if 'corners' in force.metadata:
                corners = force.metadata['corners']  # (3, 4) - 4 corners, xyz each
                rotated_corners = rotation_matrix @ corners
                force.metadata['corners'] = rotated_corners
            
            # Rotate origin if it exists in metadata
            if 'origin' in force.metadata:
                origin = force.metadata['origin'].reshape(-1)
                rotated_origin = rotation_matrix @ origin
                force.metadata['origin'] = rotated_origin

    def convert_to_meters(self) -> None:
        """Convert all position data (markers and force CoP) to metres.

        The source unit is read from the C3D ``POINT.UNITS`` parameter; ``mm``
        and ``cm`` are converted, ``m`` is a no-op.

        Raises:
            ValueError: If the C3D unit is not one of ``'mm'``, ``'cm'`` or
                ``'m'``.
        """
        current_unit = self.c3d_data['parameters']['POINT']['UNITS']['value'][0]
        
        # Determine conversion factor
        if current_unit.lower() == 'mm':
            scale_factor = 0.001  # mm to m
        elif current_unit.lower() == 'cm':
            scale_factor = 0.01   # cm to m
        elif current_unit.lower() == 'm':
            print("Data is already in meters, no conversion needed.")
            return
        else:
            raise ValueError(f"Unknown unit: {current_unit}. Supported units: 'mm', 'cm', 'm'")
        
        # print(f"Converting position data from {current_unit} to meters (scale factor: {scale_factor})...")
        
        # Convert marker positions
        for marker_name, marker in self.markers.items():
            marker.x = marker.x * scale_factor
            marker.y = marker.y * scale_factor
            marker.z = marker.z * scale_factor
        
        # Convert force plate position data (CoP, corners, origin)
        for force_name, force in self.forces.items():
            # Convert center of pressure (n_samples, 3)
            force.cop = force.cop * scale_factor
            
            # Convert corners if they exist in metadata
            if 'corners' in force.metadata:
                force.metadata['corners'] = force.metadata['corners'] * scale_factor
            
            # Convert origin if it exists in metadata
            if 'origin' in force.metadata:
                force.metadata['origin'] = force.metadata['origin'] * scale_factor
            
            # Update unit_position in metadata
            if 'unit_position' in force.metadata:
                force.metadata['unit_position'] = 'm'
        
        # print("Conversion to meters complete.")

    def write_trc(self, output_filepath: str, data_rate: float = None, 
                  camera_rate: float = None, units: str = 'M') -> None:
        """
        Write marker data to a TRC file compatible with OpenSim.
        
        Args:
            output_filepath: Path for the output TRC file
            data_rate: Data rate in Hz (defaults to marker sampling rate)
            camera_rate: Camera rate in Hz (defaults to data_rate)
            units: Units for marker positions ('mm' or 'M')
        """
        if not self.markers:
            raise Exception("No marker data to write.")
        
        # Get first marker to determine number of frames and sampling rate
        first_marker = next(iter(self.markers.values()))
        num_frames = len(first_marker.x)
        
        if data_rate is None:
            data_rate = first_marker.sampling_rate if first_marker.sampling_rate else 120.0
        if camera_rate is None:
            camera_rate = data_rate
        
        num_markers = len(self.markers)
        marker_names = list(self.markers.keys())
        
        with open(output_filepath, 'w') as f:
            # Line 1: Header info
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.abspath(output_filepath)}\n")
            
            # Line 2: Column headers for metadata
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            
            # Line 3: Metadata values
            f.write(f"{data_rate:f}\t{camera_rate:f}\t{num_frames}\t{num_markers}\t{units}\t{data_rate:f}\t0\t{num_frames}\n")
            
            # Line 4: Frame and time headers + marker names (tab separated, 3 tabs per marker)
            header_line = "Frame#\tTime"
            for name in marker_names:
                header_line += f"\t{name}\t\t"
            f.write(header_line + "\n")
            
            # Line 5: XYZ column headers
            xyz_header = "\t"
            for i, name in enumerate(marker_names, 1):
                xyz_header += f"\tX{i}\tY{i}\tZ{i}"
            f.write(xyz_header + "\n")
            
            # Line 6: Empty line (separator)
            f.write("\n")
            
            # Data lines
            for frame_idx in range(num_frames):
                time = frame_idx / data_rate
                line = f"{frame_idx + 1}\t{time}"
                
                for marker_name in marker_names:
                    marker = self.markers[marker_name]
                    x = marker.x[frame_idx]
                    y = marker.y[frame_idx]
                    z = marker.z[frame_idx]
                    
                    # Handle NaN values - write as 'nan' for OpenSim compatibility
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        line += f"\tnan\tnan\tnan"
                    else:
                        line += f"\t{x}\t{y}\t{z}"
                
                f.write(line + "\n")
        
        # print(f"TRC file written to: {output_filepath}")

    def write_mot(self, output_filepath: str, apply_force_threshold: float = 10.0) -> None:
        """
        Write force plate data to a MOT file compatible with OpenSim.
        
        Args:
            output_filepath: Path for the output MOT file
            apply_force_threshold: Threshold below which CoP becomes NaN (default 10 N)
        """
        if not self.forces:
            raise Exception("No force data to write.")
        
        # Get number of force plates and sampling info
        num_plates = len(self.forces)
        first_force = next(iter(self.forces.values()))
        num_samples = len(first_force.force)
        sampling_rate = first_force.sampling_rate if first_force.sampling_rate else 960.0
        
        # Calculate number of columns: time + 9 columns per force plate (vx,vy,vz,px,py,pz,mx,my,mz)
        num_columns = 1 + (num_plates * 9)
        
        with open(output_filepath, 'w') as f:
            # Header
            f.write(f"nColumns={num_columns}\n")
            f.write(f"nRows={num_samples}\n")
            f.write("DataType=double\n")
            f.write("version=3\n")
            f.write("endheader\n")
            
            # Column headers
            header = "time"
            for i in range(1, num_plates + 1):
                header += f"\tground_force_{i}_vx\tground_force_{i}_vy\tground_force_{i}_vz"
                header += f"\tground_force_{i}_px\tground_force_{i}_py\tground_force_{i}_pz"
                header += f"\tground_moment_{i}_mx\tground_moment_{i}_my\tground_moment_{i}_mz"
            f.write(header + "\n")
            
            # Data lines
            force_plates = list(self.forces.values())
            for sample_idx in range(num_samples):
                time = sample_idx / sampling_rate
                line = f"{time}"
                
                for plate in force_plates:
                    # Force vector (vx, vy, vz)
                    fx = plate.force[sample_idx, 0]
                    fy = plate.force[sample_idx, 1]
                    fz = plate.force[sample_idx, 2]
                    
                    # Center of pressure (px, py, pz)
                    px = plate.cop[sample_idx, 0]
                    py = plate.cop[sample_idx, 1]
                    pz = plate.cop[sample_idx, 2]
                    
                    # Moment (mx, my, mz)
                    mx = plate.moment[sample_idx, 0]
                    my = plate.moment[sample_idx, 1]
                    mz = plate.moment[sample_idx, 2]

                    line += f"\t{fx}\t{fy}\t{fz}"
                    line += f"\t{px}\t{py}\t{pz}"
                    line += f"\t{mx}\t{my}\t{mz}"
                
                f.write(line + "\n")
        
        # print(f"MOT file written to: {output_filepath}")

    def process_and_export(self, fullpath: str, axis: str = None, 
                           angle_deg: float = None, convert_to_meters: bool = True) -> None:
        """
        Convenience method to rotate data, convert units, and export to both TRC and MOT files.
        
        Args:
            fullpath: Output path without extension; ``.trc`` and ``.mot`` are
                appended.
            axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``); ``None`` to
                skip rotation.
            angle_deg: Rotation angle in degrees; ``None`` to skip rotation.
            convert_to_meters: Whether to convert position data to metres.
                Defaults to ``True``.
        """
        if axis is not None and angle_deg is not None:
            # print(f"Rotating data {angle_deg} degrees around {axis}-axis...")
            self.rotate_data(axis, angle_deg)
        
        if convert_to_meters:
            self.convert_to_meters()
        
        trc_path = f"{fullpath}.trc"
        mot_path = f"{fullpath}.mot"
        
        # Set units to 'm' if we converted to meters
        units = 'M' if convert_to_meters else None
        
        self.write_trc(trc_path, units=units)
        self.write_mot(mot_path)

    
    def lowpass_filter_force(self, cutoff: float, order: int = 4) -> None:
        """Low-pass filter every force plate in place.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for force in self.forces.values():
            force.lowpass_filter(cutoff, order)

    def lowpass_filter_markers(self, cutoff: float, order: int = 4) -> None:
        """Low-pass filter every marker trajectory in place.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for marker in self.markers.values():
            marker.lowpass_filter(cutoff, order)

    def lowpass_filter_all(self, cutoff: float, order: int = 4) -> None:
        """Low-pass filter both markers and force plates in place.

        Args:
            cutoff: Cutoff frequency in Hz applied to all channels.
            order: Filter order. Defaults to 4.
        """
        self.lowpass_filter_markers(cutoff, order)
        self.lowpass_filter_force(cutoff, order)

    def filter_low_forces(self, threshold: float = 10.0) -> None:
        """Zero out low-magnitude frames on every force plate.

        Args:
            threshold: Force magnitude below which frames are zeroed, in newtons.
                Defaults to 10.0.
        """
        for force in self.forces.values():
            force.filter_low_forces(threshold)

    def slice_markers(self, start_idx: int, end_idx: int) -> None:
        """Slice marker data between start_idx and end_idx. this doesn't affect the original c3d data, only the marker containers."""
        for marker in self.markers.values():
            marker.slice_data(start_idx, end_idx)

    def slice_c3d(self, start_frame: int, end_frame: int) -> None:
        """
        Slice the actual C3D data (points and analogs) between start_frame and end_frame.
        This modifies the underlying c3d_data structure directly so it can save the sliced C3D file.
        DOES NOT SLICE THE CONTAINER DATA
        Args:
            start_frame: Starting frame index (0-based) for point data
            end_frame: Ending frame index (inclusive) for point data
        """
        if self.c3d_data is None:
            raise Exception("No C3D data loaded to slice.")

        header = self.c3d_data['header']
        params = self.c3d_data['parameters']

        # Calculate corresponding analog frame indices based on point frame indices and sampling rates
        analog_ratio = params['ANALOG']['RATE']['value'][0] / params['POINT']['RATE']['value'][0]
        analog_ratio = int(analog_ratio)

        # Slice point data
        point_data = self.c3d_data['data']['points']
        sliced_points = point_data[:, :, start_frame:end_frame + 1]

        #slice analog data
        analog_data = self.c3d_data['data']['analogs']
        analog_start = start_frame * analog_ratio
        analog_end   = (end_frame + 1) * analog_ratio  # exclusive
        sliced_analogs = analog_data[:, :, analog_start:analog_end]

        new_c3d = c3d()

        new_c3d['header']['analogs']['size'] = header['analogs']['size']
        new_c3d['header']['points']['size'] = header['points']['size']
        new_c3d['header']['analogs']['frame_rate'] = header['analogs']['frame_rate']
        new_c3d['header']['points']['frame_rate'] = header['points']['frame_rate']

        for key,val in self.c3d_data['parameters']['POINT'].items():
            new_c3d['parameters']['POINT'][key] = val

        n_frames_new = sliced_points.shape[2]
        new_c3d['parameters']['POINT']['FRAMES']['value'] = [n_frames_new]
        
        for key,val in self.c3d_data['parameters']['ANALOG'].items():
            new_c3d['parameters']['ANALOG'][key] = val

        for key,val in self.c3d_data['parameters']['FORCE_PLATFORM'].items():
            new_c3d['parameters']['FORCE_PLATFORM'][key] = val
            
        #Slice and assign forceplate data
        keys_to_slice = ['force', 'moment', 'center_of_pressure', 'Tz']
        sliced_platforms = []
        for platform in self.c3d_data['data']['platform']:
            new_platform = {}
            for key, value in platform.items():
                if key in keys_to_slice and value is not None and len(value) > 0:
                    new_platform[key] = value[:, analog_start:analog_end]
                else:
                    new_platform[key] = value
            sliced_platforms.append(new_platform)
            
        #Assign the sliced platform
        new_c3d['data']['platform'] = sliced_platforms
        new_c3d['data']['points']  = sliced_points
        new_c3d['data']['analogs'] = sliced_analogs

        self.c3d_data = new_c3d
        self._parse_markers()
        self._parse_analogs()
        self._parse_force_plates()
        print("WARNING: Slicing the c3d data from c3dHandler level does not update trial level containers.")
        return new_c3d
    
    def prepare_osim_files(self, fullpath: str):
        """Export OpenSim-ready TRC and MOT files with the default transform.

        Creates the output directory if needed, then rotates -90° about X,
        converts to metres and writes ``<fullpath>.trc`` and ``<fullpath>.mot``.

        Args:
            fullpath: Output path without extension.
        """
        if not os.path.isdir(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath))
        self.process_and_export(fullpath=fullpath, axis='x', angle_deg=-90, convert_to_meters=True)

    def get_subject_weight(self) -> Optional[float]:
        """Estimate subject weight from the loaded force plates.

        Picks the plate that carries appreciable load (mean magnitude > 50) and
        returns the median vertical force. The lab coordinate system is Z-up, so
        the vertical force is taken from ``Fz``.

        Returns:
            The estimated weight in newtons, or ``None`` if no loaded plate is
            found.
        """
        for force_plate in self.forces.values():
            magnitudes = force_plate.get_force_magnitude()
            if np.mean(magnitudes) > 50:  # Arbitrary threshold to identify used forceplate
                return np.abs(np.median(force_plate.Fz)) #Lab cs is z up, so we can use z force to estimate weight
    
    def prepare_virtual_c3d_files(self, fullpath: str):
        """Prepare virtual c3d files for the reference trial. This is needed for scaling in opensim."""
        #TODO maybe refactor???
        if not os.path.isdir(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath))
        
        r_asis, l_asis, r_psis, l_psis = (
        self.markers['R_ASIS'].get_trajectory(),
        self.markers['L_ASIS'].get_trajectory(),
        self.markers['R_PSIS'].get_trajectory(),
        self.markers['L_PSIS'].get_trajectory(),
        )

        r_knee, r_kneemed, l_knee, l_kneemed = (
        self.markers['R_Knee'].get_trajectory(),
        self.markers['R_Kneemedial'].get_trajectory(),
        self.markers['L_Knee'].get_trajectory(),
        self.markers['L_Kneemedial'].get_trajectory(),
        )

        R_MalMed, L_MalMed = (
        self.markers['R_MalMed'].get_trajectory(),
        self.markers['L_MalMed'].get_trajectory(),
    )
        r_hjc, l_hjc, ASIS, PSIS, PELVIS_CENTER, HIP_CENTER = calculate_hip_markers(r_asis, l_asis, r_psis, l_psis, R_MalMed, L_MalMed)

        r_kjc = calculate_joint_center(r_kneemed, r_knee)
        l_kjc = calculate_joint_center(l_kneemed, l_knee)

        R_Ankle, L_Ankle = (
        self.markers['R_Ankle'].get_trajectory(),
        self.markers['L_Ankle'].get_trajectory(),
        )
        r_ajc = calculate_joint_center(R_Ankle, R_MalMed)
        l_ajc = calculate_joint_center(L_Ankle, L_MalMed)

        l_trochanter = self.markers['L_Trochanter'].get_trajectory()
        r_trochanter = self.markers['R_Trochanter'].get_trajectory()
        trcc = calculate_joint_center(l_trochanter, r_trochanter)
        #%%
        #I don't trust the hip joint center estimates.
        # r_hjc_marker = MarkerData(name='RHJC', x=r_hjc[:,0], y=r_hjc[:,1], z=r_hjc[:,2])
        # l_hjc_marker = MarkerData(name='LHJC', x=l_hjc[:,0], y=l_hjc[:,1], z=l_hjc[:,2])
        r_kjc_marker = MarkerData(name='RKJC', x=r_kjc[:,0], y=r_kjc[:,1], z=r_kjc[:,2])
        l_kjc_marker = MarkerData(name='LKJC', x=l_kjc[:,0], y=l_kjc[:,1], z=l_kjc[:,2])
        r_ajc_marker = MarkerData(name='RAJC', x=r_ajc[:,0], y=r_ajc[:,1], z=r_ajc[:,2])
        l_ajc_marker = MarkerData(name='LAJC', x=l_ajc[:,0], y=l_ajc[:,1], z=l_ajc[:,2])
        asis_marker = MarkerData(name='ASIS', x=ASIS[:,0], y=ASIS[:,1], z=ASIS[:,2])
        psis_marker = MarkerData(name='PSIS', x=PSIS[:,0], y=PSIS[:,1], z=PSIS[:,2])
        pelvis_center_marker = MarkerData(name='PC', x=PELVIS_CENTER[:,0], y=PELVIS_CENTER[:,1], z=PELVIS_CENTER[:,2])
        trcc_marker = MarkerData(name='TRCC', x=trcc[:,0], y=trcc[:,1], z=trcc[:,2]) #This is for TRC export, which needs a root marker
        # hip_center_marker = MarkerData(name='HPC', x=HIP_CENTER[:,0], y=HIP_CENTER[:,1], z=HIP_CENTER[:,2]) #Calculated from hjc, which I don't trust atm

        self.add_marker(r_ajc_marker)
        self.add_marker(l_ajc_marker)
        self.add_marker(trcc_marker)
        #%% These are for foot scaling
        markers_to_project = ['R_HeelTop','R_Meta1', 'R_Meta5', 'R_MalMed', 'R_Ankle', 'L_HeelTop','L_Meta1', 'L_Meta5', 'L_MalMed', 'L_Ankle', 'LAJC', 'RAJC']
        for marker_name in markers_to_project:
            temp_marker = deepcopy(self.markers[marker_name])
            temp_marker.name = marker_name + 'P'
            temp_marker.z = np.zeros_like(temp_marker.z)
            self.add_marker(temp_marker)

        l_metamid = (self.markers['L_Meta1P'].get_trajectory() + self.markers['L_Meta5P'].get_trajectory()) /2.0
        r_metamid = (self.markers['R_Meta1P'].get_trajectory() + self.markers['R_Meta5P'].get_trajectory()) /2.0
        l_metamid_marker = MarkerData(name='L_MetaMidP', x=l_metamid[:,0], y=l_metamid[:,1], z=l_metamid[:,2])
        r_metamid_marker = MarkerData(name='R_MetaMidP', x=r_metamid[:,0], y=r_metamid[:,1], z=r_metamid[:,2])

        acro_mid = (self.markers['R_Acromion'].get_trajectory() + self.markers['L_Acromion'].get_trajectory()) /2.0
        acro_mid_marker = MarkerData(name='ACROMID', x=acro_mid[:,0], y=acro_mid[:,1], z=acro_mid[:,2])
        #%%
        self.add_marker(r_metamid_marker)
        self.add_marker(l_metamid_marker)
        # self.add_marker(r_hjc_marker)
        # self.add_marker(l_hjc_marker)
        self.add_marker(r_kjc_marker)
        self.add_marker(l_kjc_marker)
        self.add_marker(asis_marker)
        self.add_marker(psis_marker)
        self.add_marker(pelvis_center_marker)
        # self.add_marker(hip_center_marker)
        self.add_marker(acro_mid_marker)
        self.write_c3d(f"{fullpath}_virtual.c3d")
        self.process_and_export(fullpath=fullpath, axis='x', angle_deg=-90, convert_to_meters=True)


if __name__ == '__main__':
    # Example usage
    path = 'D:\\HTO_data\\old_HTOc3ds\\03\\Pre\\Gang\\links\\'
    filename = "03_PRE_GANG12_01.c3d"
    filepath = os.path.join(path, filename)
    
    # Load C3D file using handler (keeps instance for manipulation)
    handler = C3DHandler(filepath)
    
    # Rotate data 90 degrees around the x-axis (example)
    # handler.rotate_data('x', 90)
    
    # Export to TRC and MOT files
    # handler.write_trc('output.trc', units='M')
    # handler.write_mot('output.mot')
    
    # Or use the convenience method:
    # handler.process_and_export('output', axis='x', angle_deg=90)
    
    # Alternatively, load as TrialData (read-only, more memory efficient)
    trial_1 = C3DHandler.load_trial(filepath)
    print(trial_1)
    

