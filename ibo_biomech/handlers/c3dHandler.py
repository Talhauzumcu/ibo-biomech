from ezc3d import c3d
import os
from typing import Dict, List, Optional, Tuple, Union
from ibo_biomech.containers import AnalogData, ForceData, MarkerData, TrialData
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
        self.trial_name = os.path.splitext(os.path.basename(filepath))[0] if filepath else None
        self.c3d_data = None
        self.markers: Dict[str, MarkerData] = {}
        self.analogs: Dict[str, AnalogData] = {}
        self.forces: Dict[str, ForceData] = {}

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
            self.c3d_data = c3d(self.filepath, extract_forceplat_data=True)
            self._parse_markers()
            self._parse_analogs()
            self._parse_force_plates()     
            self.is_loaded = True

        except Exception as e:
            raise Exception(f"Error loading C3D file: {str(e)}")
    
        return TrialData(
        name=deepcopy(self.trial_name),
        markers=deepcopy(self.markers),
        analogs=deepcopy(self.analogs),
        forces=deepcopy(self.forces)
    )

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
