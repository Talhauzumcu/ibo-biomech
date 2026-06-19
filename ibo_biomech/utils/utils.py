"""Shared helper functions.

Low-level utilities used across the library: building rotation matrices and
writing OpenSim TRC and MOT files from container objects.
"""
import numpy as np
import os

def get_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix about a coordinate axis.

    Args:
        axis: Axis to rotate about (``'x'``, ``'y'`` or ``'z'``).
        angle_deg: Rotation angle in degrees.

    Returns:
        The 3x3 rotation matrix.

    Raises:
        ValueError: If ``axis`` is not one of ``'x'``, ``'y'`` or ``'z'``.
    """
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
    

def write_trc(output_filepath: str, header_dict: dict, marker_data: dict) -> None:
        """Write marker data to an OpenSim-compatible TRC file.

        Args:
            output_filepath: Path for the output TRC file.
            header_dict: TRC header values. Expected keys: ``data_rate``,
                ``camera_rate``, ``num_frames``, ``num_markers``, ``units``,
                ``orig_data_rate``, ``orig_data_start_frame``,
                ``orig_num_frames`` and ``marker_labels``.
            marker_data: Mapping of marker label to
                :class:`~ibo_biomech.containers.MarkerData`.
        """
        data_rate = header_dict.get('data_rate')
        camera_rate = header_dict.get('camera_rate')
        num_frames = header_dict.get('num_frames')
        num_markers = header_dict.get('num_markers')
        units = header_dict.get('units')
        orig_data_rate = header_dict.get('orig_data_rate')
        orig_data_start_frame = header_dict.get('orig_data_start_frame')
        orig_num_frames = header_dict.get('orig_num_frames')
        marker_labels = header_dict.get('marker_labels', [])
        
        with open(output_filepath, 'w') as f:
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.abspath(output_filepath)}\n")            
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{data_rate:f}\t{camera_rate:f}\t{num_frames}\t{num_markers}\t{units}\t{orig_data_rate:f}\t{orig_data_start_frame}\t{orig_num_frames}\n")            
            header_line = "Frame#\tTime"
            for name in marker_labels:
                header_line += f"\t{name}\t\t"
            f.write(header_line + "\n")
            
            xyz_header = "\t"
            for i, name in enumerate(marker_labels, 1):
                xyz_header += f"\tX{i}\tY{i}\tZ{i}"
            f.write(xyz_header + "\n")
            f.write("\n")
            
            for frame_idx in range(num_frames):
                time = frame_idx / data_rate
                line = f"{frame_idx + 1}\t{time}"
                
                for marker_name in marker_labels:
                    marker = marker_data[marker_name]
                    x = marker.x[frame_idx]
                    y = marker.y[frame_idx]
                    z = marker.z[frame_idx]
                    line += f"\t{x}\t{y}\t{z}"
                
                f.write(line + "\n")
        
        print(f"TRC file written to: {output_filepath}")


def write_mot(output_filepath: str, forces: dict) -> None:
        """Write force plate data to an OpenSim-compatible MOT file.

        Writes nine columns per plate: force (vx, vy, vz), centre of pressure
        (px, py, pz) and moment (mx, my, mz).

        Args:
            output_filepath: Path for the output MOT file.
            forces: Mapping of plate name to
                :class:`~ibo_biomech.containers.ForceData`.

        Raises:
            Exception: If ``forces`` is empty.
        """
        if not forces:
            raise Exception("No force data to write.")
        
        # Get number of force plates and sampling info
        num_plates = len(forces)
        first_force = next(iter(forces.values()))
        num_samples = first_force.num_samples
        sampling_rate = first_force.sampling_rate if first_force.sampling_rate else 'Unknown'
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
            force_plates = list(forces.values())
            for sample_idx in range(num_samples):
                time = sample_idx / sampling_rate
                line = f"{time}"
                
                for plate in force_plates:
                    # Force vector (vx, vy, vz)
                    fx = plate.force[0, sample_idx]
                    fy = plate.force[1, sample_idx]
                    fz = plate.force[2, sample_idx]
                    
                    # Center of pressure (px, py, pz)
                    px = plate.cop[0, sample_idx]
                    py = plate.cop[1, sample_idx]
                    pz = plate.cop[2, sample_idx]

                    # Moment (mx, my, mz)
                    mx = 0
                    my = plate.Tz[sample_idx]
                    mz = 0

                    line += f"\t{fx}\t{fy}\t{fz}"
                    line += f"\t{px}\t{py}\t{pz}"
                    line += f"\t{mx}\t{my}\t{mz}"
                
                f.write(line + "\n")
        
        # print(f"MOT file written to: {output_filepath}")