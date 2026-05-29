import numpy as np
import os 

def get_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for rotation around a specified axis.
    
    Args:
        axis: Axis to rotate around ('x', 'y', or 'z')
        angle_deg: Rotation angle in degrees
        
    Returns:
        3x3 rotation matrix
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
        """
        Write marker data to a TRC file compatible with OpenSim.
        
        Args:
            output_filepath: Path for the output TRC file
            header_dict: Dictionary containing header information for the TRC file
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
