"""Shared helper functions.

Low-level utilities used across the library: building rotation matrices and
writing OpenSim TRC and MOT files from container objects.
"""

from pathlib import Path
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
    

def write_trc(output_filepath: str, header_dict: dict, marker_data: dict, time: np.ndarray = None) -> None:
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
        time = time if time is not None else np.arange(num_frames) / data_rate
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
                line = f"{frame_idx + 1}\t{time[frame_idx]}"
                
                for marker_name in marker_labels:
                    marker = marker_data[marker_name]
                    x = marker.x[frame_idx]
                    y = marker.y[frame_idx]
                    z = marker.z[frame_idx]
                    line += f"\t{x}\t{y}\t{z}"
                
                f.write(line + "\n")
        
        print(f"TRC file written to: {output_filepath}")


def write_mot(output_filepath: str, forces: dict, time: np.ndarray = None) -> None:
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
        time = time if time is not None else np.arange(num_samples) / sampling_rate
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
                line = f"{time[sample_idx]}"
                
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





def build_extloads(r_idx, output_file, mot_file=None, h5_file=None):
    """Inverse dynamics requires to read the mot file. If not provided it will have to be created here."""
    if mot_file is None:
        from ibo_biomech import FileConverter
        mot_file = h5_file.replace('.h5', '.mot')
        mot_file=FileConverter.h5_to_mot(h5_path=h5_file, mot_path=mot_file)
    mot_file = Path(mot_file).resolve() if isinstance(mot_file, str) else Path(mot_file).resolve()
    _EXTFORCE = """\t\t\t<ExternalForce name="{name}">
    \t\t\t\t<applied_to_body>{body}</applied_to_body>
    \t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
    \t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
    \t\t\t\t<force_identifier>ground_force_{idx}_v</force_identifier>
    \t\t\t\t<point_identifier>ground_force_{idx}_p</point_identifier>
    \t\t\t\t<torque_identifier>ground_moment_{idx}_m</torque_identifier>
    \t\t\t\t<data_source_name>Unassigned</data_source_name>
    \t\t\t</ExternalForce>"""

    """Write an ExternalLoads xml pointing the right foot at its correct plate."""
    l_idx = 2 if r_idx == 1 else 1
    grf = mot_file
    xml = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40500">\n'
        '\t<ExternalLoads name="externalloads">\n'
        '\t\t<objects>\n'
        + _EXTFORCE.format(name="rightfoot", body="calcn_r", idx=r_idx) + "\n"
        + _EXTFORCE.format(name="leftfoot",  body="calcn_l", idx=l_idx) + "\n"
        + '\t\t</objects>\n'
        '\t\t<groups />\n'
        f'\t\t<datafile>{grf}</datafile>\n'
        '\t</ExternalLoads>\n'
        '</OpenSimDocument>\n'
    )
    Path(output_file).write_text(xml, encoding="utf-8")
    return output_file.resolve() if isinstance(output_file, Path) else Path(output_file).resolve()

def read_storage(path: str) -> dict:
    """Read an OpenSim .sto or .mot file and return its contents as a dictionary"""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    end = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "endheader")
    metadata = {}
    for ln in lines[0:end]:
        if not ln.strip():
            continue
        if "=" in ln:
            key, value = ln.split("=", 1)
            metadata[key.strip()] = value.strip()
    header = [h.strip() for h in lines[end + 1].split("\t")]
    cols = {h: [] for h in header}
    for ln in lines[end + 2:]:
        if not ln.strip():
            continue
        for h, v in zip(header, ln.split("\t")):
            cols[h].append(float(v))

    return_dict = {h: np.asarray(v) for h, v in cols.items()}
    return_dict["metadata"] = metadata
    return return_dict

read_sto = read_storage  # alias for convenience
read_mot = read_storage  # alias for convenience

def read_trc(path: str):
    """Return (time, marker_fn, marker_names). marker_fn(name) -> (n,3) X,Y,Z array."""
    lines = Path(path).read_text(errors="replace").splitlines()
    names = lines[3].split("\t")                      # Frame#, Time, then 3 cols/marker
    start_col = {nm.strip(): i for i, nm in enumerate(names)
                 if nm.strip() and nm.strip() not in ("Frame#", "Time")}
    rows = []
    for ln in lines[5:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        try:
            float(parts[0])
        except ValueError:
            continue
        rows.append([float(x) if x.strip() not in ("", "nan", "NaN") else np.nan
                     for x in parts])
    D = np.array(rows)
    t = D[:, 1]

    def marker(nm):
        c = start_col[nm]
        return D[:, c:c + 3]
    return t, marker, set(start_col)

def time_normalize(signal, num_points=100):
    """Normalize a 1D signal to a fixed number of points using linear interpolation. 
    Assumes equal spacing in the original signal. The normalized signal will have a new time axis from 0 to 100%.
    
    Args:
        signal: 1D array-like signal to normalize.
        num_points: Number of points to normalize to. Defaults to 100.
    Returns:
        new_time: 1D array of length num_points representing the normalized time axis (0 to 100%).
        normalized_signal: 1D array of length num_points.
    """
    original_length = len(signal)
    normalized_signal = np.interp(np.linspace(0, original_length - 1, num_points), np.arange(original_length), signal)
    new_time = np.linspace(0,100, num_points)
    return (new_time,normalized_signal)