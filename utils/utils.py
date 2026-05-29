import numpy as np

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
    