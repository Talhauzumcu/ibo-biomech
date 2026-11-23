"""Force-platform extraction for C3D files, including type 5.

ezc3d computes force-platform types 1-4 but raises for type 5, and a single
type-5 plate aborts reading the whole file. Plates are therefore extracted one
by one: supported types use ezc3d's own computation, and type 5 is computed
here with the same conventions ezc3d applies to type 4.

Type 5 plates (e.g. Kistler plates exported by Qualisys QTM) record eight
analog channels. A 6x8 FORCE_PLATFORM:CAL_MATRIX converts them to
Fx, Fy, Fz, Mx, My, Mz in the plate frame.
"""
import numpy as np
from ezc3d import c3d
from ezc3d import ezc3d as _ezc3d


def extract_platforms(data):
    """Return ezc3d-style platform mappings for every used force platform.

    Args:
        data: An ezc3d ``c3d`` object.

    Returns:
        One mapping per platform with the keys ezc3d uses: ``unit_force``,
        ``unit_moment``, ``unit_position``, ``cal_matrix``, ``corners``,
        ``origin``, ``force``, ``moment``, ``center_of_pressure`` and ``Tz``.
    """
    group = data['parameters'].get('FORCE_PLATFORM')
    if group is None or 'USED' not in group:
        return []
    used = np.ravel(group['USED']['value'])
    types = np.ravel(group['TYPE']['value']) if 'TYPE' in group else np.array([])
    platforms = []
    for index in range(int(used[0]) if used.size else 0):
        if index < types.size and int(types[index]) == 5:
            platforms.append(_type5_platform(data, group, index))
        else:
            platforms.append(c3d.PlatForm(_ezc3d.ForcePlatform(index, data.c3d_swig)))
    return platforms


def _flat(group, name):
    """Parameter values in C3D storage order (first dimension fastest)."""
    return np.ravel(np.asarray(group[name]['value'], dtype=float), order='F')


def _calibration_matrix(group, index):
    """6x8 matrix mapping the CHANNEL-ordered analogs to Fx..Mz."""
    if 'CAL_MATRIX' not in group:
        raise ValueError('FORCE_PLATFORM:CAL_MATRIX is required for type-5 force platforms.')
    store = group.get('MATRIX_STORE', {}).get('value', [])
    if len(store) and store[min(index, len(store) - 1)].strip().upper() != 'BYCOLUMN':
        raise ValueError(f'Unsupported FORCE_PLATFORM:MATRIX_STORE {store!r}; '
                         'type-5 plates are only supported with BYCOLUMN storage.')
    values = np.asarray(group['CAL_MATRIX']['value'], dtype=float)
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.ndim != 3 or values.shape[2] <= index:
        raise ValueError(f'FORCE_PLATFORM:CAL_MATRIX has no matrix for force platform {index + 1}.')
    matrix = values[:, :, index]
    # The six outputs run along the axis of length 6, the eight channels along the other.
    if matrix.shape == (8, 6):
        return matrix.T
    if matrix.shape[0] >= 6 and matrix.shape[1] >= 8:
        return matrix[:6, :8]
    raise ValueError(f'FORCE_PLATFORM:CAL_MATRIX must be 6x8 for type-5 plates, got {matrix.shape}.')


def _type5_platform(data, group, index):
    """Compute one type-5 platform, mirroring ezc3d's type-4 conventions."""
    channels = np.asarray(group['CHANNEL']['value'], dtype=int)
    if channels.ndim == 1:
        channels = channels[:, np.newaxis]
    if channels.ndim != 2 or channels.shape[0] < 8 or channels.shape[1] <= index:
        raise ValueError('FORCE_PLATFORM:CHANNEL must list 8 analog channels for each type-5 force platform.')
    channels = channels[:8, index] - 1  # 1-based in C3D
    analogs = data['data']['analogs']
    if np.any(channels < 0) or np.any(channels >= analogs.shape[1]):
        raise ValueError('FORCE_PLATFORM:CHANNEL references an analog channel that does not exist.')

    corners = _flat(group, 'CORNERS')
    origin = _flat(group, 'ORIGIN')
    if corners.size < 12 * (index + 1) or origin.size < 3 * (index + 1):
        raise ValueError('FORCE_PLATFORM:CORNERS/ORIGIN are not filled for every force platform.')
    corners = corners[12 * index:12 * (index + 1)].reshape(3, 4, order='F')
    origin = origin[3 * index:3 * (index + 1)].copy()
    if origin[2] > 0:
        # Same correction ezc3d/BTK apply: the origin must point from the surface to the sensor.
        origin = -origin

    # Plate axes from the corners, built exactly as ezc3d does for the other types.
    axis_x = corners[:, 0] - corners[:, 1]
    axis_y = corners[:, 0] - corners[:, 3]
    axis_z = np.cross(axis_x, axis_y)
    axis_y = np.cross(axis_z, axis_x)
    rotation = np.column_stack([axis / np.linalg.norm(axis) for axis in (axis_x, axis_y, axis_z)])

    matrix = _calibration_matrix(group, index)
    calibrated = matrix @ analogs[0, channels, :]
    force = calibrated[:3]
    moment = calibrated[3:] + np.cross(force, origin[:, np.newaxis], axis=0)  # about the surface centre
    with np.errstate(divide='ignore', invalid='ignore'):  # unloaded samples give inf/nan, as in ezc3d
        cop = np.vstack((-moment[1] / force[2], moment[0] / force[2], np.zeros_like(force[2])))
    tz = moment + np.cross(force, cop, axis=0)  # moment about the CoP

    point_units = data['parameters']['POINT'].get('UNITS', {}).get('value', [])
    force_units = group.get('UNITS', {}).get('value', [])
    unit_position = point_units[0] if len(point_units) else 'm'
    unit_force = force_units[0] if len(force_units) else 'N'
    return {
        'unit_force': unit_force,
        'unit_moment': unit_force + unit_position,
        'unit_position': unit_position,
        'cal_matrix': matrix,
        'corners': corners,
        'origin': origin,
        'force': rotation @ force,
        'moment': rotation @ moment,
        'center_of_pressure': rotation @ cop + corners.mean(axis=1)[:, np.newaxis],
        'Tz': rotation @ tz,
    }
