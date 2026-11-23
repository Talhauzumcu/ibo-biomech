"""C3D force platforms of type 5: eight channels and a 6x8 calibration matrix.

ezc3d rejects type 5, so these plates are computed by
``ibo_biomech.handlers._force_platforms``. All files are synthetic.
"""
import numpy as np
import pytest
from ezc3d import c3d

from ibo_biomech import C3DHandler

N_FRAMES, N = 20, 200          # 100 Hz points, 1000 Hz analogs
A, B = 500., 800.               # Kistler sensor offsets along plate x and y (mm)
# Layout of a real QTM export: plate x along lab +Y, plate y along lab +X, plate z down.
CORNERS = np.array([[2310., 2310., 0., 0.], [1050., 0., 0., 1050.], [0., 0., 0., 0.]])
CENTRE = np.array([1155., 525., 0.])
# Plate channel order used by QTM, and the order the analogs are stored in the file.
PLATE_ORDER = ['Z1', 'Z2', 'Z3', 'Z4', 'Y14', 'Y23', 'X12', 'X34']
FILE_ORDER = ['X34', 'X12', 'Y23', 'Y14', 'Z4', 'Z3', 'Z2', 'Z1']
SENSITIVITY = np.array([-300., -300., -300., -300., 150., 150., 130., 130.])  # N/V
# Kistler equations as rows Fx, Fy, Fz, Mx, My, Mz over PLATE_ORDER channels in newtons.
KISTLER = np.array([[0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [B, B, -B, -B, 0, 0, 0, 0],
                    [-A, A, A, -A, 0, 0, 0, 0],
                    [0, 0, 0, 0, A, -A, -B, B]], dtype=float)


def _source(analogs, labels):
    source = c3d()
    source['parameters']['POINT']['RATE']['value'] = [100.]
    source['parameters']['POINT']['LABELS']['value'] = ['Marker']
    source['parameters']['POINT']['UNITS']['value'] = ['mm']
    source['data']['points'] = np.ones((4, 1, N_FRAMES))
    source['parameters']['ANALOG']['RATE']['value'] = [1000.]
    source['parameters']['ANALOG']['LABELS']['value'] = labels
    source['parameters']['ANALOG']['UNITS']['value'] = ['V'] * len(labels)
    source['data']['analogs'] = analogs[np.newaxis]
    return source


def _set_plates(source, types, channel, corners, origin, cal_matrix, matrix_store='BYCOLUMN'):
    values = dict(USED=[len(types)], TYPE=types, ZERO=[0, 0], CHANNEL=channel,
                  CORNERS=corners, ORIGIN=origin)
    for name, value in values.items():
        source['parameters']['FORCE_PLATFORM'][name]['value'] = value
    if cal_matrix is not None:
        source.add_parameter('FORCE_PLATFORM', 'CAL_MATRIX', cal_matrix)
    if matrix_store is not None:
        source.add_parameter('FORCE_PLATFORM', 'MATRIX_STORE', [matrix_store])


def _known_load():
    """A moving load in the plate frame and the eight channel voltages it produces."""
    t = np.arange(N) / 1000.
    fx, fy = 30. * np.sin(9 * t), -20. * np.cos(7 * t)
    fz = 700. + 150. * np.sin(5 * t)
    px, py = 200. * np.sin(3 * t), -300. + 400. * t  # CoP on the plate surface
    free = 4000. * np.sin(4 * t)                   # free moment about plate z (Nmm)
    p, q = px / A, py / B
    z = fz / 4 * np.array([(1 + p) * (1 + q), (1 - p) * (1 + q), (1 - p) * (1 - q), (1 + p) * (1 - q)])
    mz = free + px * fy - py * fx
    forces = np.vstack([z, fy / 2, fy / 2, (fx - mz / B) / 2, (fx + mz / B) / 2])  # PLATE_ORDER
    load = dict(fx=fx, fy=fy, fz=fz, px=px, py=py, free=free, mz=mz)
    return load, forces / SENSITIVITY[:, np.newaxis]


def _write_known_load(path, layout='6x8', matrix_store='BYCOLUMN', with_cal_matrix=True):
    load, volts = _known_load()
    analogs = volts[[PLATE_ORDER.index(label) for label in FILE_ORDER]]
    channel = np.array([[FILE_ORDER.index(label) + 1] for label in PLATE_ORDER])
    matrix = KISTLER * SENSITIVITY  # converts PLATE_ORDER volts to Fx..Mz
    if layout == '8x6':
        matrix = matrix.T
    source = _source(analogs, FILE_ORDER)
    _set_plates(source, [5], channel, CORNERS[:, :, np.newaxis], np.zeros((3, 1)),
                matrix[:, :, np.newaxis] if with_cal_matrix else None, matrix_store)
    source.write(str(path))
    return load


@pytest.mark.parametrize('layout', ['6x8', '8x6'])
def test_type5_matches_known_load(tmp_path, layout):
    path = tmp_path / 'type5.c3d'
    load = _write_known_load(path, layout)
    plate = C3DHandler(str(path)).load_data().forces['forceplate_0']
    fx, fy, fz, px, py = (load[key] for key in ('fx', 'fy', 'fz', 'px', 'py'))
    # Plate x -> lab +Y, plate y -> lab +X, plate z -> lab -Z.
    # C3D stores analogs and parameters as float32, hence the tolerances.
    np.testing.assert_allclose(plate.force, np.vstack([fy, fx, -fz]), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(plate.cop, np.vstack([py + CENTRE[0], px + CENTRE[1], np.zeros(N)]), atol=1e-2)
    np.testing.assert_allclose(plate.moment, np.vstack([-px * fz, py * fz, -load['mz']]), rtol=1e-5, atol=1.)
    np.testing.assert_allclose(plate.Tz, np.vstack([np.zeros(N), np.zeros(N), -load['free']]), rtol=1e-5, atol=1.)
    assert plate.metadata['calibration_matrix'].shape == (6, 8)
    assert (plate.unit_force, plate.unit_moment, plate.unit_cop) == ('N', 'Nmm', 'mm')


def test_type5_uses_ezc3d_type4_conventions(tmp_path):
    """A type-5 plate equals ezc3d's own type-4 result for the same calibrated signals."""
    rng = np.random.default_rng(0)
    analogs = rng.normal(size=(8, N))
    calibration = rng.normal(size=(6, 6)) * np.array([[100.], [100.], [100.], [1e4], [1e4], [1e4]])
    cal_matrix = np.zeros((6, 8, 2))
    cal_matrix[:, :6, 0] = calibration
    # Plate 1 reads the channels in reverse order; its columns are reordered to match.
    cal_matrix[:, 2:, 1] = calibration[:, ::-1]
    channel = np.column_stack([np.arange(1, 9), np.arange(8, 0, -1)])
    angle = np.deg2rad(20.)
    tilt = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]])
    corners = tilt @ np.array([[200., 0., 0., 200.], [400., 400., 0., 0.], [0., 0., 0., 0.]])
    origin = np.array([[10., 10.], [-5., -5.], [40., 40.]])  # positive z: both flip it
    source = _source(analogs, [f'ch{i}' for i in range(8)])
    _set_plates(source, [4, 5], channel, np.repeat(corners[:, :, np.newaxis], 2, axis=2), origin,
                cal_matrix, matrix_store=None)
    path = tmp_path / 'mixed.c3d'
    source.write(str(path))
    handler = C3DHandler(str(path))
    forces = handler.load_data().forces
    for field in ('force', 'moment', 'cop', 'Tz'):
        np.testing.assert_allclose(getattr(forces['forceplate_1'], field),
                                   getattr(forces['forceplate_0'], field), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(handler.c3d_data['data']['platform'][1]['origin'], [-10., 5., -40.])


def test_supported_types_still_use_ezc3d(synthetic_c3d):
    handler = C3DHandler(str(synthetic_c3d))
    handler.load_data()
    expected = c3d(str(synthetic_c3d), extract_forceplat_data=True)['data']['platform']
    actual = handler.c3d_data['data']['platform']
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert set(got.keys()) == set(want.keys())
        for key in want.keys():
            if isinstance(want[key], str):
                assert got[key] == want[key]
            else:
                np.testing.assert_array_equal(got[key], want[key])


def test_type5_processed_c3d_round_trip(tmp_path):
    path = tmp_path / 'type5.c3d'
    _write_known_load(path)
    handler = C3DHandler(str(path))
    source = handler.load_data().forces['forceplate_0']
    output = handler.write_c3d(str(tmp_path / 'processed.c3d'))
    reloaded = C3DHandler(output).load_data().forces['forceplate_0']
    for field in ('force', 'moment', 'cop', 'Tz'):
        np.testing.assert_allclose(getattr(reloaded, field), getattr(source, field), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('options, message', [
    (dict(with_cal_matrix=False), 'CAL_MATRIX'),  # ezc3d writes an empty default
    (dict(matrix_store='BYROW'), 'MATRIX_STORE'),
])
def test_type5_rejects_missing_or_unknown_calibration(tmp_path, options, message):
    path = tmp_path / 'bad.c3d'
    _write_known_load(path, **options)
    with pytest.raises(Exception, match=message):
        C3DHandler(str(path)).load_data()
