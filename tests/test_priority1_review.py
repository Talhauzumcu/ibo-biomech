"""Semantic checks for docs/remaining-issues.md (2026-09-24).

These regressions now assert the supported schema and processing contracts.
"""
from copy import deepcopy

import h5py
import numpy as np
import pytest

from ibo_biomech import C3DHandler, FileConverter, H5Handler
from ibo_biomech.containers import ForceData
from ibo_biomech.utils.utils import write_mot


def test_c3d_h5_save_reload_preserves_force_arrays(synthetic_c3d, converted_h5, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    original = handler.load_data().forces['forceplate_0']
    # Check the parser against ezc3d, including the nonvertical free moment.
    np.testing.assert_allclose(original.Tz, handler.c3d_data['data']['platform'][0]['Tz'])
    assert np.any(original.Tz[1])
    h5 = H5Handler(str(converted_h5))
    trial = h5.load_data()
    output = tmp_path / 'saved.h5'
    h5.save_data(trial, str(output))
    for actual in [trial.forces['forceplate_0'],
                   H5Handler(str(output)).load_data().forces['forceplate_0']]:
        for name in ['force', 'moment', 'cop', 'Tz', 'corners', 'origin',
                     'position', 'rotation', 'time']:
            np.testing.assert_allclose(getattr(actual, name), getattr(original, name))
        assert actual.coordinateSystem == original.coordinateSystem == 1
        assert (actual.unit_force, actual.unit_moment, actual.unit_cop) == ('N', 'Nmm', 'mm')
        assert actual.sampling_rate == original.sampling_rate
    FileConverter.h5_to_mot(str(output), str(tmp_path / 'converted.mot'))
    rows = np.loadtxt(tmp_path / 'converted.mot', skiprows=6)
    assert rows.shape == (200, 10)
    # Default export rotates -90 degrees about X and converts Nmm to Nm.
    expected_moment = original.Tz[[0, 2, 1]].T * [0.001, 0.001, -0.001]
    np.testing.assert_allclose(rows[:, 7:10], expected_moment, atol=1e-14)


def test_noncanonical_geometry_is_rejected(converted_h5):
    with h5py.File(converted_h5, 'r+') as file:
        file['ForcePlates/0'].move('Corners', 'corners')
    with pytest.raises(ValueError, match='missing required datasets'):
        H5Handler(str(converted_h5)).load_data()


@pytest.mark.parametrize('factor', [3, 4])
def test_downsample_preserves_clock_and_proper_rotations(force_data, factor, tmp_path):
    force_data.time += 5.
    expected_time = force_data.time[::factor].copy()
    # Moving orientations detect accidentally retaining only one static matrix.
    angles = np.linspace(0., 1., force_data.num_samples)
    for i, angle in enumerate(angles):
        force_data.rotation[:, :, i] = [[np.cos(angle), -np.sin(angle), 0.],
                                        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    expected_rotation = force_data.rotation[:, :, ::factor].copy()
    force_data.downsample(factor)
    np.testing.assert_allclose(force_data.time, expected_time)
    assert np.all(np.diff(force_data.time) > 0)
    np.testing.assert_allclose(np.diff(force_data.time), factor / 100.)
    np.testing.assert_allclose(force_data.rotation, expected_rotation)
    for rotation in np.moveaxis(force_data.rotation, -1, 0):
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.)
    for name in ['force', 'moment', 'cop', 'Tz', 'corners', 'position', 'rotation']:
        assert getattr(force_data, name).shape[-1] == len(expected_time)
    path = tmp_path / 'downsampled.mot'
    write_mot(str(path), {force_data.name: force_data}, time=force_data.time)
    np.testing.assert_allclose(np.loadtxt(path, skiprows=6)[:, 0], expected_time)


@pytest.mark.parametrize('factor', [0, -1, 1.5])
def test_invalid_downsample_factor_does_not_mutate(force_data, factor):
    before = deepcopy(force_data)
    with pytest.raises(ValueError, match='positive integer'):
        force_data.downsample(factor)
    np.testing.assert_array_equal(force_data.force, before.force)
    np.testing.assert_array_equal(force_data.time, before.time)
    assert force_data.sampling_rate == before.sampling_rate


def test_downsample_preserves_static_geometry(force_data):
    expected = force_data.corners[:, :, ::3].copy()
    force_data.downsample(3)
    np.testing.assert_allclose(force_data.corners, expected)


def test_downsample_rejects_nonmonotonic_clock(force_data):
    force_data.time[20] = force_data.time[19]
    with pytest.raises(ValueError):
        force_data.downsample(3)


def test_marker_unit_conversion_survives_two_saves(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.convert_marker_units('m')
    for index in range(2):
        path = tmp_path / f'markers_{index}.h5'
        handler.save_data(trial, str(path))
        handler = H5Handler(str(path))
        trial = handler.load_data()
        marker = trial.markers['Marker']
        assert marker.unit == 'm'
        np.testing.assert_allclose(marker.x, 1.)


def test_mixed_marker_units_are_rejected(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    extra = deepcopy(trial.markers['Marker'])
    extra.name = 'Other'
    extra.convert_units('m')
    trial.add_marker(extra)
    with pytest.raises(ValueError, match='unit'):
        handler.save_data(trial, str(tmp_path / 'mixed.h5'))


def test_force_unit_conversion_survives_save(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.convert_force_units('m')
    path = tmp_path / 'forces_m.h5'
    handler.save_data(trial, str(path))
    actual = H5Handler(str(path)).load_data().forces['forceplate_0']
    assert (actual.unit_moment, actual.unit_cop) == ('Nm', 'm')


@pytest.mark.parametrize('collection', ['analogs', 'emgs'])
def test_channel_metadata_survives_save(converted_h5, tmp_path, collection):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.analogs['Fx'].unit = 'V'
    trial.analogs['Fx'].channel = 42
    if collection == 'emgs':
        trial.parse_EMG_data([42])
    path = tmp_path / 'channel_metadata.h5'
    handler.save_data(trial, str(path))
    actual = getattr(H5Handler(str(path)).load_data(), collection)['Fx']
    assert (actual.unit, actual.channel) == ('V', 42)


def test_new_force_coordinate_system_survives_save(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    extra = deepcopy(trial.forces['forceplate_0'])
    extra.name = 'new_plate'
    trial.add_force(extra)
    path = tmp_path / 'new_plate.h5'
    handler.save_data(trial, str(path))
    assert H5Handler(str(path)).load_data().forces['new_plate'].coordinateSystem == 1


@pytest.mark.parametrize('field', ['count', 'frames', 'residuals', 'virtual'])
def test_marker_metadata_after_crop_and_add(converted_h5, tmp_path, field):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.crop('markers', 5, 15)
    extra = deepcopy(trial.markers['Marker'])
    extra.name, extra.virtual = 'Virtual', 1
    trial.add_marker(extra)
    path = tmp_path / 'cropped.h5'
    handler.save_data(trial, str(path))
    with h5py.File(path, 'r') as file:
        group = file['Trajectories/Labeled']
        if field == 'count':
            assert group.attrs['NumLabeled'] == 2
        elif field == 'frames':
            attrs = file['Trajectories'].attrs
            assert attrs['EndFrame'] - attrs['StartFrame'] + 1 == 10
        elif field == 'residuals':
            assert group['Residuals'].shape[-1] == 10
        else:
            assert H5Handler(str(path)).load_data().markers['Virtual'].virtual == 1


@pytest.mark.parametrize('axis,angle', [('x', -90), ('x', 0), ('z', 90)])
def test_free_moment_vector_rotation_masking_and_export(force_data, tmp_path, axis, angle):
    force_data.Tz[:] = np.array([[1.], [2.], [3.]])
    force_data.force[:, 0] = 0.
    force_data.filter_low_forces(0.1)
    force_data.rotate(axis, angle)
    expected = {('x', -90): [1., 3., -2.], ('x', 0): [1., 2., 3.],
                ('z', 90): [-2., 1., 3.]}[(axis, angle)]
    np.testing.assert_allclose(force_data.Tz[:, 1], expected, atol=1e-14)
    np.testing.assert_array_equal(force_data.Tz[:, 0], 0.)
    path = tmp_path / 'moments.mot'
    write_mot(str(path), {force_data.name: force_data}, time=force_data.time)
    rows = np.loadtxt(path, skiprows=6)
    np.testing.assert_allclose(rows[:, 7:10], force_data.Tz.T)


@pytest.mark.parametrize('scalar_tz', [False, True], ids=['missing', 'scalar'])
def test_incompatible_free_moment_is_rejected(converted_h5, scalar_tz):
    with h5py.File(converted_h5, 'r+') as file:
        plate = file['ForcePlates/0']
        del plate['Tz']
        if scalar_tz:
            plate.create_dataset('Tz', data=np.ones(200))
    with pytest.raises(ValueError, match='Tz'):
        H5Handler(str(converted_h5)).load_data()


@pytest.mark.parametrize('method', ['lowpass_filter', 'highpass_filter'])
def test_filter_preserves_geometry_and_filters_free_moment(force_data, method):
    before = deepcopy(force_data)
    getattr(force_data, method)(5.)
    for name in ['position', 'corners', 'origin', 'rotation', 'time']:
        np.testing.assert_array_equal(getattr(force_data, name), getattr(before, name))
    assert not np.allclose(force_data.Tz, before.Tz)


def test_minimal_plate_can_be_filtered(force_data):
    minimal = ForceData(name='minimal', force=force_data.force, moment=force_data.moment,
                        cop=force_data.cop, sampling_rate=100.)
    minimal.lowpass_filter(5.)
    assert minimal.Tz.shape == minimal.force.shape


@pytest.mark.parametrize('method,arg', [('lowpass_filter', 5.), ('downsample', 3)])
def test_rejected_processing_does_not_mutate(force_data, method, arg):
    force_data.Tz = np.zeros(1)
    before = force_data.force.copy()
    with pytest.raises((ValueError, IndexError)):
        getattr(force_data, method)(arg)
    np.testing.assert_array_equal(force_data.force, before)


def test_repeated_c3d_load_has_stable_channels(synthetic_c3d):
    handler = C3DHandler(str(synthetic_c3d))
    first = handler.load_data()
    second = handler.load_data()
    assert list(second.analogs) == list(first.analogs)


def test_c3d_writer_preserves_processed_markers(synthetic_c3d, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    handler.load_data()
    handler.markers['Marker'].rotate('x', 90)
    expected = handler.markers['Marker'].get_trajectory().copy()
    path = tmp_path / 'processed.c3d'
    handler.write_c3d(str(path))
    np.testing.assert_allclose(C3DHandler(str(path)).load_data().markers['Marker'].get_trajectory(), expected)
