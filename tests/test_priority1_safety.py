"""Validation, ownership and metadata regressions for priority 1 fixes."""
from copy import deepcopy

import h5py
import numpy as np
import pytest
from ezc3d import c3d

from ibo_biomech import C3DHandler, FileConverter, H5Handler
from ibo_biomech.containers import ForceData
from ibo_biomech.utils.utils import get_fp_cs, write_mot


def assert_plate_unchanged(actual, before):
    for name in ('force', 'moment', 'cop', 'Tz', 'position', 'corners', 'rotation', 'origin', 'time'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(before, name))
    assert actual.num_samples == before.num_samples
    assert actual.sampling_rate == before.sampling_rate
    assert actual.metadata == before.metadata


@pytest.mark.parametrize('factor', [True, False, 0, -2, 2.5])
def test_downsample_rejected_factors_are_atomic(force_data, factor):
    before = deepcopy(force_data)
    with pytest.raises(ValueError, match='positive integer'):
        force_data.downsample(factor)
    assert_plate_unchanged(force_data, before)


@pytest.mark.parametrize('problem', ['short', 'backwards', 'irregular', 'wrong_rate'])
def test_downsample_rejects_invalid_clocks_before_mutation(force_data, problem):
    if problem == 'short':
        force_data.time = force_data.time[:-1]
    elif problem == 'backwards':
        force_data.time = force_data.time[::-1]
    elif problem == 'irregular':
        force_data.time[2] += .001
    else:
        force_data.sampling_rate *= 2
    before = deepcopy(force_data)
    with pytest.raises(ValueError):
        force_data.downsample(3)
    assert_plate_unchanged(force_data, before)


def test_downsample_infers_rate_and_preserves_supplied_time(force_data):
    force_data.sampling_rate = None
    force_data.time += 5.
    expected = force_data.time[::3].copy()
    force_data.downsample(np.int64(3))
    assert force_data.sampling_rate == pytest.approx(100. / 3)
    np.testing.assert_array_equal(force_data.time, expected)


def test_downsample_without_clock_does_not_invent_one():
    plate = ForceData(name='unknown', force=np.ones((3, 41)))
    plate.downsample(4)
    assert plate.num_samples == 11
    assert plate.time is None and plate.sampling_rate is None
    assert np.isnan(plate.rotation).all()


def test_downsample_one_is_noop(force_data):
    before = deepcopy(force_data)
    force_data.downsample(1)
    assert_plate_unchanged(force_data, before)


@pytest.mark.parametrize('operation', ['filter', 'mask', 'rotate', 'crop', 'units'])
def test_invalid_vector_does_not_partially_mutate(force_data, operation):
    force_data.Tz = np.zeros(force_data.num_samples)
    before = deepcopy(force_data)
    calls = {'filter': lambda: force_data.highpass_filter(5),
             'mask': lambda: force_data.filter_low_forces(100),
             'rotate': lambda: force_data.rotate('x', 90),
             'crop': lambda: force_data.crop(2, 10),
             'units': lambda: force_data.convert_units('m')}
    with pytest.raises(ValueError, match='Tz'):
        calls[operation]()
    assert_plate_unchanged(force_data, before)


def test_unknown_geometry_survives_filtering_and_serialization(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    plate = ForceData(name='unknown', force=np.ones((3, 200)), sampling_rate=1000.)
    plate.lowpass_filter(5)
    assert plate.Tz.shape == (3, 200)
    assert np.isnan(plate.rotation).all() and np.isnan(plate.corners).all()
    trial.add_force(plate)
    path = tmp_path / 'unknown.h5'
    handler.save_data(trial, str(path))
    actual = H5Handler(str(path)).load_data().forces['unknown']
    assert np.isnan(actual.rotation).all() and np.isnan(actual.origin).all()


@pytest.mark.parametrize('field,shape', [('Tz', (200,)), ('moment', (3, 1)),
                                        ('corners', (4, 3, 200)), ('rotation', (3, 2, 200))])
def test_force_constructor_rejects_malformed_arrays(force_kwargs, field, shape):
    force_kwargs[field] = np.zeros(shape)
    with pytest.raises(ValueError, match=field):
        ForceData(**force_kwargs)


def test_measured_skewed_corners_produce_proper_rotation():
    corners = np.array([[200., 1., 0., 201.], [400., 399., 0., 0.], [1., 0., 0., 0.]])
    rotation, position = get_fp_cs(corners)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    np.testing.assert_allclose(position, corners.mean(axis=1))


def test_only_current_h5_schema_is_accepted(converted_h5):
    with h5py.File(converted_h5, 'r+') as file:
        del file['ForcePlates/0'].attrs['SchemaVersion']
    with pytest.raises(ValueError, match='only force schema'):
        H5Handler(str(converted_h5)).load_data()


@pytest.mark.parametrize('attribute,value', [('NumSamples', 199), ('FreeMomentFrame', 'local')])
def test_inconsistent_force_metadata_is_rejected(converted_h5, attribute, value):
    with h5py.File(converted_h5, 'r+') as file:
        file['ForcePlates/0'].attrs[attribute] = value
    with pytest.raises(ValueError, match=attribute):
        H5Handler(str(converted_h5)).load_data()


def test_marker_measurement_metadata_survives_crop_twice(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    marker = trial.markers['Marker']
    marker.residuals = np.arange(20, dtype=float)
    marker.residuals[8] = -1.
    marker.camera_masks[0] = np.arange(20) % 2 == 0
    marker.sample_types = np.arange(20)
    marker.crop(5, 15)
    for index in range(2):
        path = tmp_path / f'measurements_{index}.h5'
        handler.save_data(trial, str(path))
        handler = H5Handler(str(path))
        trial = handler.load_data()
        actual = trial.markers['Marker']
        assert actual.first_frame == 5
        np.testing.assert_array_equal(actual.residuals, marker.residuals)
        np.testing.assert_array_equal(actual.sample_types, marker.sample_types)
        np.testing.assert_array_equal(actual.camera_masks, marker.camera_masks)


def test_virtual_midpoint_after_crop_keeps_frame_and_clock(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.crop('markers', 5, 15)
    marker = trial.markers['Marker']
    midpoint = (marker + marker) / 2
    midpoint.name = 'Midpoint'
    trial.add_marker(midpoint)
    path = tmp_path / 'midpoint.h5'
    handler.save_data(trial, str(path))
    actual = H5Handler(str(path)).load_data().markers['Midpoint']
    assert actual.first_frame == 5 and actual.virtual == 1
    np.testing.assert_array_equal(actual.time, marker.time)
    assert np.isnan(actual.residuals).all()
    assert actual.camera_masks is None
    np.testing.assert_array_equal(actual.sample_types, np.full(10, 2))


def test_cropped_annotations_are_archived_with_source_scope(converted_h5):
    with h5py.File(converted_h5, 'r+') as file:
        file['Events'].create_dataset('Time', data=[0.01, 0.19])
        file['RigidBodies'].create_dataset('Pose', data=np.ones((4, 4, 20)))
        file['Trajectories'].create_group('Unlabeled').create_dataset('Data', data=np.ones((1, 4, 20)))
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.crop('markers', 5, 15)
    handler.save_data(trial, str(converted_h5))
    with h5py.File(converted_h5) as file:
        assert 'Events' not in file and 'RigidBodies' not in file
        assert 'Unlabeled' not in file['Trajectories']
        np.testing.assert_array_equal(file['SourceData/Events/Time'][:], [0.01, 0.19])
        assert 'not aligned' in file['SourceData'].attrs['Scope']


def test_rejected_save_preserves_existing_destination(converted_h5, tmp_path, monkeypatch):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    output = tmp_path / 'destination.h5'
    output.write_bytes(b'keep this destination')
    def fail_after_marker_write(*args):
        raise RuntimeError('simulated write failure')
    monkeypatch.setattr(handler, '_save_forces', fail_after_marker_write)
    with pytest.raises(RuntimeError, match='simulated'):
        handler.save_data(trial, str(output))
    assert output.read_bytes() == b'keep this destination'
    assert not list(tmp_path.glob('.destination.h5.*'))


def test_mismatched_marker_clocks_rejected_before_file_creation(converted_h5, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    other = deepcopy(trial.markers['Marker'])
    other.name = 'other'
    other.time += .001
    trial.add_marker(other)
    output = tmp_path / 'invalid.h5'
    with pytest.raises(ValueError, match='clock'):
        handler.save_data(trial, str(output))
    assert not output.exists()


def test_removed_channels_are_removed_and_skipped_collections_preserved(converted_h5):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data(load_markers=False)
    assert trial.markers == {}
    del trial.analogs['Fx']
    trial.forces.clear()
    handler.save_data(trial, str(converted_h5))
    actual = handler.load_data()
    assert 'Marker' in actual.markers
    assert 'Fx' not in actual.analogs and not actual.forces


def test_write_mot_defaults_to_offset_clock(force_data, tmp_path):
    force_data.time += 5.
    output = tmp_path / 'offset.mot'
    write_mot(str(output), {force_data.name: force_data})
    np.testing.assert_array_equal(np.loadtxt(output, skiprows=6)[:, 0], force_data.time)


def test_c3d_trial_is_shared_and_raw_export_is_explicit(synthetic_c3d, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    trial = handler.load_data()
    assert trial.markers is handler.markers
    before = trial.markers['Marker'].get_trajectory().copy()
    trial.rotate_markers('x', 90)
    processed, raw = tmp_path / 'processed.c3d', tmp_path / 'raw.c3d'
    handler.write_c3d(str(processed), trial)
    handler.write_raw_c3d(str(raw))
    np.testing.assert_allclose(C3DHandler(str(processed)).load_data().markers['Marker'].get_trajectory(),
                               trial.markers['Marker'].get_trajectory())
    np.testing.assert_allclose(C3DHandler(str(raw)).load_data().markers['Marker'].get_trajectory(), before)


def test_c3d_force_edits_rejected_before_overwriting(synthetic_c3d, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    trial = handler.load_data()
    trial.forces['forceplate_0'].rotate('x', 90)
    output = tmp_path / 'keep.c3d'
    output.write_bytes(b'original destination')
    with pytest.raises(ValueError, match='Processed force vectors'):
        handler.write_c3d(str(output), trial)
    assert output.read_bytes() == b'original destination'


def test_c3d_analog_processing_and_reordering_preserves_plate_mapping(synthetic_c3d, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    trial = handler.load_data()
    trial.analogs['Fz'].data *= 2
    reordered = dict(reversed(list(trial.analogs.items())))
    trial.analogs.clear()
    trial.analogs.update(reordered)
    path = tmp_path / 'analogs.c3d'
    handler.write_c3d(str(path), trial)
    reloaded = C3DHandler(str(path)).load_data()
    np.testing.assert_allclose(reloaded.forces['forceplate_0'].force, trial.forces['forceplate_0'].force * 2)
    assert list(reloaded.analogs) == list(trial.analogs)


def test_c3d_crop_preserves_offsets_events_and_validity(synthetic_c3d, tmp_path):
    raw = c3d(str(synthetic_c3d))
    raw['header']['points']['first_frame'] = 500
    raw['data']['meta_points']['residuals'][0, 0, 8] = -1.
    raw['data']['meta_points']['camera_masks'][0, 0, 7] = True
    raw.add_event([0., 5.01], label='before')
    raw.add_event([0., 5.08], label='inside')
    raw.add_event([0., 5.18], label='after')
    raw.write(str(synthetic_c3d))
    handler = C3DHandler(str(synthetic_c3d))
    trial = handler.load_data()
    handler.slice_c3d(5, 14)
    assert trial is handler.trial
    assert len(trial.markers['Marker'].x) == 10
    assert len(trial.analogs) == 6
    path = tmp_path / 'cropped.c3d'
    handler.write_c3d(str(path))
    result = C3DHandler(str(path))
    actual = result.load_data()
    assert actual.markers['Marker'].first_frame == 505
    assert actual.markers['Marker'].time[0] == pytest.approx(5.05)
    assert actual.markers['Marker'].residuals[3] < 0
    assert actual.markers['Marker'].camera_masks[0, 2]
    assert result.c3d_data['parameters']['EVENT']['LABELS']['value'] == ['inside']
    assert actual.forces['forceplate_0'].num_samples == 100


def test_duplicate_analog_labels_remain_stable_after_reloading(synthetic_c3d):
    raw = c3d(str(synthetic_c3d))
    raw['parameters']['ANALOG']['LABELS']['value'] = ['A', 'A', 'A_2', 'A', 'B', 'C']
    raw.write(str(synthetic_c3d))
    handler = C3DHandler(str(synthetic_c3d))
    names = list(handler.load_data().analogs)
    assert names == ['A', 'A_3', 'A_2', 'A_4', 'B', 'C']
    assert list(handler.load_data().analogs) == names


def test_marker_only_conversion_has_valid_empty_analog_group(tmp_path):
    raw = c3d()
    raw['parameters']['POINT']['RATE']['value'] = [100.]
    raw['parameters']['POINT']['LABELS']['value'] = ['Marker']
    raw['parameters']['POINT']['UNITS']['value'] = ['mm']
    raw['data']['points'] = np.ones((4, 1, 20))
    source, output = tmp_path / 'marker_only.c3d', tmp_path / 'marker_only.h5'
    raw.write(str(source))
    FileConverter.c3d_to_h5(str(source), str(output))
    trial = H5Handler(str(output)).load_data()
    assert list(trial.markers) == ['Marker']
    assert trial.analogs == {} and trial.forces == {}
