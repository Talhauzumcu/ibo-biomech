"""Source sample indices and nested HDF5 metadata across processing and I/O."""
from collections.abc import Mapping

import h5py
import numpy as np
import pytest
from ezc3d import c3d

from ibo_biomech import AnalogData, EMGData, ForceData, MarkerData, RigidBody
from ibo_biomech import C3DHandler, FileConverter, H5Handler


def make_signal(cls, **kwargs):
    samples = np.arange(11, dtype=float)
    if cls is MarkerData:
        data = dict(x=samples, y=samples.copy(), z=samples.copy())
    elif cls is ForceData:
        data = dict(force=np.tile(samples, (3, 1)))
    elif cls is RigidBody:
        data = dict(position=np.tile(samples, (3, 1)))
    else:
        data = dict(data=samples)
    return cls('signal', sampling_rate=100, **data, **kwargs)


@pytest.mark.parametrize('cls', [MarkerData, AnalogData, EMGData, ForceData, RigidBody])
def test_nonzero_origin_and_repeated_crop(cls):
    signal = make_signal(cls, first_frame=np.int64(100))
    assert (signal.first_frame, signal.last_frame, signal.num_samples) == (100, 110, 11)
    np.testing.assert_allclose(signal.time, 1 + np.arange(11) / 100)
    signal.crop(2, 9)
    signal.crop(1, 5)
    assert (signal.first_frame, signal.last_frame, signal.num_samples) == (103, 106, 4)
    np.testing.assert_allclose(signal.time, [1.03, 1.04, 1.05, 1.06])


@pytest.mark.parametrize('cls', [MarkerData, AnalogData, EMGData, ForceData, RigidBody])
@pytest.mark.parametrize('kwargs', [
    {'first_frame': None}, {'first_frame': 1.5}, {'first_frame': True},
    {'num_samples': 99}, {'num_samples': 11.0},
    {'last_frame': 99}, {'last_frame': 10.0},
])
def test_invalid_sample_metadata_rejected(cls, kwargs):
    with pytest.raises(ValueError):
        make_signal(cls, **kwargs)


@pytest.mark.parametrize('cls', [MarkerData, AnalogData, EMGData, ForceData, RigidBody])
def test_supplied_time_is_preserved(cls):
    time = 25 + np.arange(11) / 100
    signal = make_signal(cls, first_frame=100, last_frame=110, num_samples=11, time=time)
    np.testing.assert_array_equal(signal.time, time)


@pytest.mark.parametrize('cls', [MarkerData, AnalogData, EMGData, ForceData, RigidBody])
@pytest.mark.parametrize('field', ['last_frame', 'num_samples'])
def test_invalid_metadata_after_construction_rejects_crop_without_mutation(cls, field):
    signal = make_signal(cls, first_frame=100)
    before = signal.time.copy()
    setattr(signal, field, None)
    with pytest.raises(ValueError):
        signal.crop(2, 5)
    np.testing.assert_array_equal(signal.time, before)
    assert signal.first_frame == 100


@pytest.mark.parametrize('position', [np.ones(3), np.empty((0, 3)), np.array(1.)])
def test_rigid_body_validates_position_before_count(position):
    with pytest.raises(ValueError, match='position must have shape'):
        RigidBody('body', position=position)


def test_downsample_then_crop_retains_source_indices_and_time():
    plate = make_signal(ForceData, first_frame=100)
    original_time = plate.time.copy()
    plate.downsample(3)
    assert (plate.first_frame, plate.last_frame, plate.num_samples, plate.frame_step) == (100, 109, 4, 3)
    plate.crop(1, 4)
    assert (plate.first_frame, plate.last_frame, plate.num_samples) == (103, 109, 3)
    plate.downsample(2)
    assert (plate.first_frame, plate.last_frame, plate.num_samples, plate.frame_step) == (103, 109, 2, 6)
    np.testing.assert_array_equal(plate.time, original_time[[3, 9]])
    plate.crop(1, 2)
    assert (plate.first_frame, plate.last_frame, plate.num_samples) == (109, 109, 1)
    plate.validate()


@pytest.fixture
def offset_h5(synthetic_c3d, tmp_path):
    raw = c3d(str(synthetic_c3d))
    raw['header']['points']['first_frame'] = 500
    raw.write(str(synthetic_c3d))
    path = tmp_path / 'offset.h5'
    FileConverter.c3d_to_h5(str(synthetic_c3d), str(path), subject_id='P01', body_mass=70.)
    return path


def assert_metadata_equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, Mapping):
            assert_metadata_equal(actual[key], value)
        else:
            np.testing.assert_equal(np.asarray(actual[key]), np.asarray(value))


def test_nested_metadata_parameters_subject_and_dataframe(offset_h5, synthetic_c3d, tmp_path):
    handler = H5Handler(str(offset_h5))
    trial = handler.load_data()
    metadata = trial.metadata
    assert set(metadata) == {'FileInfo', 'Project', 'Location', 'C3DParameters'}
    assert metadata['Project']['SubjectID'] == 'P01'
    assert metadata['Project']['BodyMass'] == 70.
    assert metadata['FileInfo']['PathFile'] == str(synthetic_c3d)
    assert_metadata_equal(metadata['C3DParameters'], c3d(str(synthetic_c3d))['parameters'])
    subject = handler.load_subject_data()
    assert subject.id == 'P01' and subject.body_mass == 70.
    frame = trial.as_df(trial.markers)
    assert (frame['SubjectID'] == 'P01').all()
    assert 'C3DParameters' not in frame
    # Signal saves retain source parameters even after crops change the live counts.
    trial.crop('markers', 2, 10)
    output = tmp_path / 'processed.h5'
    handler.save_data(trial, str(output))
    assert_metadata_equal(H5Handler(str(output)).load_data().metadata['C3DParameters'],
                          metadata['C3DParameters'])
    with h5py.File(output) as file:
        assert not len(file['MetaData'].attrs)
        assert 'LastUpdate' in file['MetaData/FileInfo'].attrs

    # Keep the existing save_data contract: metadata edits use modify_metadata.
    trial.metadata['Project']['SubjectID'] = 'in-memory edit'
    handler.save_data(trial, str(output))
    assert H5Handler(str(output)).load_data().metadata['Project']['SubjectID'] == 'P01'


@pytest.mark.parametrize('same_path', [True, False])
def test_nested_metadata_update_preserves_other_fields(offset_h5, tmp_path, same_path):
    handler = H5Handler(str(offset_h5))
    before = handler.load_data().metadata
    source_bytes = offset_h5.read_bytes()
    destination = offset_h5 if same_path else tmp_path / 'annotated.h5'
    updated = handler.modify_metadata({'Project': {'SubjectID': 'P02'}, 'Location': {'Lat': 52.}},
                                     out_path=str(destination))
    metadata = updated.load_data().metadata
    assert metadata['Project']['SubjectID'] == 'P02'
    assert metadata['Project']['BodyMass'] == 70.
    assert metadata['Location'] == {'Lat': 52., 'Lon': 'Unknown'}
    assert metadata['FileInfo']['LastUpdate'] != before['FileInfo']['LastUpdate']
    assert_metadata_equal(metadata['C3DParameters'], before['C3DParameters'])
    assert updated.load_subject_data().id == 'P02'
    if not same_path:
        assert offset_h5.read_bytes() == source_bytes


@pytest.mark.parametrize('updates', [{'SubjectID': 'P02'}, {'Project': 'P02'},
                                    {'Project': {'SubjectID': 'P02', 'Bad': object()}}])
def test_invalid_metadata_updates_leave_file_unchanged(offset_h5, updates):
    before = offset_h5.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        H5Handler(str(offset_h5)).modify_metadata(updates)
    assert offset_h5.read_bytes() == before


@pytest.mark.parametrize('operation', ['load', 'save', 'modify'])
def test_flat_metadata_is_rejected(offset_h5, operation):
    handler = H5Handler(str(offset_h5))
    trial = handler.load_data()
    with h5py.File(offset_h5, 'r+') as file:
        del file['MetaData']
        file.create_group('MetaData').attrs['SubjectID'] = 'old'
    before = offset_h5.read_bytes()
    with pytest.raises(ValueError, match='MetaData requires'):
        if operation == 'load':
            handler.load_data()
        elif operation == 'save':
            handler.save_data(trial, str(offset_h5))
        else:
            handler.modify_metadata({'Project': {'SubjectID': 'P02'}})
    assert offset_h5.read_bytes() == before


def test_all_sample_ranges_survive_processing_and_two_h5_saves(offset_h5):
    handler = H5Handler(str(offset_h5))
    trial = handler.load_data()
    assert (trial.markers['Marker'].first_frame, trial.markers['Marker'].last_frame) == (500, 519)
    assert (trial.analogs['Fx'].first_frame, trial.analogs['Fx'].last_frame) == (5000, 5199)
    assert trial.forces['forceplate_0'].first_frame == 5000
    trial.crop('markers', 2, 10)
    trial.crop('analogs', 20, 100)
    trial.parse_EMG_data([0])
    assert trial.emgs['Fx'].first_frame == 5020
    trial.forces['forceplate_0'].downsample(3)
    trial.forces['forceplate_0'].crop(2, 20)
    trial.add_rigid_body(RigidBody('body', position=np.ones((3, 20)), first_frame=500,
                                    sampling_rate=100))
    trial.rigid_bodies['body'].crop(2, 10)
    virtual = trial.markers['Marker'] / 2
    trial.add_marker(virtual)
    for _ in range(2):
        handler.save_data(trial, str(offset_h5))
        restored = handler.load_data()
        for collection in ('markers', 'analogs', 'emgs', 'forces', 'rigid_bodies'):
            for name, signal in getattr(trial, collection).items():
                actual = getattr(restored, collection)[name]
                for field in ('first_frame', 'last_frame', 'num_samples', 'sampling_rate'):
                    assert getattr(actual, field) == getattr(signal, field)
                np.testing.assert_array_equal(actual.time, signal.time)
        assert restored.forces['forceplate_0'].frame_step == 3
        trial = restored
    # The stride must remain useful after reloading, including for another crop.
    trial.forces['forceplate_0'].crop(1, 3)
    assert (trial.forces['forceplate_0'].first_frame,
            trial.forces['forceplate_0'].last_frame) == (5009, 5012)


@pytest.mark.parametrize('path,count', [('Trajectories', 'NumFrames'), ('Analog', 'NumSamples'),
                                      ('ForcePlates/0', 'NumSamples')])
@pytest.mark.parametrize('corruption', ['count', 'end', 'missing_start'])
def test_inconsistent_h5_sample_ranges_rejected(offset_h5, path, count, corruption):
    with h5py.File(offset_h5, 'r+') as file:
        attrs = file[path].attrs
        if corruption == 'count':
            attrs[count] += 1
        elif corruption == 'end':
            attrs['EndFrame'] += 1
        else:
            del attrs['StartFrame']
    with pytest.raises(ValueError):
        H5Handler(str(offset_h5)).load_data()


def test_c3d_crop_round_trip_uses_each_stream_sample_grid(offset_h5, synthetic_c3d, tmp_path):
    handler = C3DHandler(str(synthetic_c3d))
    handler.load_data()
    handler.slice_c3d(2, 9)
    destination = tmp_path / 'cropped.c3d'
    handler.write_c3d(str(destination))
    actual = C3DHandler(str(destination)).load_data()
    for collection in ('markers', 'analogs', 'forces'):
        for name, signal in getattr(handler.trial, collection).items():
            restored = getattr(actual, collection)[name]
            assert (restored.first_frame, restored.last_frame, restored.num_samples) == (
                signal.first_frame, signal.last_frame, signal.num_samples)


@pytest.mark.parametrize('collection', ['analogs', 'forces'])
def test_c3d_rejects_frame_offsets_that_disagree_with_source_clock(
        offset_h5, synthetic_c3d, tmp_path, collection):
    handler = C3DHandler(str(synthetic_c3d))
    trial = handler.load_data()
    for signal in getattr(trial, collection).values():
        signal.first_frame += 1
        signal.last_frame += 1
    destination = tmp_path / 'unchanged.c3d'
    destination.write_bytes(b'existing file')
    with pytest.raises(ValueError, match='frame'):
        handler.write_c3d(str(destination))
    assert destination.read_bytes() == b'existing file'


@pytest.mark.parametrize('collection', ['markers', 'analogs'])
def test_generated_collection_clock_retains_frame_offset(offset_h5, collection):
    handler = H5Handler(str(offset_h5))
    trial = handler.load_data()
    expected = next(iter(getattr(trial, collection).values())).time.copy()
    for signal in getattr(trial, collection).values():
        signal.time = None
    handler.save_data(trial, str(offset_h5))
    actual = next(iter(getattr(handler.load_data(), collection).values()))
    np.testing.assert_allclose(actual.time, expected)


@pytest.mark.parametrize('collection', ['markers', 'analogs', 'forces'])
def test_invalid_sample_range_save_is_atomic(offset_h5, collection):
    handler = H5Handler(str(offset_h5))
    trial = handler.load_data()
    next(iter(getattr(trial, collection).values())).last_frame += 1
    before = offset_h5.read_bytes()
    with pytest.raises(ValueError, match='last_frame'):
        handler.save_data(trial, str(offset_h5))
    assert offset_h5.read_bytes() == before
