from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ibo_biomech import H5Handler, RigidBody


@pytest.fixture
def saved_bodies(tmp_path):
    path = tmp_path / 'bodies.h5'
    with h5py.File(path, 'w') as file:
        file.create_group('MetaData')
    handler = H5Handler(str(path))
    trial = handler.load_data()
    trial.add_rigid_body(RigidBody('pelvis/左', markers=['LASI', '左骨', 'RASI'],
        position=np.arange(15).reshape(3, 5),
        rotation=Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix(),
        sampling_rate=100, time=2 + np.arange(5) / 100, unit='m'))
    trial.add_rigid_body(RigidBody('unknown', position=np.full((3, 2), np.nan)))
    handler.save_data(trial, str(path))
    return handler, trial


def test_round_trip_preserves_all_fields_and_derived_rpy(saved_bodies):
    handler, expected = saved_bodies
    actual = handler.load_data()
    assert list(actual.rigid_bodies) == list(expected.rigid_bodies)
    for name, body in expected.rigid_bodies.items():
        result = actual.rigid_bodies[name]
        for field in ('name', 'markers', 'num_samples', 'unit', 'sampling_rate'):
            assert getattr(result, field) == getattr(body, field)
        for field in ('position', 'rotation', 'time', 'RPY'):
            np.testing.assert_equal(getattr(result, field), getattr(body, field))
        assert not result.RPY.flags.writeable
    with h5py.File(handler.h5_path) as file:
        assert file['RigidBodies'].attrs['SchemaVersion'] == 1
        assert 'RPY' not in file['RigidBodies/0']


def test_crop_replace_remove_and_repeated_save(saved_bodies):
    handler, _ = saved_bodies
    trial = handler.load_data()
    trial.rigid_bodies.pop('unknown')
    body = trial.rigid_bodies['pelvis/左']
    body.crop(1, 4)
    body.rotate('z', 10)
    body.markers = ['new']
    for _ in range(2):
        handler.save_data(trial, handler.h5_path)
        actual = handler.load_data()
        assert list(actual.rigid_bodies) == ['pelvis/左']
        np.testing.assert_equal(actual.rigid_bodies[body.name].rotation, body.rotation)
        np.testing.assert_equal(actual.rigid_bodies[body.name].time, body.time)
        assert actual.rigid_bodies[body.name].markers == ['new']
    trial.rigid_bodies.clear()
    handler.save_data(trial, handler.h5_path)
    assert handler.load_data().rigid_bodies == {}


def test_skipped_bodies_preserved_when_other_clocks_change(saved_bodies):
    handler, _ = saved_bodies
    with h5py.File(handler.h5_path, 'r+') as file:
        group = file['Analog']
        del group['Data']
        group.create_dataset('Data', data=np.ones((1, 3)))
    trial = handler.load_data(load_rigid_bodies=False)
    assert not trial.rigid_bodies
    handler.save_data(trial, handler.h5_path)
    assert len(handler.load_data().rigid_bodies) == 2
    with h5py.File(handler.h5_path) as file:
        assert 'SourceData/RigidBodies' not in file


def test_rigid_body_clock_change_archives_events_only(saved_bodies):
    handler, _ = saved_bodies
    with h5py.File(handler.h5_path, 'r+') as file:
        file.create_group('Events').create_dataset('Time', data=[2.01])
    trial = handler.load_data()
    trial.rigid_bodies['pelvis/左'].crop(1, 4)
    handler.save_data(trial, handler.h5_path)
    with h5py.File(handler.h5_path) as file:
        assert 'SourceData/Events/Time' in file
        assert 'SourceData/RigidBodies' not in file
        assert file['RigidBodies/0/Position'].shape == (3, 3)


@pytest.mark.parametrize('corruption', ['count', 'shape', 'duplicate', 'markers', 'missing', 'version'])
def test_malformed_rigid_bodies_rejected(saved_bodies, corruption):
    handler, _ = saved_bodies
    with h5py.File(handler.h5_path, 'r+') as file:
        group = file['RigidBodies/0']
        if corruption == 'count':
            group.attrs['NumSamples'] = 7
        elif corruption == 'shape':
            del group['Rotation']
            group.create_dataset('Rotation', data=np.eye(3))
        elif corruption == 'duplicate':
            file['RigidBodies/1'].attrs['Name'] = group.attrs['Name']
        elif corruption == 'markers':
            del group['Markers']
            group.create_dataset('Markers', data=[1, 2])
        elif corruption == 'missing':
            del group['Position']
        else:
            file['RigidBodies'].attrs['SchemaVersion'] = 99
    with pytest.raises(ValueError):
        handler.load_data()


@pytest.mark.parametrize('invalid', ['name', 'markers', 'rotation'])
def test_invalid_save_leaves_destination_unchanged(saved_bodies, invalid):
    handler, trial = saved_bodies
    body = trial.rigid_bodies['pelvis/左']
    before = Path(handler.h5_path).read_bytes()
    if invalid == 'name':
        body.name = 'other'
    elif invalid == 'markers':
        body.markers = [123]
    else:
        body.rotation[:] = 0
    with pytest.raises(ValueError):
        handler.save_data(trial, handler.h5_path)
    assert Path(handler.h5_path).read_bytes() == before


def test_legacy_payload_preserved_when_adding_typed_bodies(tmp_path):
    path = tmp_path / 'legacy.h5'
    with h5py.File(path, 'w') as file:
        file.create_group('MetaData')
        file.create_group('RigidBodies').create_dataset('Pose', data=np.eye(4))
    handler = H5Handler(str(path))
    trial = handler.load_data()
    assert trial.rigid_bodies == {}
    handler.save_data(trial, str(path))
    with h5py.File(path) as file:
        np.testing.assert_equal(file['RigidBodies/Pose'][:], np.eye(4))
    trial.add_rigid_body(RigidBody('new'))
    handler.save_data(trial, str(path))
    with h5py.File(path) as file:
        np.testing.assert_equal(file['SourceData/RigidBodies/Pose'][:], np.eye(4))
    assert list(handler.load_data().rigid_bodies) == ['new']
