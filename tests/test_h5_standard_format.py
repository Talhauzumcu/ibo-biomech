from pathlib import Path

import h5py
import numpy as np
import pytest

from ibo_biomech import C3DHandler, Event, FileConverter, H5Handler, RigidBody


def assert_standard_output(path):
    with h5py.File(path) as file:
        assert 'SourceData' not in file
        assert 'SchemaVersion' not in file.attrs
        def check(name, obj):
            assert 'SchemaVersion' not in obj.attrs, name
        file.visititems(check)
        assert set(file['Events']) == {'Name', 'Description', 'Frame', 'Time',
                                      'Context', 'Subject', 'IconID', 'GenericFlag'}


def test_virtual_flags_survive_c3d_conversion_and_h5_processing(synthetic_c3d, tmp_path):
    source = C3DHandler(str(synthetic_c3d))
    trial = source.load_data()
    assert trial.markers['Marker'].virtual is False
    derived = trial.markers['Marker'] / 2
    derived.name = 'Derived'
    trial.add_marker(derived)
    c3d_path = tmp_path / 'virtual.c3d'
    source.write_c3d(str(c3d_path), trial)
    h5_path = tmp_path / 'virtual.h5'
    FileConverter.c3d_to_h5(str(c3d_path), str(h5_path))
    handler = H5Handler(str(h5_path))
    trial = handler.load_data()
    trial.crop('markers', 5, 15)
    for _ in range(2):
        handler.save_data(trial, str(h5_path))
        trial = handler.load_data()
        assert trial.markers['Marker'].virtual is False
        assert trial.markers['Derived'].virtual is True
        with h5py.File(h5_path) as file:
            group = file['Trajectories/Labeled']
            assert 'Type' not in group
            assert group['Virtual'].dtype == np.dtype(bool)
            np.testing.assert_array_equal(group['Virtual'][:], [False, True])
    trc_path = tmp_path / 'virtual.trc'
    FileConverter.h5_to_trc(str(h5_path), str(trc_path), angle=0, convert_to_meters=False)
    expected = np.column_stack([
        trial.markers['Marker'].get_trajectory().T,
        trial.markers['Derived'].get_trajectory().T,
    ])
    np.testing.assert_allclose(np.loadtxt(trc_path, skiprows=6)[:, 2:], expected, atol=1e-5)


def test_empty_marker_collection_writes_boolean_virtual_flags(converted_h5):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.markers.clear()
    handler.save_data(trial, str(converted_h5))
    assert handler.load_data().markers == {}
    with h5py.File(converted_h5) as file:
        group = file['Trajectories/Labeled']
        assert 'Type' not in group
        assert group['Virtual'].shape == (0,)
        assert group['Virtual'].dtype == np.dtype(bool)


@pytest.mark.parametrize('flags', [None, [False, True], [[False]], [0]])
def test_invalid_virtual_flags_rejected(converted_h5, flags):
    with h5py.File(converted_h5, 'r+') as file:
        group = file['Trajectories/Labeled']
        del group['Virtual']
        if flags is not None:
            group.create_dataset('Virtual', data=flags)
    with pytest.raises(ValueError, match='Virtual'):
        H5Handler(str(converted_h5)).load_data()


def test_direct_marker_crop_preserves_skipped_bodies_and_filters_events(converted_h5):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    # Body sampled twice as fast as markers, on the same recording clock.
    body = RigidBody('body', position=np.arange(120).reshape(3, 40),
                     rotation=np.eye(3), sampling_rate=200)
    trial.add_rigid_body(body)
    trial.events = [Event('before', 4, .04), Event('start', 5, .05),
                    Event('inside', 9, .09), Event('end', 15, .15)]
    handler.save_data(trial, str(converted_h5))
    trial = handler.load_data(load_rigid_bodies=False, load_events=False)
    trial.markers['Marker'].crop(5, 15)
    handler.save_data(trial, str(converted_h5))
    loaded = handler.load_data()
    np.testing.assert_array_equal(loaded.rigid_bodies['body'].position, body.position)
    np.testing.assert_array_equal(loaded.rigid_bodies['body'].time, body.time)
    assert [e.name for e in loaded.events] == ['start', 'inside']
    assert_standard_output(converted_h5)


def test_trial_crop_only_changes_selected_collection(converted_h5):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.add_rigid_body(RigidBody('body', position=np.ones((3, 20)), rotation=np.eye(3)))
    trial.events = [Event('outside', 1, .01), Event('inside', 8, .08)]
    trial.crop('markers', 5, 15)
    assert trial.rigid_bodies['body'].num_samples == 20
    assert trial.rigid_bodies['body'].time is None
    assert [e.name for e in trial.events] == ['outside', 'inside']
    trial.rigid_bodies['body'].crop(2, 8)
    assert len(trial.markers['Marker'].x) == 10
    assert trial.rigid_bodies['body'].num_samples == 6
    assert [e.name for e in trial.events] == ['outside', 'inside']
    handler.save_data(trial, str(converted_h5))
    actual = handler.load_data()
    assert actual.rigid_bodies['body'].num_samples == 6
    assert actual.rigid_bodies['body'].time is None
    assert [e.name for e in actual.events] == ['inside']
    assert_standard_output(converted_h5)


def test_reference_file_round_trip_and_crop(tmp_path):
    reference = Path(__file__).resolve().parents[1] / 'test_h5_with_all.h5'
    if not reference.exists():
        pytest.skip('Local reference recording is not committed.')
    handler = H5Handler(str(reference))
    trial = handler.load_data()
    output = tmp_path / 'all.h5'
    handler.save_data(trial, str(output))
    reloaded = H5Handler(str(output)).load_data()
    assert set(reloaded.rigid_bodies) == set(trial.rigid_bodies)
    for name, body in trial.rigid_bodies.items():
        np.testing.assert_array_equal(reloaded.rigid_bodies[name].position, body.position)
        np.testing.assert_array_equal(reloaded.rigid_bodies[name].rotation, body.rotation)
    trial.crop('markers', 30, 130)
    handler.save_data(trial, str(output))
    actual = H5Handler(str(output)).load_data()
    for body in actual.rigid_bodies.values():
        assert body.num_samples == 225
        np.testing.assert_array_equal(body.position, trial.rigid_bodies[body.name].position)
        np.testing.assert_equal(body.time, trial.rigid_bodies[body.name].time)
    assert all(118 <= e.frame < 218 for e in actual.events)
