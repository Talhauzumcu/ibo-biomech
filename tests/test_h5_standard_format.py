from pathlib import Path

import h5py
import numpy as np
import pytest

from ibo_biomech import Event, H5Handler, RigidBody


def assert_standard_output(path):
    with h5py.File(path) as file:
        assert 'SourceData' not in file
        assert 'SchemaVersion' not in file.attrs
        def check(name, obj):
            assert 'SchemaVersion' not in obj.attrs, name
        file.visititems(check)
        assert set(file['Events']) == {'Name', 'Description', 'Frame', 'Time',
                                      'Context', 'Subject', 'IconID', 'GenericFlag'}


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
