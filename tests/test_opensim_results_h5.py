"""Named OpenSim results use the IK/ID HDF5 layout and retain explicit units."""
from copy import deepcopy

import h5py
import numpy as np
import pytest

from ibo_biomech import Data, H5Handler, IDResults, IKResults
from ibo_biomech.containers.motResults import MotResults


def test_named_results_round_trip_with_ik_and_id(converted_h5, mot_file, tmp_path):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.attach_IK_results(mot_file)
    trial.attach_ID_results(mot_file)
    trial.attach_opensim_results('joint_states', mot_file)
    states = trial.opensim_results['joint_states']
    states.add_column('/jointset/pelvis/translation', np.arange(5.), unit='m', time=states.time)
    states.unit = 'mixed'
    states.metadata['note'] = 'angles and translations'
    forces = MotResults(name='forces', time=np.arange(7.) / 50 + 2., unit='N',
                        metadata={'unit': 'N', 'source': 'analysis'})
    forces.add_column('vertical_force', np.arange(7.) * 10, unit='N')
    trial.opensim_results['forces'] = forces
    output = tmp_path / 'results.h5'
    handler.save_data(trial, str(output))
    handler = H5Handler(str(output))
    for _ in range(2):
        restored = handler.load_data()
        assert list(restored.opensim_results) == ['joint_states', 'forces']
        for name, expected in trial.opensim_results.items():
            actual = restored.opensim_results[name]
            assert type(actual) is MotResults
            assert actual.name == name
            assert actual.unit == expected.unit
            assert actual.metadata == expected.metadata
            assert actual.columns == expected.columns
            np.testing.assert_array_equal(actual.time, expected.time)
            for label, column in expected.data.items():
                assert actual[label].unit == column.unit
                np.testing.assert_array_equal(actual[label].data, column.data)
                np.testing.assert_array_equal(actual[label].time, expected.time)
        assert isinstance(restored.ik_results, IKResults)
        assert isinstance(restored.id_results, IDResults)
        np.testing.assert_array_equal(restored.ik_results.time, trial.ik_results.time)
        np.testing.assert_array_equal(restored.id_results.time, trial.id_results.time)
        handler.save_data(restored, str(output))
    with h5py.File(output) as file:
        group = file['OpenSim/joint_states']
        assert set(group) == {'Data', 'Time'}
        assert list(group.attrs['Labels']) == ['time'] + states.columns
        assert group.attrs['NumSamples'] == 5
        np.testing.assert_array_equal(group['Data'][0], states.time)


def test_replacing_cropping_and_removing_named_results(converted_h5, mot_file):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.attach_opensim_results('states', mot_file)
    trial.attach_opensim_results('remove_me', mot_file)
    handler.save_data(trial, str(converted_h5))
    trial = handler.load_data()
    states = trial.opensim_results['states']
    states.crop(1, 4)
    del states.data['pelvis_tilt']
    states.add_column('new_signal', np.array([7., 8., 9.]), unit='N')
    states.metadata['updated'] = 'yes'
    del trial.opensim_results['remove_me']
    handler.save_data(trial, str(converted_h5))
    actual = handler.load_data().opensim_results
    assert list(actual) == ['states']
    assert actual['states'].columns == ['hip_flexion_r', 'new_signal']
    assert actual['states'].metadata['updated'] == 'yes'
    np.testing.assert_array_equal(actual['states'].time, states.time)
    np.testing.assert_array_equal(actual['states']['new_signal'].data, [7., 8., 9.])
    trial.opensim_results.clear()
    handler.save_data(trial, str(converted_h5))
    assert handler.load_data().opensim_results == {}
    with h5py.File(converted_h5) as file:
        assert not len(file['OpenSim'])


def test_skipped_opensim_results_are_preserved(converted_h5, mot_file):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.attach_opensim_results('states', mot_file)
    handler.save_data(trial, str(converted_h5))
    skipped = handler.load_data(load_opensim_results=False)
    assert skipped.opensim_results == {}
    skipped.markers['Marker'].crop(5, 15)
    handler.save_data(skipped, str(converted_h5))
    actual = handler.load_data().opensim_results['states']
    np.testing.assert_array_equal(actual.time, trial.opensim_results['states'].time)
    assert actual.metadata == trial.opensim_results['states'].metadata


def test_missing_group_and_empty_results(converted_h5):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    assert trial.opensim_results == {}
    trial.opensim_results['empty'] = MotResults(name='empty')
    handler.save_data(trial, str(converted_h5))
    actual = handler.load_data().opensim_results['empty']
    assert actual.columns == []
    assert actual.unit == ''
    assert actual.time.shape == (0,)


@pytest.mark.parametrize('samples', [0, 1])
def test_short_results_round_trip(converted_h5, samples):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    time = np.arange(samples, dtype=float)
    result = MotResults(name='short', time=time, unit='N')
    result.data['force'] = Data('force', data=np.ones(samples), unit='N')
    trial.opensim_results['short'] = result
    handler.save_data(trial, str(converted_h5))
    actual = handler.load_data().opensim_results['short']
    np.testing.assert_array_equal(actual.time, time)
    np.testing.assert_array_equal(actual['force'].data, np.ones(samples))
    assert actual['force'].sampling_rate is None


@pytest.mark.parametrize('invalid', ['name', 'type', 'shape', 'time', 'clock', 'label'])
def test_invalid_opensim_save_leaves_destination_unchanged(converted_h5, mot_file, invalid):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.attach_opensim_results('states', mot_file)
    result = trial.opensim_results['states']
    if invalid == 'name':
        trial.opensim_results[''] = trial.opensim_results.pop('states')
    elif invalid == 'type':
        trial.opensim_results['states'] = None
    elif invalid == 'shape':
        result.data['pelvis_tilt'].data = np.ones((1, 5))
    elif invalid == 'time':
        result.time = result.time.reshape(1, -1)
    elif invalid == 'clock':
        result.data['pelvis_tilt'].time = result.time + 1
    else:
        result.data['time'] = deepcopy(result.data['pelvis_tilt'])
    before = converted_h5.read_bytes()
    with pytest.raises(ValueError, match='OpenSim'):
        handler.save_data(trial, str(converted_h5))
    assert converted_h5.read_bytes() == before


@pytest.mark.parametrize('corruption', ['count', 'labels', 'units', 'data', 'clock', 'missing'])
def test_inconsistent_opensim_h5_rejected(converted_h5, mot_file, corruption):
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    trial.attach_opensim_results('states', mot_file)
    handler.save_data(trial, str(converted_h5))
    with h5py.File(converted_h5, 'r+') as file:
        group = file['OpenSim/states']
        if corruption == 'count':
            group.attrs['NumSamples'] += 1
        elif corruption == 'labels':
            group.attrs['Labels'] = ['time', 'pelvis_tilt', 'pelvis_tilt']
        elif corruption == 'units':
            group.attrs['ColumnUnits'] = ['deg']
        elif corruption == 'data':
            del group['Data']
            group.create_dataset('Data', data=np.ones((2, 5)))
        elif corruption == 'clock':
            group['Time'][:] += 1
        else:
            del group['Time']
    with pytest.raises(ValueError, match='OpenSim/states'):
        handler.load_data()
