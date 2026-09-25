import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ibo_biomech import RigidBody, rigidBody
from ibo_biomech.containers import TrialData


def test_rpy_convention_and_read_only_live_calculation():
    # Rz(90) @ Ry(30) @ Rx(0), with a noncommuting pair of rotations.
    c = np.sqrt(3) / 2
    matrix = np.array([[0, -1, 0], [c, 0, .5], [-.5, 0, c]])
    body = RigidBody('body', position=np.zeros((3, 2)), rotation=matrix)
    assert rigidBody is RigidBody
    assert body.num_samples == 2
    np.testing.assert_allclose(body.RPY, [[0, 0], [30, 30], [90, 90]], atol=1e-12)
    with pytest.raises(AttributeError):
        body.RPY = np.zeros((3, 2))
    with pytest.raises(ValueError):
        body.RPY[0, 0] = 1
    body.rotation[:, :, 0] = np.eye(3)
    np.testing.assert_allclose(body.RPY[:, 0], 0, atol=1e-12)


def test_rpy_varies_per_frame_and_round_trips_gimbal_lock():
    angles = [[10, 20, 30], [-40, -35, 100], [20, 90, 40]]
    matrices = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    body = RigidBody('body', position=np.zeros((3, 3)), rotation=matrices.transpose(1, 2, 0))
    with pytest.warns(UserWarning, match='Gimbal lock'):
        result = body.RPY
    np.testing.assert_allclose(result[:, :2].T, angles[:2], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_euler('xyz', result.T, degrees=True).as_matrix(), matrices, atol=1e-12)


def test_unknown_pose_and_independent_defaults():
    first, second = RigidBody('first'), RigidBody('second')
    assert np.isnan(first.rotation).all()
    assert np.isnan(first.RPY).all()
    first.markers.append('m')
    assert second.markers == []


@pytest.mark.parametrize('kwargs', [
    {'position': np.zeros((2, 5))},
    {'position': np.zeros((3, 0))},
    {'rotation': np.zeros((3, 3, 2))},
    {'rotation': np.zeros((3, 3))},
    {'rotation': np.diag([1, 1, -1])},
    {'rotation': np.diag([1, 1, np.nan])},
    {'time': np.arange(2)},
])
def test_invalid_pose_rejected(kwargs):
    with pytest.raises(ValueError):
        RigidBody('body', **kwargs)


def test_trial_storage_and_pose_operations():
    markers = ['m']
    positions = np.tile([[1000.], [0.], [0.]], (1, 4))
    body = RigidBody('body', markers=markers, position=positions,
                     rotation=np.eye(3), sampling_rate=100)
    trial = TrialData(rigid_bodies={'body': body})
    assert trial.rigid_body_rate == 100
    assert trial.get_rigid_body_names() == ['body']
    assert trial.get_rigid_body('body') is body
    assert trial.get_rigid_body('absent') is None
    assert TrialData().rigid_bodies == {}
    trial.rotate_rigid_bodies('z', 90)
    trial.convert_units('m')
    trial.crop('rigid_bodies', 1, 3)
    assert body.num_samples == 2
    np.testing.assert_allclose(body.position, [[0, 0], [1, 1], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(body.RPY, [[0, 0], [0, 0], [90, 90]], atol=1e-12)
    np.testing.assert_allclose(body.time, [.01, .02])
    np.testing.assert_array_equal(positions[0], 1000)
    assert body.markers == ['m']
    assert 'm' in body.markers
    assert body.markers is not markers
    frame = trial.as_df(trial.rigid_bodies)
    np.testing.assert_allclose(frame['body_yaw'], 90)
    replacement = RigidBody('body', sampling_rate=100)
    trial.add_rigid_body(replacement)
    assert trial.get_rigid_body('body') is replacement
    assert len(trial.rigid_bodies) == 1


def test_add_rigid_body_warns_on_rate_mismatch(capsys):
    trial = TrialData()
    trial.add_rigid_body(RigidBody('first', sampling_rate=100))
    trial.add_rigid_body(RigidBody('second', sampling_rate=200))
    assert 'does not match' in capsys.readouterr().out
    assert trial.rigid_body_rate == 100


@pytest.mark.parametrize('single_sample_axis', [False, True])
def test_set_rotation_broadcasts_copies_and_updates_rpy(single_sample_axis):
    body = RigidBody('body', position=np.zeros((3, 4)))
    matrix = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    value = matrix[..., None] if single_sample_axis else matrix
    body.set_rotation(value)
    np.testing.assert_allclose(body.RPY, np.tile([[10], [20], [30]], (1, 4)))
    value[:] = 0
    body.validate()
    assert body.rotation.shape == (3, 3, 4)


def test_set_rotation_full_trajectory_after_crop_and_reset():
    body = RigidBody('body', position=np.zeros((3, 4)))
    body.crop(1, 3)
    matrices = Rotation.from_euler('z', [[10], [20]], degrees=True).as_matrix().transpose(1, 2, 0)
    body.set_rotation(matrices)
    np.testing.assert_allclose(body.yaw, [10, 20])
    assert not np.shares_memory(body.rotation, matrices)
    body.set_rotation(None)
    assert body.rotation.shape == (3, 3, 2)
    assert np.isnan(body.RPY).all()


@pytest.mark.parametrize('rotation', [
    np.zeros((3, 3)),
    np.diag([1, 1, -1]),
    np.diag([1, 1, np.nan]),
    np.full((3, 3), np.inf),
    np.zeros((3, 3, 5)),
])
def test_set_rotation_rejects_invalid_input_without_changing_pose(rotation):
    body = RigidBody('body', position=np.zeros((3, 2)), rotation=np.eye(3))
    before = body.rotation.copy()
    with pytest.raises(ValueError):
        body.set_rotation(rotation)
    np.testing.assert_array_equal(body.rotation, before)
