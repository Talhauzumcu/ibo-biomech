import numpy as np
import pytest
from ibo_biomech.containers import ForceData


def test_post_init_cleans_nans_and_sets_num_samples(force_kwargs):
    kw = dict(force_kwargs)
    kw["force"] = kw["force"].copy()
    kw["force"][0, 0] = np.nan
    fd = ForceData(**kw)
    assert not np.isnan(fd.force).any()
    assert fd.num_samples == fd.force.shape[1]


def test_force_shape_assertion():
    with pytest.raises(AssertionError):
        ForceData(name="bad", force=np.zeros((2, 10)))  # must be (3, n_samples)


def test_component_properties(force_data):
    assert np.array_equal(force_data.Fx, force_data.force[0, :])
    assert np.array_equal(force_data.Fy, force_data.force[1, :])
    assert np.array_equal(force_data.Fz, force_data.force[2, :])
    assert np.array_equal(force_data.Mx, force_data.moment[0, :])
    assert np.array_equal(force_data.cop_x, force_data.cop[0, :])
    assert np.array_equal(force_data.x, force_data.position[0, :])


def test_unit_parsing_from_metadata(force_data):
    assert force_data.unit_force == "N"
    assert force_data.unit_moment == "Nmm"
    assert force_data.unit_cop == "mm"
    assert force_data.unit_position == force_data.unit_cop  # alias


def test_get_force_magnitude(force_data):
    mag = force_data.get_force_magnitude()
    expected = np.sqrt(np.sum(force_data.force**2, axis=0))
    assert np.allclose(mag, expected)


def test_filter_low_forces_zeroes_below_threshold(force_data):
    mag_before = force_data.get_force_magnitude()
    threshold = np.median(mag_before)
    force_data.filter_low_forces(threshold=threshold)
    mag_after = force_data.get_force_magnitude()
    below = mag_before < threshold
    assert np.all(mag_after[below] == 0)
    assert np.all(force_data.force[:, below] == 0)
    assert np.all(force_data.moment[:, below] == 0)
    assert np.all(force_data.cop[:, below] == 0)


def test_downsample_updates_sampling_rate_and_length(force_data):
    n0 = force_data.num_samples
    fs0 = force_data.sampling_rate
    force_data.downsample(factor=4)
    assert force_data.sampling_rate == pytest.approx(fs0 / 4)
    assert force_data.num_samples == pytest.approx(n0 / 4, abs=2)  # decimate rounding
    assert force_data.force.shape[1] == force_data.num_samples


def test_crop_trims_every_geometry_array(force_data):
    force_data.crop(10, 30)
    assert force_data.num_samples == 20
    assert force_data.force.shape == (3, 20)
    assert force_data.moment.shape == (3, 20)
    assert force_data.cop.shape == (3, 20)
    assert force_data.location.shape == (3, 4, 20)
    assert force_data.position.shape == (3, 20)
    assert force_data.rotation.shape == (3, 3, 20)
    assert force_data.Tz.shape == (20,)
    assert force_data.time.shape == (20,)


def test_crop_invalid_range_raises(force_data):
    with pytest.raises(ValueError):
        force_data.crop(0, 10_000)


def test_lowpass_filter_filters_axis1(force_data):
    force0 = force_data.force.copy()
    force_data.lowpass_filter(cutoff=5, order=4)
    assert not np.array_equal(force_data.force, force0)
    assert force_data.force.shape == force0.shape


def test_rotate_applies_to_force_moment_cop(force_data):
    force0 = force_data.force.copy()
    moment0 = force_data.moment.copy()
    cop0 = force_data.cop.copy()
    force_data.rotate(axis="z", angle_deg=30)
    assert not np.array_equal(force_data.force, force0)
    assert not np.array_equal(force_data.moment, moment0)
    assert not np.array_equal(force_data.cop, cop0)


def test_rotate_applies_to_nonzero_position(force_kwargs):
    kw = dict(force_kwargs)
    kw["position"] = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, force_kwargs["force"].shape[1]))
    fd = ForceData(**kw)
    position0 = fd.position.copy()
    fd.rotate(axis="x", angle_deg=90)
    assert not np.array_equal(fd.position, position0)
    # locked-in convention, same as MarkerData's rotate: (y, z) -> (-z, y)
    assert fd.position[1, 0] == pytest.approx(-3.0)
    assert fd.position[2, 0] == pytest.approx(2.0)


def test_convert_units_scales_moment_cop_tz(force_data):
    cop0 = force_data.cop.copy()
    moment0 = force_data.moment.copy()
    force_data.convert_units("m")
    assert np.allclose(force_data.cop, cop0 * 0.001)
    assert np.allclose(force_data.moment, moment0 * 0.001)
    assert force_data.unit_cop == "m"
    assert force_data.unit_moment == "Nm"


def test_convert_units_noop_when_already_target(force_data):
    force_data.unit_cop = "m"
    cop0 = force_data.cop.copy()
    force_data.convert_units("m")
    assert np.array_equal(force_data.cop, cop0)


def test_data_property_stacks_force_moment_cop(force_data):
    stacked = force_data.data
    assert stacked.shape == (9, force_data.num_samples)
    assert np.array_equal(stacked[:3], force_data.force)
    assert np.array_equal(stacked[3:6], force_data.moment)
    assert np.array_equal(stacked[6:9], force_data.cop)
