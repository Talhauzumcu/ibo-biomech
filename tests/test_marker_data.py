import numpy as np
import pytest
from ibo_biomech.containers import MarkerData


def test_trajectory_shape(marker_data):
    traj = marker_data.get_trajectory()
    assert traj.shape == (3, len(marker_data.x))


def test_frame_trajectory_is_homogeneous(marker_data):
    frame = marker_data.get_frame_trajectory()
    assert frame.shape == (4, len(marker_data.x))
    assert np.all(frame[3, :] == 1.0)


def test_magnitude(marker_data):
    mag = marker_data.get_magnitude()
    expected = np.sqrt(marker_data.x**2 + marker_data.y**2 + marker_data.z**2)
    assert np.allclose(mag, expected)


def test_crop_trims_all_axes_and_time(marker_data):
    n0 = len(marker_data.x)
    marker_data.crop(10, 30)
    assert len(marker_data.x) == 20
    assert len(marker_data.y) == 20
    assert len(marker_data.z) == 20
    assert len(marker_data.time) == 20
    assert n0 == 200  # sanity on fixture


def test_crop_invalid_range_raises(marker_data):
    with pytest.raises(ValueError):
        marker_data.crop(-1, 5)
    with pytest.raises(ValueError):
        marker_data.crop(0, 10_000)


def test_rotate_x_axis_90_convention():
    # Locked-in characterization of the current rotation convention (verified
    # against ibo_biomech.utils.get_rotation_matrix('x', 90)): a +90deg
    # rotation about X maps (x, y, z) -> (x, -z, y). This is exactly the kind
    # of sign convention a refactor could silently flip, so pin it down
    # explicitly rather than asserting self-consistency.
    m = MarkerData(name="t", x=np.array([1.0]), y=np.array([2.0]), z=np.array([3.0]),
                    sampling_rate=100.0)
    m.rotate(axis="x", angle_deg=90)
    assert m.x[0] == pytest.approx(1.0)
    assert m.y[0] == pytest.approx(-3.0)
    assert m.z[0] == pytest.approx(2.0)


def test_rotate_updates_trajectory_shape(marker_data):
    shape_before = marker_data.get_trajectory().shape
    marker_data.rotate(axis="z", angle_deg=45)
    assert marker_data.get_trajectory().shape == shape_before


def test_lowpass_filter_requires_sampling_rate():
    m = MarkerData(name="t", x=np.random.randn(50), y=np.random.randn(50), z=np.random.randn(50))
    with pytest.raises(ValueError):
        m.lowpass_filter(cutoff=5)


def test_lowpass_filter_applies_to_all_three_axes(marker_data):
    x0, y0, z0 = marker_data.x.copy(), marker_data.y.copy(), marker_data.z.copy()
    marker_data.lowpass_filter(cutoff=2, order=4)
    assert not np.array_equal(marker_data.x, x0)
    assert not np.array_equal(marker_data.y, y0)
    assert not np.array_equal(marker_data.z, z0)


def test_convert_units_mm_to_m_and_back(marker_data):
    x0 = marker_data.x.copy()
    marker_data.convert_units("m")
    assert marker_data.unit == "m"
    assert np.allclose(marker_data.x, x0 * 0.001)
    marker_data.convert_units("m")  # no-op when already target unit
    assert np.allclose(marker_data.x, x0 * 0.001)
    marker_data.convert_units("mm")
    assert np.allclose(marker_data.x, x0)


def test_convert_units_unsupported_raises(marker_data):
    with pytest.raises(ValueError):
        marker_data.convert_units("cm")


def test_add_produces_virtual_marker(marker_data):
    other = MarkerData(name="L_Knee", x=marker_data.x.copy(), y=marker_data.y.copy(),
                        z=marker_data.z.copy(), sampling_rate=marker_data.sampling_rate)
    combined = marker_data + other
    assert combined.virtual == 1
    assert combined.name == f"{marker_data.name}_plus_{other.name}"
    assert np.allclose(combined.x, marker_data.x + other.x)


def test_add_shape_mismatch_raises(marker_data):
    short = MarkerData(name="short", x=np.zeros(5), y=np.zeros(5), z=np.zeros(5))
    with pytest.raises(ValueError):
        marker_data + short


def test_truediv_by_scalar_produces_virtual_marker(marker_data):
    halved = marker_data / 2
    assert halved.virtual == 1
    assert np.allclose(halved.x, marker_data.x / 2)


def test_truediv_by_zero_raises(marker_data):
    with pytest.raises(ZeroDivisionError):
        marker_data / 0


def test_midpoint_pattern_matches_readme_example(marker_data):
    # Exercises the documented "add virtual marker" workflow (R_Knee, L_Knee
    # -> MidKnee via (a + b) / 2) end to end.
    other = MarkerData(name="L_Knee", x=marker_data.x.copy(), y=marker_data.y.copy(),
                        z=marker_data.z.copy(), sampling_rate=marker_data.sampling_rate)
    mid = (marker_data + other) / 2
    assert np.allclose(mid.x, marker_data.x)  # identical inputs -> midpoint == either input
    assert mid.virtual == 1
