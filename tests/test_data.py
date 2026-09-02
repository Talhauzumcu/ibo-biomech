import numpy as np
import pytest
from ibo_biomech.containers import Data


def make_data(n=50, fs=100.0, **overrides):
    kwargs = dict(name="col", data=np.sin(np.arange(n) / fs * 2 * np.pi * 2), unit="deg")
    kwargs.update(overrides)
    return Data(**kwargs)


def test_sampling_rate_derived_from_uniform_time():
    t = np.arange(20) / 50.0
    d = Data(name="x", data=np.zeros(20), time=t)
    assert d.sampling_rate == pytest.approx(50.0)


def test_sampling_rate_none_when_time_nonuniform():
    t = np.array([0.0, 0.1, 0.35, 0.4])
    d = Data(name="x", data=np.zeros(4), time=t)
    assert d.sampling_rate is None


def test_time_length_mismatch_raises():
    with pytest.raises(ValueError):
        Data(name="x", data=np.zeros(5), time=np.zeros(4))


def test_time_generated_from_sampling_rate_when_absent():
    d = Data(name="x", data=np.zeros(10), sampling_rate=20.0)
    assert d.time is not None
    assert len(d.time) == 10
    assert d.time[-1] == pytest.approx(9 / 20.0)


def test_crop_out_of_range_raises():
    d = make_data(n=50)
    with pytest.raises(ValueError):
        d.crop(0, 100)
    with pytest.raises(ValueError):
        d.crop(-1, 10)
    with pytest.raises(ValueError):
        d.crop(10, 10)


def test_crop_trims_data_and_time():
    d = make_data(n=50, sampling_rate=100.0)
    d.crop(10, 20)
    assert len(d.data) == 10
    assert len(d.time) == 10


def test_lowpass_requires_sampling_rate():
    d = Data(name="x", data=np.random.randn(50))
    with pytest.raises(ValueError):
        d.lowpass_filter(cutoff=5)


def test_lowpass_filter_smooths_signal():
    fs = 100.0
    n = 500
    t = np.arange(n) / fs
    noisy = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    d = Data(name="x", data=noisy, sampling_rate=fs)
    d.lowpass_filter(cutoff=5, order=4)
    # high-frequency content should be substantially attenuated
    clean = np.sin(2 * np.pi * 1 * t)
    assert np.corrcoef(d.data, clean)[0, 1] > 0.95


def test_array_and_indexing_protocol():
    d = make_data(n=10)
    assert np.array_equal(np.asarray(d), d.data)
    assert d[0] == d.data[0]
    d[0] = 99.0
    assert d.data[0] == 99.0
    assert len(d) == 10
    assert list(iter(d)) == list(d.data)


def test_arithmetic_ops_with_scalar_and_data():
    d1 = make_data(n=10)
    d2 = make_data(n=10)
    added = d1 + d2
    assert np.array_equal(added.data, d1.data + d2.data)
    added_scalar = d1 + 5
    assert np.array_equal(added_scalar.data, d1.data + 5)
    sub = d1 - 1
    assert np.array_equal(sub.data, d1.data - 1)
    mul = d1 * 2
    assert np.array_equal(mul.data, d1.data * 2)
    div = d1 / 2
    assert np.array_equal(div.data, d1.data / 2)
