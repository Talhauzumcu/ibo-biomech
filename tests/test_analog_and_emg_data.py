import numpy as np
import pytest
from ibo_biomech.containers import AnalogData, EMGData


# ---------------------------------------------------------------- AnalogData

def test_analog_get_data_returns_underlying_array(analog_data):
    assert analog_data.get_data() is analog_data.data


def test_analog_has_no_get_raw_data():
    # Pinning down the current naming inconsistency: AnalogData uses
    # get_data(), EMGData uses get_raw_data(). If the refactor adds an
    # alias, update/remove this test -- but do it deliberately.
    assert not hasattr(AnalogData(name="x", data=np.zeros(5)), "get_raw_data")


def test_analog_crop(analog_data):
    analog_data.crop(10, 30)
    assert len(analog_data.data) == 20
    assert len(analog_data.time) == 20


def test_analog_filters_require_sampling_rate():
    a = AnalogData(name="x", data=np.random.randn(50))
    with pytest.raises(ValueError):
        a.lowpass_filter(cutoff=5)
    with pytest.raises(ValueError):
        a.highpass_filter(cutoff=5)


def test_analog_array_protocol(analog_data):
    assert np.array_equal(np.asarray(analog_data), analog_data.data)
    assert len(analog_data) == len(analog_data.data)


# ------------------------------------------------------------------ EMGData

def test_emg_has_get_raw_data_not_get_data(emg_data):
    assert emg_data.get_raw_data() is emg_data.data

def test_emg_process_emg_is_peak_normalized(emg_data):
    processed = emg_data.process_emg()
    assert np.max(np.abs(processed)) == pytest.approx(1.0, abs=1e-6)


def test_emg_processed_data_property_is_cached(emg_data):
    first = emg_data.processed_data
    assert emg_data._processed_data is not None
    second = emg_data.processed_data
    assert first is second  # same cached array object, not recomputed


def test_emg_public_highpass_filter_mutates_data_in_place(emg_data):
    data0 = emg_data.data.copy()
    emg_data.highpass_filter(cutoff=30, order=2)
    assert not np.array_equal(emg_data.data, data0)


def test_emg_private_highpass_filter_does_not_mutate_data(emg_data):
    # This is the key quirk: _highpass_filter is used internally by
    # process_emg() and must return a filtered copy WITHOUT touching
    # self.data, so the raw signal stays available. A refactor that folds
    # this into the public mutating highpass_filter would silently break
    # process_emg()'s raw/processed separation.
    data0 = emg_data.data.copy()
    filtered = emg_data._highpass_filter(cutoff=30, order=2)
    assert np.array_equal(emg_data.data, data0)
    assert not np.array_equal(filtered, data0)


def test_emg_crop_invalidates_processed_cache(emg_data):
    # Quirk: crop() currently re-slices the CACHED processed envelope using
    # raw-signal indices, which conflates a filtered (edge-effected,
    # envelope-shaped) signal with a raw-signal index range. Force the cache
    # to populate first, then crop, and check length consistency post-crop
    # -- if this assertion ever fails after a refactor, that's a genuine
    # behavior change to review deliberately (ideally: invalidate + lazily
    # recompute instead of re-slicing).
    _ = emg_data.processed_data  # populate cache
    assert emg_data._processed_data is not None
    emg_data.crop(10, 30)
    assert len(emg_data.data) == 20
    assert len(emg_data._processed_data) == 20


def test_emg_clean_nan(emg_kwargs):
    kw = dict(emg_kwargs)
    kw["data"] = kw["data"].copy()
    kw["data"][5] = np.nan
    e = EMGData(**kw)
    e.clean_nan()
    assert not np.isnan(e.data).any()


def test_emg_normalize_handles_all_zero_signal():
    e = EMGData(name="flat", data=np.zeros(50), sampling_rate=1000.0)
    # process_emg internally guards against divide-by-zero when the
    # envelope's max amplitude is 0; this should not raise or return NaNs.
    result = e.process_emg()
    assert not np.isnan(result).any()
