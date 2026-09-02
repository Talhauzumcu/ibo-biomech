import numpy as np
import pytest
from ibo_biomech.containers import TrialData, MarkerData, ForceData, AnalogData


@pytest.fixture
def trial(marker_kwargs, force_kwargs, analog_kwargs):
    t = TrialData(name="trial01")
    t.add_marker(MarkerData(**marker_kwargs))
    other_marker_kwargs = dict(marker_kwargs)
    other_marker_kwargs["name"] = "L_Knee"
    t.add_marker(MarkerData(**other_marker_kwargs))
    t.add_force(ForceData(**force_kwargs))
    t.add_analog(AnalogData(**analog_kwargs))
    return t


def test_post_init_caches_labels_and_rates_when_passed_to_constructor(marker_kwargs):
    # marker_labels/marker_rate ARE correct when markers are passed directly
    # into the TrialData constructor (this is how both C3DHandler and
    # H5Handler build trials).
    m = MarkerData(**marker_kwargs)
    t = TrialData(name="t", markers={m.name: m})
    assert t.marker_labels == [m.name]
    assert t.marker_rate == marker_kwargs["sampling_rate"]


def test_KNOWN_BUG_add_marker_does_not_refresh_cached_labels(trial):
    # BUG, not a refactor target per se, but worth fixing alongside it:
    # marker_labels/analog_labels/marker_rate/analog_rate are computed once
    # in __post_init__ and never refreshed by add_marker()/add_analog(). The
    # `trial` fixture builds an empty TrialData() then calls add_marker(),
    # exactly like the documented "add a virtual marker" README workflow --
    # so trial.marker_labels stays permanently empty/stale even though
    # trial.markers is populated. This test locks in the CURRENT (buggy)
    # behavior; flip it once add_marker/add_analog are fixed to refresh
    # these caches.
    assert trial.marker_labels == []
    assert trial.marker_rate is None
    assert set(trial.markers.keys()) == {"R_Knee", "L_Knee"}  # actual data is fine


def test_post_init_rates_none_when_empty():
    t = TrialData(name="empty")
    assert t.marker_rate is None
    assert t.analog_rate is None


def test_get_marker_and_get_markers(trial, marker_kwargs):
    m = trial.get_marker(marker_kwargs["name"])
    assert m is not None
    assert m.name == marker_kwargs["name"]
    assert trial.get_marker("nonexistent") is None
    fetched = trial.get_markers([marker_kwargs["name"], "nonexistent"])
    assert fetched[0] is m
    assert fetched[1] is None


def test_get_analog_by_channel(trial, analog_kwargs):
    a = trial.get_analog_by_channel(analog_kwargs["channel"])
    assert a is not None
    assert a.name == analog_kwargs["name"]
    assert trial.get_analog_by_channel(999) is None


def test_rotate_markers_rotates_every_marker(trial):
    before = {name: m.x.copy() for name, m in trial.markers.items()}
    trial.rotate_markers(axis="z", angle_deg=90)
    for name, m in trial.markers.items():
        assert not np.array_equal(m.x, before[name])


def test_convert_units_converts_markers_and_forces(trial):
    trial.convert_units("m")
    for m in trial.markers.values():
        assert m.unit == "m"
    for f in trial.forces.values():
        assert f.unit_cop == "m"


def test_crop_dispatches_by_data_type(trial):
    force_len_before = next(iter(trial.forces.values())).num_samples
    trial.crop("markers", 10, 30)
    for m in trial.markers.values():
        assert len(m.x) == 20
    # forces must be untouched by a 'markers'-only crop call
    for f in trial.forces.values():
        assert f.num_samples == force_len_before


def test_crop_unknown_data_type_raises(trial):
    with pytest.raises(ValueError):
        trial.crop("bogus", 0, 10)


def test_lowpass_filter_markers_analogs_forces_are_independent(trial):
    m_before = {n: m.x.copy() for n, m in trial.markers.items()}
    a_before = {n: a.data.copy() for n, a in trial.analogs.items()}
    trial.lowpass_filter_markers(cutoff_freq=5)
    for n, m in trial.markers.items():
        assert not np.array_equal(m.x, m_before[n])
    # analogs untouched by a markers-only filter call
    for n, a in trial.analogs.items():
        assert np.array_equal(a.data, a_before[n])


def test_parse_emg_data_from_analog_channel(trial, analog_kwargs):
    trial.parse_EMG_data([analog_kwargs["channel"]])
    assert analog_kwargs["name"] in trial.emgs
    emg = trial.emgs[analog_kwargs["name"]]
    assert emg.channel == analog_kwargs["channel"]


def test_add_marker_overwrites_same_name(trial, marker_kwargs):
    n_before = len(trial.markers)
    trial.add_marker(MarkerData(**marker_kwargs))  # same name, should replace not duplicate
    assert len(trial.markers) == n_before


def test_attach_ik_results(trial, mot_file):
    trial.attach_IK_results(mot_file)
    assert trial.ik_results is not None
    assert trial.ik_results.unit == "deg"


def test_attach_id_results(trial, mot_file):
    trial.attach_ID_results(mot_file)
    assert trial.id_results is not None
