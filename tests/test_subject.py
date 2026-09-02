import numpy as np
import pytest
from ibo_biomech.containers import Subject, TrialData, MarkerData


@pytest.fixture
def subject(marker_kwargs):
    s = Subject(id="S01", condition="control", body_mass=70.0, body_height=1.75, age=30)
    for i in range(3):
        t = TrialData(name=f"trial{i}")
        kw = dict(marker_kwargs)
        kw["name"] = "R_Knee"
        t.add_marker(MarkerData(**kw))
        s.add_trial(f"trial{i}", t)
    return s


def test_add_trial_and_get_by_idx(subject):
    assert len(subject.trials) == 3
    t0 = subject.get_trial_by_idx(0)
    assert t0.name == "trial0"
    t_last = subject.get_trial_by_idx(2)
    assert t_last.name == "trial2"


def test_get_trial_by_idx_out_of_range_returns_none(subject):
    assert subject.get_trial_by_idx(-1) is None
    assert subject.get_trial_by_idx(99) is None


def test_lowpass_filter_skips_bad_trial_without_aborting(subject, capsys):
    # Quirk: Subject.lowpass_filter catches per-trial exceptions and prints
    # rather than raising, so one malformed trial shouldn't stop the batch.
    bad_trial = TrialData(name="bad")
    bad_trial.add_marker(MarkerData(name="NoRate", x=np.zeros(5), y=np.zeros(5), z=np.zeros(5)))
    subject.add_trial("bad", bad_trial)
    subject.lowpass_filter(cutoff_marker=5, cutoff_analog=5, cutoff_force=5)
    captured = capsys.readouterr()
    assert "Error occurred while filtering trial bad" in captured.out
    assert len(subject.trials) == 4  # bad trial is kept, not dropped


def test_save_and_load_cache_roundtrip(subject, tmp_path):
    cache_dir = tmp_path / "cache"
    subject.save_cache(cache_dir=str(cache_dir))
    cache_file = cache_dir / f"{subject.id}_cache.pkl"
    assert cache_file.exists()

    loaded = Subject.load_from_cache(cache_path=str(cache_file))
    assert loaded.id == subject.id
    assert loaded.condition == subject.condition
    assert len(loaded.trials) == len(subject.trials)


def test_load_from_cache_missing_file_returns_none(tmp_path, capsys):
    result = Subject.load_from_cache(cache_path=str(tmp_path / "nope.pkl"))
    assert result is None
    assert "not found" in capsys.readouterr().out
