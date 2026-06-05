"""Smoke tests — verify the package imports and basic container construction work."""
import numpy as np
import pytest

from ibo_biomech import C3DHandler, H5Handler, FileConverter
from ibo_biomech import AnalogData, ForceData, MarkerData, TrialData, Subject


def test_package_imports():
    assert C3DHandler is not None
    assert H5Handler is not None
    assert FileConverter is not None
    assert MarkerData is not None


def test_marker_data_construction():
    x = np.array([1.0, 2.0, 3.0])
    marker = MarkerData(name="TEST", x=x, y=x.copy(), z=x.copy(), sampling_rate=100.0)
    assert marker.name == "TEST"
    assert len(marker.x) == 3


def test_analog_data_construction():
    data = np.zeros(300)
    analog = AnalogData(name="EMG1", data=data, sampling_rate=1000.0, channel=0)
    assert analog.name == "EMG1"
    assert analog.sampling_rate == 1000.0


def test_trial_data_construction():
    x = np.ones(100)
    marker = MarkerData(name="M1", x=x, y=x, z=x, sampling_rate=100.0)
    trial = TrialData(trial_name="test_trial", markers={"M1": marker})
    assert trial.trial_name == "test_trial"
    assert "M1" in trial.markers


def test_subject_construction():
    subject = Subject(id="P01", condition="walking", body_mass=70.0)
    assert subject.id == "P01"


def test_c3d_handler_file_not_found():
    handler = C3DHandler("nonexistent.c3d")
    with pytest.raises(FileNotFoundError):
        handler.load_data()
