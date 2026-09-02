"""
Shared fixtures for the ibo-biomech characterization suite.

These fixtures build small, deterministic synthetic signals rather than
relying on real .c3d/.h5 files, so the suite runs anywhere without fixture
data. Handler-level tests (C3DHandler, H5Handler, osimHandler) are NOT
covered here -- they need real sample files from example_data/ and should
live in a separate module once those are available in this sandbox.
"""
import numpy as np
import pytest


SAMPLING_RATE = 100.0
N_SAMPLES = 200


def _sine(freq_hz, n=N_SAMPLES, fs=SAMPLING_RATE, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


@pytest.fixture
def time_vector():
    return np.arange(N_SAMPLES) / SAMPLING_RATE


@pytest.fixture
def marker_kwargs():
    """Plain kwargs (not the object) so tests can construct fresh instances."""
    return dict(
        name="R_Knee",
        x=_sine(1.0, amp=50.0) + 1000.0,
        y=_sine(1.3, amp=30.0) + 500.0,
        z=_sine(0.7, amp=20.0) + 200.0,
        unit="mm",
        sampling_rate=SAMPLING_RATE,
    )


@pytest.fixture
def marker_data(marker_kwargs):
    from ibo_biomech.containers import MarkerData
    return MarkerData(**marker_kwargs)


@pytest.fixture
def force_kwargs():
    n = N_SAMPLES
    force = np.vstack([_sine(1.0, amp=5.0), _sine(1.2, amp=3.0), _sine(0.9, amp=700.0) + 700.0])
    moment = np.vstack([_sine(0.5, amp=2.0), _sine(0.6, amp=2.0), _sine(0.4, amp=1.0)])
    cop = np.vstack([_sine(1.0, amp=10.0) + 100.0, _sine(1.0, amp=10.0) + 200.0, np.zeros(n)])
    return dict(
        name="forceplate_0",
        force=force,
        moment=moment,
        cop=cop,
        location=np.zeros((3, 4, n)),
        position=np.zeros((3, n)),
        rotation=np.zeros((3, 3, n)),
        offset=np.zeros((3, 1)),
        Tz=np.zeros(n),
        coordinateSystem=0,
        metadata={"unit_force": "N", "unit_moment": "Nmm", "unit_position": "mm"},
        sampling_rate=SAMPLING_RATE,
    )


@pytest.fixture
def force_data(force_kwargs):
    from ibo_biomech.containers import ForceData
    return ForceData(**force_kwargs)


@pytest.fixture
def analog_kwargs():
    return dict(
        name="EMG_VastusLat",
        data=_sine(20.0, amp=0.001) + np.random.RandomState(0).normal(0, 0.0001, N_SAMPLES),
        sampling_rate=1000.0,  # analog/EMG channels are typically sampled much faster
        unit="V",
        channel=3,
    )


@pytest.fixture
def analog_data(analog_kwargs):
    from ibo_biomech.containers import AnalogData
    return AnalogData(**analog_kwargs)


@pytest.fixture
def emg_kwargs(analog_kwargs):
    kw = dict(analog_kwargs)
    kw["name"] = "EMG_VastusLat"
    return kw


@pytest.fixture
def emg_data(emg_kwargs):
    from ibo_biomech.containers import EMGData
    return EMGData(**emg_kwargs)


MOT_TEXT = """Coordinates
version=1
nRows=5
nColumns=3
inDegrees=yes
endheader
time\tpelvis_tilt\thip_flexion_r
0.00\t1.0\t10.0
0.01\t1.1\t10.5
0.02\t1.2\t11.0
0.03\t1.3\t11.5
0.04\t1.4\t12.0
"""


@pytest.fixture
def mot_file(tmp_path):
    p = tmp_path / "sample.mot"
    p.write_text(MOT_TEXT)
    return str(p)
