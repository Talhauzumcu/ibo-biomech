"""Shared validation for clocks and serialized channel collections."""
import numpy as np


def validate_clock(time, size, rate=None, *, uniform=True):
    """Validate a clock without modifying it; return its known/inferred rate."""
    if rate is not None and (not np.isfinite(rate) or rate <= 0):
        raise ValueError('Sampling rate must be finite and positive.')
    if time is None:
        return rate
    time = np.asarray(time)
    if time.shape != (size,) or not np.all(np.isfinite(time)):
        raise ValueError('Time must be a finite vector matching the sample count.')
    if size > 1:
        steps = np.diff(time)
        if np.any(steps <= 0):
            raise ValueError('Time must be strictly increasing.')
        if uniform and not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-10):
            raise ValueError('Processing requires a uniform time vector.')
        if rate is not None and not np.allclose(steps, 1. / rate, rtol=1e-6, atol=1e-10):
            raise ValueError('Time does not match the sampling rate.')
        if rate is None and uniform:
            return float(1. / steps[0])
    return rate


def collection_clock(channels, *, markers=False):
    """Validate equal sample counts, clocks and rates in a shared dataset."""
    first = next(iter(channels.values()))
    size = len(first.x if markers else first.data)
    rate = validate_clock(first.time, size, first.sampling_rate)
    time = first.time
    if time is None and rate is not None:
        time = np.arange(size) / rate
    for name, channel in channels.items():
        values = channel.get_trajectory() if markers else np.asarray(channel.data)
        expected = (3, size) if markers else (size,)
        if values.shape != expected:
            raise ValueError(f'{name}: expected data shape {expected}, got {values.shape}.')
        other_rate = validate_clock(channel.time, size, channel.sampling_rate)
        other_time = channel.time
        if other_time is None and other_rate is not None:
            other_time = np.arange(size) / other_rate
        if (rate is None) != (other_rate is None) or (
                rate is not None and not np.isclose(rate, other_rate)):
            raise ValueError('Channels in one dataset must share a sampling rate.')
        if (time is None) != (other_time is None) or (
                time is not None and not np.allclose(time, other_time, rtol=0, atol=1e-9)):
            raise ValueError('Channels in one dataset must share a clock.')
    return size, time, rate
