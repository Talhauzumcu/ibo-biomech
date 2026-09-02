"""Shared mixins for container classes."""
import numpy as np


class ArrayLikeMixin:
    """Provides ``__array__``/``__getitem__``/``__setitem__``/``__len__``/``__iter__``
    for any container that exposes its underlying samples via ``self.data``.

    This was previously copy-pasted verbatim across :class:`Data`,
    :class:`ForceData`, :class:`AnalogData` and :class:`EMGData`. It assumes
    nothing about ``self.data`` beyond it being indexable/iterable/sized
    (works whether ``data`` is a plain attribute, as in ``AnalogData``, or a
    computed ``@property``, as in ``ForceData``).
    """

    def __array__(self):
        """Allow the object to be converted to a NumPy array."""
        return self.data

    def __getitem__(self, index):
        """Allow indexing into the data."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Allow setting values in the data."""
        self.data[index] = value

    def __len__(self):
        """Return the number of samples in the data."""
        return len(self.data)

    def __iter__(self):
        """Allow iteration over the data."""
        return iter(self.data)