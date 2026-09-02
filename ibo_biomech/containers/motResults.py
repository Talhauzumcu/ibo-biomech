"""Shared base for OpenSim .mot-file results containers (ID and IK).

:class:`IDResults` and :class:`IKResults` were previously near-identical
copies of each other (read/write/crop/filter/add_column/__repr__/__len__ were
all duplicated verbatim). This module factors that into :class:`MotResults`;
the two concrete subclASSES each keep only what's genuinely different about
them.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .data import Data
import numpy as np


@dataclass
class MotResults:
    """Base container for a set of named, time-aligned signals read from
    (and written to) an OpenSim ``.mot`` file.

    Attributes:
        name: Name of the results container.
        filepath: If given, read from this path in ``__post_init__``.
        unit: ``'deg'`` or ``'rad'``, parsed from the file's ``inDegrees`` flag.
        data: Mapping of column name to :class:`Data`.
        time: Shared time vector, shape ``(n_samples,)``.
        metadata: Raw header key/value pairs from the .mot file.
    """

    name: str = None
    filepath: str = None
    unit: str = ""
    data: Dict[str, Data] = field(default_factory=dict)
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.filepath is not None:
            self.read(self.filepath)
        self.columns = list(self.data.keys())

    def read(self, filepath: str):
        """Read results from a .mot file and store them in :attr:`data`."""
        from ibo_biomech.utils.utils import read_mot
        mot_data = read_mot(filepath)
        self.time = mot_data['time']
        self.metadata = mot_data.get('metadata', {})
        self.unit = 'deg' if self.metadata.get('inDegrees') == 'yes' else 'rad'
        self.columns = [key for key in mot_data.keys() if (key != 'time' and key != 'metadata')]
        for column in self.columns:
            self.data[column] = Data(name=column,
                                      data=mot_data[column],
                                      unit=self.unit,
                                      time=self.time)

    def highpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase high-pass Butterworth filter to every column.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for data in self.data.values():
            data.highpass_filter(cutoff, order)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> None:
        """Apply a zero-phase low-pass Butterworth filter to every column.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order. Defaults to 4.
        """
        for data in self.data.values():
            data.lowpass_filter(cutoff, order)

    def crop(self, start_idx: int, end_idx: int) -> None:
        """Crop the time vector and every column in place to ``[start_idx, end_idx)``.

        Args:
            start_idx: First sample index to keep.
            end_idx: First sample index to drop (exclusive).
        """
        from ibo_biomech.utils.utils import validate_crop_range
        validate_crop_range(start_idx, end_idx, len(self.time))

        self.time = self.time[start_idx:end_idx]
        for data in self.data.values():
            data.crop(start_idx, end_idx)

    def _extra_header_lines(self) -> List[str]:
        """Hook for subclass-specific .mot header lines.

        Base implementation adds nothing. :class:`IKResults` overrides this
        to add the ``inDegrees=`` line -- that's the one real difference
        between the two writers, everything else in :meth:`write` is shared.
        """
        return []

    def write(self, filepath: str):
        """Write the results to a .mot file."""
        num_columns = len(self.columns) + 1  # +1 for time column
        num_samples = len(self.time)
        with open(filepath, 'w') as f:
            # Header
            f.write("Coordinates\n")
            f.write("version=1\n")
            f.write(f"nRows={num_samples}\n")
            f.write(f"nColumns={num_columns}\n")
            for line in self._extra_header_lines():
                f.write(line)
            f.write("endheader\n")

            # Column headers
            header = "time"
            for column in self.columns:
                header += f"\t{column}"
            f.write(header + "\n")

            # Data rows
            for i in range(num_samples):
                row = f"{self.time[i]}"
                for column in self.columns:
                    row += f"\t{self.data[column].data[i]}"
                f.write(row + "\n")
        return filepath

    def add_column(self, name: str, data: np.ndarray, unit: str = "", time: Optional[np.ndarray] = None):
        """Add a new column.

        Args:
            name: Name of the new column.
            data: Signal samples for the new column, shape ``(n_samples,)``.
            unit: Physical unit of the signal (e.g. ``"V"``).
            time: Time vector corresponding to the data samples, shape ``(n_samples,)``.
        """
        if name in self.data:
            raise ValueError(f"Column '{name}' already exists.")
        if len(data) != len(self.time):
            raise ValueError("Data length must match the length of the time vector.")

        self.data[name] = Data(name=name, data=data, unit=unit, time=time)
        self.columns.append(name)

    def __getitem__(self, column: str) -> Data:
        """Look up a column by name.
        Args:
            column: Column name.

        Returns:
            The matching :class:`Data`.

        Raises:
            KeyError: If ``column`` is not a known column name.
        """
        return self.data[column]

    def __repr__(self) -> str:
        """Return a concise summary of the data."""
        columns_str = "\n".join(self.columns)
        return (
            f"{type(self).__name__}(name={self.name!r}, samples={len(self.time)}, "
            f"unit={self.unit!r}) \n"
            f"Columns: \n{columns_str}"
        )

    def __str__(self) -> str:
        """Return the same concise summary as :meth:`__repr__`."""
        return self.__repr__()

    def __len__(self):
        """Return the number of columns."""
        return len(self.data.keys())

    def __contains__(self, column: str) -> bool:
        """Return True if the given column name exists in the data."""
        return column in self.data