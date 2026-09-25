"""Named events on a trial's source-frame clock."""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass
class Event:
    """One named event with a description, source frame and timestamp.

    ``frame`` is the nearest zero-based source point frame, on the same scale
    as MarkerData.first_frame, not an index relative to a cropped trajectory.
    ``time`` is in seconds on the trial clock and retains sub-frame precision.
    Given a trajectory's first frame/time, ``frame = first_frame +
    round((time - first_time) * point_sampling_rate)``. For the native C3D
    clock, ``first_time = first_frame / rate``; only then does this simplify
    to ``round(time * rate)``. Subtract first_frame to get an array index.
    Repeated names are allowed in TrialData.events.

    """

    name: str
    frame: int
    time: float
    description: str = ''

    def __post_init__(self):
        self.validate()
        self.frame = int(self.frame)
        self.time = float(self.time)

    def validate(self) -> None:
        """Reject malformed annotations before writing a trial."""
        for field in ('name', 'description'):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f'Event {field} must be a string.')
        if isinstance(self.frame, (bool, np.bool_)) or not isinstance(self.frame, Integral):
            raise ValueError('Event frame must be an integer.')
        if self.frame < 0:
            raise ValueError('Event frame must be nonnegative.')
        if (isinstance(self.time, (bool, np.bool_)) or not isinstance(self.time, Real)
                or not np.isfinite(self.time) or self.time < 0):
            raise ValueError('Event time must be finite and nonnegative.')
