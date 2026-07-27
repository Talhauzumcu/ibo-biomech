"""Subject-level data container.

This module defines :class:`Subject`, which groups all trials of a single
participant together with descriptive data and a simple pickle-based cache.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .forceData import ForceData
from .markerData import MarkerData
from .analogData import AnalogData
from .trialData import TrialData
from pathlib import Path
import pickle

@dataclass
class Subject:
    """A participant, their descriptive data and all of their trials.

    Attributes:
        id: Subject identifier.
        condition: Experimental condition label.
        body_mass: Body mass in kilograms.
        body_height: Body height (units as recorded).
        age: Age in years.
        trials: Mapping of trial name to :class:`TrialData`.
        trials_loaded: Whether trial data has been populated.
    """
    id: str = ""
    condition: str = ""
    body_mass: float = None
    body_height: float = None
    age: int = None
    trials: Dict[str, TrialData] = field(default_factory=dict)
    trials_loaded: bool = False

    def add_trial(self, trial_name: str, trial_data: TrialData) -> None:
        """Add (or replace) a trial keyed by name.

        Args:
            trial_name: Name to store the trial under.
            trial_data: The trial to store.
        """
        self.trials[trial_name] = trial_data

    def get_trial_by_idx(self, idx: int) -> Optional[TrialData]:
        """Return a trial by its positional index in the trials mapping.

        Args:
            idx: Zero-based index into the insertion-ordered trials.

        Returns:
            The matching :class:`TrialData`, or ``None`` if ``idx`` is out of
            range.
        """
        if idx < 0 or idx >= len(self.trials):
            return None
        trial_name = list(self.trials.keys())[idx]
        return self.trials[trial_name]

    def lowpass_filter(self, cutoff_marker: float, cutoff_analog: float, cutoff_force: float, order: int = 4):
        """Low-pass filter every trial with the given per-type cutoffs.

        Errors raised while filtering a trial are caught and printed so that one
        bad trial does not abort the whole subject.

        Args:
            cutoff_marker: Cutoff for markers in Hz.
            cutoff_analog: Cutoff for analog channels in Hz.
            cutoff_force: Cutoff for force plates in Hz.
            order: Filter order. Defaults to 4.
        """
        for trial in self.trials.values():
            try:
                trial.lowpass_filter(cutoff_marker=cutoff_marker, cutoff_analog=cutoff_analog, cutoff_force=cutoff_force, order=order)
            except Exception as e:
                print(f"Error occurred while filtering trial {trial.name}: {e}")

    def save_cache(self, cache_dir: str='subject_cache'):
        """Pickle the subject to ``<cache_dir>/<id>_cache.pkl``.

        The directory is created if it does not exist.

        Args:
            cache_dir: Directory to write the cache file into. Defaults to
                ``"subject_cache"``.
        """
        cache_path = Path(cache_dir) / f'{self.id}_cache.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_cache(cache_path: str = 'subject_cache') -> Optional["Subject"]:
        """Load a subject previously written with :meth:`save_cache`.

        Args:
            cache_path: Path to the pickled cache file.

        Returns:
            The restored :class:`Subject`, or ``None`` if the file does not
            exist.
        """
        subject = Subject()  # Create an empty subject instance to load data into
        cache_path = Path(cache_path)
        if not cache_path.exists():
            print(f"Cache file {cache_path} not found. Returning empty Subject instance.")
            return None
        with open(cache_path, 'rb') as f:
            cached_subject = pickle.load(f)
            for attr in vars(cached_subject):
                setattr(subject, attr, getattr(cached_subject, attr))

        return subject

    def __str__(self):
        """Return a short summary of the subject."""
        return (
            f"Subject(id={self.id!r}, condition={self.condition!r}, "
            f"trials={len(self.trials)}, trials_loaded={self.trials_loaded})"
        )

    def __repr__(self):
        """Return a short summary of the subject."""
        return self.__str__()
