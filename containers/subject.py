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
    "Container class for a subject, containing all trials and subject descriptive data"
    id: str = ""
    condition: str = ""
    body_mass: float = None
    body_height: float = None
    age: int = None
    trials = Dict[str, TrialData] = field(default_factory=dict)
    trials_loaded: bool = False
    
    def add_trial(self, trial_name: str, trial_data: TrialData) -> None:
        """Add a trial to the subject's trials dictionary."""
        self.trials[trial_name] = trial_data
    
    def get_trial_by_idx(self, idx: int) -> Optional[TrialData]:
        """Get a trial by its index in the trials dictionary."""
        if idx < 0 or idx >= len(self.trials):
            return None
        trial_name = list(self.trials.keys())[idx]
        return self.trials[trial_name]

    def lowpass_filter(self, cutoff_marker: float, cutoff_analog: float, cutoff_force: float, order: int = 4):
        """Apply lowpass filter to all trials."""
        for trial in self.trials.values():
            try:
                trial.lowpass_filter(cutoff_marker=cutoff_marker, cutoff_analog=cutoff_analog, cutoff_force=cutoff_force, order=order)
            except Exception as e:
                print(f"Error occurred while filtering trial {trial.name}: {e}")

    def save_cache(self, cache_dir: str='subject_cache'):
        """Save the subject's data to a cache file."""
        cache_path = Path(cache_dir) / f'{self.id}_cache.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_cache(cache_path='subject_cache'):
        """Load the subject's data from a cache file.
        Returns:
            Subject: The subject instance with data loaded from cache, or None if cache file not found
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
        return f"Subject(id={self.id}, trials={list(self.trials.keys())})"
    
    def __repr__(self):
        return f"Subject(id={self.id}, trials={list(self.trials.keys())})"