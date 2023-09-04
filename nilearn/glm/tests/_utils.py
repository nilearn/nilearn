import numpy as np
import pandas as pd

from nilearn.conftest import _rng


def _conditions():
    return ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"]


def _onsets():
    return [30, 70, 100, 10, 30, 90, 30, 40, 60]


def _durations():
    return np.ones(len(_onsets()))


def _modulated_event_paradigm():
    events = pd.DataFrame(
        {
            "trial_type": _conditions(),
            "onset": _onsets(),
            "duration": _durations(),
            "modulation": _rng().uniform(size=len(_onsets())),
        }
    )
    return events


def _block_paradigm():
    events = pd.DataFrame(
        {
            "trial_type": _conditions(),
            "onset": _onsets(),
            "duration": 5 * _durations(),
        }
    )
    return events


def _modulated_block_paradigm():
    durations = 5 + 5 * _rng().uniform(size=len(_onsets()))
    modulation = 1 + _rng().uniform(size=len(_onsets()))
    events = pd.DataFrame(
        {
            "trial_type": _conditions(),
            "onset": _onsets(),
            "duration": durations,
            "modulation": modulation,
        }
    )
    return events


def _spm_paradigm(block_duration):
    frame_times = np.linspace(0, 99, 100)
    conditions = ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"]
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    durations = block_duration * np.ones(len(onsets))
    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": durations}
    )
    return events, frame_times


def _design_with_null_duration():
    durations = _durations()
    durations[2] = 0
    durations[5] = 0
    durations[8] = 0
    events = pd.DataFrame(
        {
            "trial_type": _conditions(),
            "onset": _onsets(),
            "duration": durations,
        }
    )
    return events


def _design_with_nan_duration():
    durations = _durations()
    durations[2] = np.nan
    durations[5] = np.nan
    durations[8] = np.nan
    events = pd.DataFrame(
        {
            "trial_type": _conditions(),
            "onset": _onsets(),
            "duration": durations,
        }
    )
    return events


def _duplicate_events_paradigm():
    conditions = ["c0", "c0", "c0", "c0", "c1", "c1"]
    onsets = [10, 30, 70, 70, 10, 30]
    durations = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": durations}
    )
    return events
