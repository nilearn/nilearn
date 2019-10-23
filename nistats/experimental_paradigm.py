"""
An experimental protocol is handled as a pandas DataFrame
that includes an 'onset' field.

This yields the onset time of the events in the experimental paradigm.
It can also contain:

    * a 'trial_type' field that yields the condition identifier.
    * a 'duration' field that yields event duration (for so-called block
        paradigms).
    * a 'modulation' field that associated a scalar value to each event.

Author: Bertrand Thirion, 2015

"""
from __future__ import with_statement

import warnings

import numpy as np


def check_events(events):
    """Test that the events data describes a valid experimental paradigm

    It is valid if the events data  has an 'onset' key.

    Parameters
    ----------
    events : pandas DataFrame
        Events data that describes a functional experimental paradigm.

    Returns
    -------
    trial_type : array of shape (n_events,), dtype='s'
        Per-event experimental conditions identifier.
        Defaults to np.repeat('dummy', len(onsets)).

    onset : array of shape (n_events,), dtype='f'
        Per-event onset time (in seconds)

    duration : array of shape (n_events,), dtype='f'
        Per-event durantion, (in seconds)
        defaults to zeros(n_events) when no duration is provided

    modulation : array of shape (n_events,), dtype='f'
        Per-event modulation, (in seconds)
        defaults to ones(n_events) when no duration is provided
    """
    if 'onset' not in events.keys():
        raise ValueError('The provided events data has no onset column.')
    if 'duration' not in events.keys():
        raise ValueError('The provided events data has no duration column.')

    onset = np.array(events['onset'])
    duration = np.array(events['duration']).astype(np.float)
    n_events = len(onset)
    trial_type = np.array(events['trial_type'])
    modulation = np.ones(n_events)
    if 'trial_type' not in events.keys():
        warnings.warn("'trial_type' column not found in the given events data.")
        trial_type = np.repeat('dummy', n_events)
    if 'modulation' in events.keys():
        warnings.warn("'modulation' column found in the given events data.")
        modulation = np.array(events['modulation']).astype(np.float)
    return trial_type, onset, duration, modulation
