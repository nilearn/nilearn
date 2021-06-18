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
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

VALID_FIELDS = set(["onset",
                    "duration",
                    "trial_type",
                    "modulation",
                    ])

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
        defaults to ones(n_events) when no duration is provided.

    """
    # Check that events is a Pandas DataFrame
    if not isinstance(events, pd.DataFrame):
        raise TypeError("Events should be a Pandas DataFrame. "
                        "A {} was provided instead.".format(
                            type(events)))
    # Column checks
    for col_name in ['onset', 'duration']:
        if col_name not in events.columns:
            raise ValueError("The provided events data "
                             "has no {} column.".format(
                                 col_name))

    # Make a copy of the dataframe
    events_copy = events.copy()

    # Handle missing trial types
    if 'trial_type' not in events_copy.columns:
        warnings.warn("'trial_type' column not found "
                      "in the given events data.")
        events_copy['trial_type'] = 'dummy'

    # Handle modulation
    if 'modulation' in events_copy.columns:
        print("A 'modulation' column was found in "
              "the given events data and is used.")
    else:
        events_copy['modulation'] = 1

    # Warn for each unexpected column that will
    # not be used afterwards
    unexpected_columns = set(events_copy.columns).difference(VALID_FIELDS)
    for unexpected_column in unexpected_columns:
        warnings.warn(("Unexpected column `{}` in events "
                       "data will be ignored.").format(
                            unexpected_column))

    # Make sure we have a numeric type for duration
    if not is_numeric_dtype(events_copy['duration']):
        try:
            events_copy = events_copy.astype({'duration': float})
        except:
            raise ValueError("Could not cast duration to float "
                             "in events data.")

    # Handle duplicate events
    # Two events are duplicates if they have the same:
    #   - trial type
    #   - onset
    COLUMN_DEFINING_EVENT_IDENTITY = ['trial_type',
                                      'onset',
                                      'duration',]

    # Duplicate handling strategy
    STRATEGY = {'modulation': np.sum, # Sum the modulation values of duplicate events
                }

    cleaned_events = events_copy.groupby(
                        COLUMN_DEFINING_EVENT_IDENTITY,
                        sort=False).agg(STRATEGY).reset_index()

    # If there are duplicates, give a warning
    if len(cleaned_events) != len(events_copy):
        warnings.warn("Duplicated events were detected. "
                      "Amplitudes of these events will be summed. "
                      "You might want to verify your inputs.")

    trial_type = cleaned_events['trial_type'].values
    onset = cleaned_events['onset'].values
    duration = cleaned_events['duration'].values
    modulation = cleaned_events['modulation'].values
    return trial_type, onset, duration, modulation

