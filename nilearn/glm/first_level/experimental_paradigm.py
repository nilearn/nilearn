"""An experimental protocol is handled as a pandas DataFrame \
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

import pandas as pd
from pandas.api.types import is_numeric_dtype

from nilearn._utils import logger


def check_events(events):
    """Test that the events data describes a valid experimental paradigm.

    It is valid if the events data has ``'onset'`` and ``'duration'`` keys
    with numeric non NaN values.

    Parameters
    ----------
    events : pandas DataFrame
        Events data that describes a functional experimental paradigm.

    Returns
    -------
    events : pandas DataFrame
        Events data that describes a functional experimental paradigm.

    The dataframe has the following columns:

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

    Raises
    ------
    TypeError
        If the events data is not a pandas DataFrame.

    ValueError
        If the events data has:

            - no ``'onset'`` or ``'duration'`` column,
            - has non numeric values
              in the ``'onset'`` or ``'duration'`` columns
            - has nan values in the ``'onset'`` or ``'duration'`` columns.

    Warns
    -----
    UserWarning
        If the events data:

            - has no ``'trial_type'`` column,
            - has any event with a duration equal to 0,
            - contains columns other than ``'onset'``, ``'duration'``,
              ``'trial_type'`` or ``'modulation'``,
            - contains duplicated events, meaning event with same:

                - ``'trial_type'``
                - ``'onset'``
                - ``'duration'``

    """
    # Check that events is a Pandas DataFrame
    if not isinstance(events, pd.DataFrame):
        raise TypeError(
            "Events should be a Pandas DataFrame. "
            f"A {type(events)} was provided instead."
        )

    events = _check_columns(events)

    events_copy = events.copy()

    events_copy = _handle_missing_trial_types(events_copy)

    _check_null_duration(events_copy)

    _check_unexpected_columns(events_copy)

    return _handle_modulation(events_copy)


def _check_columns(events):
    """Check events has onset and duration numeric columns with no NaN."""
    for col_name in ["onset", "duration"]:
        if col_name not in events.columns:
            raise ValueError(
                f"The provided events data has no {col_name} column."
            )
        if events[col_name].isna().any():
            raise ValueError(
                f"The following column must not contain nan values: {col_name}"
            )
        # Make sure we have a numeric type for duration
        if not is_numeric_dtype(events[col_name]):
            try:
                events = events.astype({col_name: float})
            except ValueError as e:
                raise ValueError(
                    f"Could not cast {col_name} to float in events data."
                ) from e
    return events


def _handle_missing_trial_types(events):
    """Create 'dummy' events trial_type if the column is not present."""
    if "trial_type" not in events.columns:
        warnings.warn(
            "'trial_type' column not found in the given events data."
        )
        events["trial_type"] = "dummy"
    return events


def _check_null_duration(events):
    """Warn if there are events with null duration."""
    conditions_with_null_duration = events["trial_type"][
        events["duration"] == 0
    ].unique()
    if len(conditions_with_null_duration) > 0:
        ordered_list = [
            f"- '{x}'\n" for x in sorted(conditions_with_null_duration)
        ]
        ordered_list = "".join(ordered_list)
        warnings.warn(
            (
                "The following conditions contain events with null duration:\n"
                f"{ordered_list}"
            ),
            stacklevel=4,
        )


def _handle_modulation(events):
    """Set the modulation column to 1 if it is not present."""
    if "modulation" in events.columns:
        logger.log(
            "A 'modulation' column was found in "
            "the given events data and is used.",
            stack_level=2,
        )
    else:
        events["modulation"] = 1
    return events


VALID_FIELDS = {"onset", "duration", "trial_type", "modulation"}


def _check_unexpected_columns(events):
    """Warn for each unexpected column that will not be used afterwards."""
    unexpected_columns = list(set(events.columns).difference(VALID_FIELDS))
    if unexpected_columns:
        warnings.warn(
            "The following unexpected columns "
            "in events data will be ignored: "
            f"{', '.join(unexpected_columns)}"
        )


# Two events are duplicates if they have the same:
#   - trial type
#   - onset
#   - duration
COLUMN_DEFINING_EVENT_IDENTITY = ["trial_type", "onset", "duration"]

# Duplicate handling strategy
# Sum the modulation values of duplicate events
STRATEGY = {"modulation": "sum"}


def handle_modulation_of_duplicate_events(events):
    """Deal with modulation of duplicate events if they have one.

    Currently the strategy is to sum the modulation values of duplicate events.
    """
    cleaned_events = (
        events.groupby(COLUMN_DEFINING_EVENT_IDENTITY, sort=False)
        .agg(STRATEGY)
        .reset_index()
    )

    # If there are duplicates, give a warning
    if len(cleaned_events) != len(events):
        warnings.warn(
            "Duplicated events were detected. "
            "Amplitudes of these events will be summed. "
            "You might want to verify your inputs."
        )

    return cleaned_events
