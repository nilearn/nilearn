"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils.data_gen import basic_paradigm
from nilearn.glm.first_level.experimental_paradigm import (
    check_events,
    handle_modulation_of_duplicate_events,
)

from ._testing import (
    block_paradigm,
    design_with_nan_durations,
    design_with_nan_onsets,
    design_with_null_durations,
    duplicate_events_paradigm,
    modulated_block_paradigm,
    modulated_event_paradigm,
)


def test_check_events():
    events = basic_paradigm()
    events_copy = check_events(events)

    # Check that given trial type is right
    assert_array_equal(
        events_copy["trial_type"],
        ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"],
    )

    # Check that missing modulation yields an array one ones
    assert_array_equal(events_copy["modulation"], np.ones(len(events)))

    # Modulation is provided
    events["modulation"] = np.ones(len(events))
    events_copy = check_events(events)
    assert_array_equal(events_copy["modulation"], events["modulation"])


def test_check_events_errors():
    """Test the function which tests that the events \
       data describes a valid experimental paradigm.
    """
    events = basic_paradigm()
    # Errors checkins
    # Wrong type
    with pytest.raises(
        TypeError, match="Events should be a Pandas DataFrame."
    ):
        check_events([])

    # Missing onset
    missing_onset = events.drop(columns=["onset"])
    with pytest.raises(
        ValueError, match="The provided events data has no onset column."
    ):
        check_events(missing_onset)

    # Missing duration
    missing_duration = events.drop(columns=["duration"])
    with pytest.raises(
        ValueError, match="The provided events data has no duration column."
    ):
        check_events(missing_duration)

    # Duration wrong type
    wrong_duration = events.copy()
    wrong_duration["duration"] = "foo"
    with pytest.raises(ValueError, match="Could not cast duration to float"):
        check_events(wrong_duration)


def test_check_events_warnings():
    """Test the function which tests that the events \
       data describes a valid experimental paradigm.
    """
    events = basic_paradigm()
    # Warnings checkins
    # Missing trial type
    events = events.drop(columns=["trial_type"])
    with pytest.warns(UserWarning, match="'trial_type' column not found"):
        events_copy = check_events(events)

    # Check that missing trial type yields a 'dummy' array
    assert len(np.unique(events_copy["trial_type"])) == 1
    assert events_copy["trial_type"][0] == "dummy"

    # An unexpected field is provided
    events["foo"] = np.zeros(len(events))
    with pytest.warns(
        UserWarning,
        match=(
            "The following unexpected columns "
            "in events data will be ignored: foo"
        ),
    ):
        events_copy2 = check_events(events)

    assert_array_equal(events_copy["trial_type"], events_copy2["trial_type"])
    assert_array_equal(events_copy["onset"], events_copy2["onset"])
    assert_array_equal(events_copy["duration"], events_copy2["duration"])
    assert_array_equal(events_copy["modulation"], events_copy2["modulation"])


def write_events(events, tmpdir):
    """Write events of an experimental paradigm \
       to a file and return the address.
    """
    tsvfile = Path(tmpdir, "events.tsv")
    events.to_csv(tsvfile, sep="\t")
    return tsvfile


@pytest.mark.parametrize(
    "events",
    [
        block_paradigm(),
        modulated_event_paradigm(),
        modulated_block_paradigm(),
        basic_paradigm(),
    ],
)
def test_read_events(events, tmp_path):
    """Test that a events for an experimental paradigm are correctly read."""
    csvfile = write_events(events, tmp_path)
    read_paradigm = pd.read_table(csvfile)

    assert (read_paradigm["onset"] == events["onset"]).all()


def test_check_events_warnings_null_duration():
    """Test that events with null duration throw a warning."""
    with pytest.warns(
        UserWarning,
        match="The following conditions contain events with null duration",
    ):
        check_events(design_with_null_durations())


@pytest.mark.parametrize(
    "design",
    [
        design_with_nan_durations,
        design_with_nan_onsets,
    ],
)
def test_check_events_nan_designs(design):
    """Test that events with nan values."""
    with pytest.raises(
        ValueError, match=("The following column must not contain nan values:")
    ):
        check_events(design())


def test_sum_modulation_of_duplicate_events():
    """Test the function check_events \
       when the paradigm contains duplicate events.
    """
    events = duplicate_events_paradigm()

    # Check that a warning is given to the user
    with pytest.warns(UserWarning, match="Duplicated events were detected."):
        events_copy = handle_modulation_of_duplicate_events(events)
    assert_array_equal(
        events_copy["trial_type"], ["c0", "c0", "c0", "c1", "c1"]
    )
    assert_array_equal(events_copy["onset"], [10, 30, 70, 10, 30])
    assert_array_equal(events_copy["duration"], [1.0, 1.0, 1.0, 1.0, 1.0])
    # Modulation was updated
    assert_array_equal(events_copy["modulation"], [1, 1, 2, 1, 1])
