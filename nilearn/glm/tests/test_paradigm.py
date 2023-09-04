"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils.data_gen import basic_paradigm
from nilearn.glm.first_level import check_events

from ._utils import (
    _block_paradigm,
    _duplicate_events_paradigm,
    _modulated_block_paradigm,
    _modulated_event_paradigm,
)


def test_check_events():
    events = basic_paradigm()
    trial_type, _, _, modulation = check_events(events)

    # Check that given trial type is right
    assert_array_equal(
        trial_type, ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"]
    )

    # Check that missing modulation yields an array one ones
    assert_array_equal(modulation, np.ones(len(events)))

    # Modulation is provided
    events["modulation"] = np.ones(len(events))
    _, _, _, mod = check_events(events)
    assert_array_equal(mod, events["modulation"])


def test_check_events_errors():
    """Test the function which tests that the events
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
    """Test the function which tests that the events
    data describes a valid experimental paradigm.
    """
    events = basic_paradigm()
    # Warnings checkins
    # Missing trial type
    events = events.drop(columns=["trial_type"])
    with pytest.warns(UserWarning, match="'trial_type' column not found"):
        trial_type, onset, duration, modulation = check_events(events)

    # Check that missing trial type yields a 'dummy' array
    assert len(np.unique(trial_type)) == 1
    assert trial_type[0] == "dummy"

    # An unexpected field is provided
    events["foo"] = np.zeros(len(events))
    with pytest.warns(
        UserWarning, match="Unexpected column 'foo' in events data."
    ):
        trial_type2, onset2, duration2, modulation2 = check_events(events)

    assert_array_equal(trial_type, trial_type2)
    assert_array_equal(onset, onset2)
    assert_array_equal(duration, duration2)
    assert_array_equal(modulation, modulation2)


def test_duplicate_events():
    """Test the function check_events when the paradigm contains
    duplicate events.

    """
    events = _duplicate_events_paradigm()

    # Check that a warning is given to the user
    with pytest.warns(UserWarning, match="Duplicated events were detected."):
        trial_type, onset, duration, modulation = check_events(events)
    assert_array_equal(trial_type, ["c0", "c0", "c0", "c1", "c1"])
    assert_array_equal(onset, [10, 30, 70, 10, 30])
    assert_array_equal(duration, [1.0, 1.0, 1.0, 1.0, 1.0])
    # Modulation was updated
    assert_array_equal(modulation, [1, 1, 2, 1, 1])


def write_events(events, tmpdir):
    """Function to write events of an experimental paradigm
    to a file and return the address.
    """
    tsvfile = os.path.join(tmpdir, "events.tsv")
    events.to_csv(tsvfile, sep="\t")
    return tsvfile


@pytest.mark.parametrize(
    "events",
    [
        _block_paradigm(),
        _modulated_event_paradigm(),
        _modulated_block_paradigm(),
        basic_paradigm(),
    ],
)
def test_read_events(events, tmp_path):
    """Test that a events for an experimental paradigm are correctly read."""
    csvfile = write_events(events, tmp_path)
    read_paradigm = pd.read_table(csvfile)

    assert (read_paradigm["onset"] == events["onset"]).all()
