"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

import os
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from nilearn.glm.first_level import check_events


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def duplicate_events_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c0','c1', 'c1']
    onsets = [10, 30, 70, 70, 10, 30]
    durations = [1., 1., 1., 1., 1., 1]
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def modulated_block_paradigm():
    rng = np.random.RandomState(42)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 + 5 * rng.uniform(size=len(onsets))
    values = rng.uniform(size=len(onsets))
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations,
                           'modulation': values})
    return events


def modulated_event_paradigm():
    rng = np.random.RandomState(42)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    values = rng.uniform(size=len(onsets))
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'durations': durations,
                           'amplitude': values})
    return events


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def write_events(events, tmpdir):
    """Function to write events of an experimental paradigm
    to a file and return the address.
    """
    tsvfile = os.path.join(tmpdir, 'events.tsv')
    events.to_csv(tsvfile, sep='\t')
    return tsvfile


def test_check_events():
    """Test the function which tests that the events
    data describes a valid experimental paradigm.
    """
    events = basic_paradigm()
    # Errors checkins
    # Wrong type
    with pytest.raises(TypeError,
                       match="Events should be a Pandas DataFrame."):
        check_events([])
    # Missing onset
    missing_onset = events.drop(columns=['onset'])
    with pytest.raises(ValueError,
                       match='The provided events data has no onset column.'):
        check_events(missing_onset)

    # Missing duration
    missing_duration = events.drop(columns=['duration'])
    with pytest.raises(ValueError,
                       match='The provided events data has no duration column.'):
        check_events(missing_duration)

    # Duration wrong type
    wrong_duration = events.copy()
    wrong_duration['duration'] = 'foo'
    with pytest.raises(ValueError,
                       match="Could not cast duration to float"):
        check_events(wrong_duration)

    # Warnings checkins
    # Missing trial type
    missing_ttype = events.drop(columns=['trial_type'])
    with pytest.warns(UserWarning,
                      match="'trial_type' column not found"):
        ttype, onset, duration, modulation = check_events(missing_ttype)

    # Check that missing trial type yields a 'dummy' array
    assert len(np.unique(ttype)) == 1
    assert ttype[0] == 'dummy'

    ttype, onset, duration, modulation = check_events(events)

    # Check that given trial type is right
    assert_array_equal(ttype,
                       ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2'])

    # Check that missing modulation yields an array one ones
    assert_array_equal(modulation, np.ones(len(events)))

    # Modulation is provided
    events['modulation'] = np.ones(len(events))
    _, _, _, mod = check_events(events)
    assert_array_equal(mod, events['modulation'])

    # An unexpected field is provided
    events = events.drop(columns=['modulation'])
    events['foo'] = np.zeros(len(events))
    with pytest.warns(UserWarning,
                      match="Unexpected column `foo` in events data."):
        ttype2, onset2, duration2, modulation2 = check_events(events)
    assert_array_equal(ttype, ttype2)
    assert_array_equal(onset, onset2)
    assert_array_equal(duration, duration2)
    assert_array_equal(modulation, modulation2)


def test_duplicate_events():
    """Test the function check_events when the paradigm contains
    duplicate events.

    """
    events = duplicate_events_paradigm()
    # Check that a warning is given to the user
    with pytest.warns(UserWarning,
                      match="Duplicated events were detected."):
        ttype, onset, duration, modulation = check_events(events)
    assert_array_equal(ttype, ['c0', 'c0', 'c0', 'c1', 'c1'])
    assert_array_equal(onset, [10, 30, 70, 10, 30])
    assert_array_equal(duration, [1. , 1. , 1., 1. , 1. ])
    # Modulation was updated
    assert_array_equal(modulation, [1, 1, 2, 1, 1])


def test_read_events():
    """ test that a events for an experimental paradigm are correctly read.
    """
    import tempfile
    tmpdir = tempfile.mkdtemp()
    for events in (block_paradigm(),
                   modulated_event_paradigm(),
                   modulated_block_paradigm(),
                   basic_paradigm()):
        csvfile = write_events(events, tmpdir)
        read_paradigm = pd.read_table(csvfile)
        assert (read_paradigm['onset'] == events['onset']).all()
