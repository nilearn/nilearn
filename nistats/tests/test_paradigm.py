"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

import os

import numpy as np
import pandas as pd


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    events = pd.DataFrame({'name': conditions,
                            'onset': onsets,
                            'duration': durations})
    return events


def modulated_block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 + 5 * np.random.rand(len(onsets))
    values = np.random.rand(len(onsets))
    events = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': durations,
                          'modulation': values})
    return events


def modulated_event_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    values = np.random.rand(len(onsets))
    events = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'durations': durations,
                          'amplitude': values})
    return events


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 * np.ones(9)
    events = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': durations})
    return events


def write_events(events, tmpdir):
    """Function to write events of an experimental paradigm to a file and return the address.
    """
    tsvfile = os.path.join(tmpdir, 'events.tsv')
    events.to_csv(tsvfile, sep='\t')
    return tsvfile


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
