"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

import numpy as np
import os
import pandas as pd

from nistats.experimental_paradigm import paradigm_from_csv

from nose.tools import assert_true


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm = pd.DataFrame({'name': conditions, 'onset': onsets})
    return paradigm


def modulated_block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 + 5 * np.random.rand(len(onsets))
    values = np.random.rand(len(onsets))
    paradigm = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': duration,
                          'modulation': values})
    return paradigm


def modulated_event_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    values = np.random.rand(len(onsets))
    paradigm = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'amplitude': values})
    return paradigm


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 * np.ones(9)
    paradigm = pd.DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': duration})
    return paradigm


def write_paradigm(paradigm, tmpdir):
    """Function to write a paradigm to a file and return the address
    """
    csvfile = os.path.join(tmpdir, 'paradigm.csv')
    paradigm.to_csv(csvfile)
    return csvfile


def test_read_paradigm():
    """ test that a paradigm is correctly read
    """
    import tempfile
    tmpdir = tempfile.mkdtemp()
    for paradigm in (block_paradigm(),
                     modulated_event_paradigm(),
                     modulated_block_paradigm(),
                     basic_paradigm()):
        csvfile = write_paradigm(paradigm, tmpdir)
        read_paradigm = paradigm_from_csv(csvfile)
        assert_true((read_paradigm['onset'] == paradigm['onset']).all())
