# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the design_matrix utilities.

Note that the tests just look whether the data produced has correct dimension,
not whether it is exact.
"""

import numpy as np

from ..experimental_paradigm import (EventRelatedParadigm, BlockParadigm,
                                     load_paradigm_from_csv_file)


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm =  EventRelatedParadigm(conditions, onsets)
    return paradigm


def modulated_block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 + 5 * np.random.rand(len(onsets))
    values = np.random.rand(len(onsets))
    paradigm = BlockParadigm(conditions, onsets, duration, values)
    return paradigm


def modulated_event_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    values = np.random.rand(len(onsets))
    paradigm = EventRelatedParadigm(conditions, onsets, values)
    return paradigm


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 * np.ones(9)
    paradigm = BlockParadigm (conditions, onsets, duration)
    return paradigm

def write_paradigm(paradigm, session):
    """Function to write a paradigm to a file and return the address
    """
    import tempfile
    csvfile = tempfile.mkdtemp() + '/paradigm.csv'
    paradigm.write_to_csv(csvfile, session)
    return csvfile

def test_read_paradigm():
    """ test that a paradigm is correctly read
    """
    session = 'sess'
    paradigm = block_paradigm()
    csvfile = write_paradigm(paradigm, session)
    read_paradigm = load_paradigm_from_csv_file(csvfile)[session]
    assert (read_paradigm.onset == paradigm.onset).all()

    paradigm = modulated_event_paradigm()
    csvfile = write_paradigm(paradigm, session)
    read_paradigm = load_paradigm_from_csv_file(csvfile)[session]
    assert (read_paradigm.onset == paradigm.onset).all()

    paradigm = modulated_block_paradigm()
    csvfile = write_paradigm(paradigm, session)
    read_paradigm = load_paradigm_from_csv_file(csvfile)[session]
    assert (read_paradigm.onset == paradigm.onset).all()

    paradigm = basic_paradigm()
    csvfile = write_paradigm(paradigm, session)
    read_paradigm = load_paradigm_from_csv_file(csvfile)[session]
    assert (read_paradigm.onset == paradigm.onset).all()


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
