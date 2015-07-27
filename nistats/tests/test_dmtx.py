# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""

from __future__ import with_statement

import numpy as np
import os.path as osp
from ..design_matrix import (
    _convolve_regressors, make_design_matrix,
    _cosine_drift, plot_design_matrix, check_design_matrix)
from pandas import DataFrame
from ..experimental_paradigm import check_paradigm

from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_almost_equal, dec, assert_array_equal

# Set the backend to avoid having DISPLAY problems
from nilearn.plotting import _set_mpl_backend
# Avoid making pyflakes unhappy
_set_mpl_backend
try:
    import matplotlib.pyplot
    # Avoid making pyflakes unhappy
    matplotlib.pyplot
except ImportError:
    have_mpl = False
else:
    have_mpl = True

# load the spm file to test cosine basis
my_path = osp.dirname(osp.abspath(__file__))
full_path_design_matrix_file = osp.join(my_path, 'spm_dmtx.npz')
DESIGN_MATRIX = np.load(full_path_design_matrix_file)


def design_matrix_light(
    frame_times, paradigm=None, hrf_model='canonical',
    drift_model='cosine', period_cut=128, drift_order=1, fir_delays=[0],
    add_regs=None, add_reg_names=None, min_onset=-24, path=None):
    """ Idem make_design_matrix, but only returns the computed matrix
    and associated names """
    dmtx = make_design_matrix(frame_times, paradigm, hrf_model,
    drift_model, period_cut, drift_order, fir_delays,
    add_regs, add_reg_names, min_onset)
    _, matrix, names = check_design_matrix(dmtx)
    return matrix, names


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets})
    return paradigm


def modulated_block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 + 5 * np.random.rand(len(onsets))
    values = 1 + np.random.rand(len(onsets))
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': duration,
                          'modulation': values})
    return paradigm


def modulated_event_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    values = 1 + np.random.rand(len(onsets))
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets,
                          'modulation': values})
    return paradigm


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 * np.ones(9)
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': duration})
    return paradigm


@dec.skipif(not have_mpl)
def test_show_design_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1., 128)
    DM = make_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3)
    ax = plot_design_matrix(DM)
    assert (ax is not None)


def test_cosine_drift():
    # add something so that when the tests are launched
    #from a different directory
    # we still find the file ' 'dctmtx_N_20_order_4.txt' ? 

    spm_drifts = DESIGN_MATRIX['cosbf_dt_1_nt_20_hcut_0p1']
    # np.loadtxt('dctmtx_N_20_order_4.txt')
    tim = np.arange(20)
    P = 10  # period is half the time, gives us an order 4
    nistats_drifts = _cosine_drift(P, tim)
    assert_almost_equal(spm_drifts[:, 1:], nistats_drifts[:, : - 1])
    # nistats_drifts is placing the constant at the end [:, : - 1]


def test_design_matrix0():
    # Test design matrix creation when no paradigm is provided
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)

    _, X, names = check_design_matrix(make_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3))
    assert_equal(len(names), 4)
    assert_almost_equal(X[:, 0], np.linspace(- 0.5, .5, 128))


def test_design_matrix0c():
    # test design matrix creation when regressors are provided manually
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    ax = np.random.randn(128, 4)
    _, X, names = check_design_matrix(make_design_matrix(
                frame_times, drift_model='polynomial',
                drift_order=3, add_regs=ax))
    assert_almost_equal(X[:, 0], ax[:, 0])


def test_design_matrix0d():
    # test design matrix creation when regressors are provided manually
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    ax = np.random.randn(128, 4)
    _, X, names = check_design_matrix(make_design_matrix(
            frame_times, drift_model='polynomial', drift_order=3, add_regs=ax))
    assert_equal(len(names), 8)
    assert_equal(X.shape[1], 8)


def test_design_matrix1():
    # basic test based on basic_paradigm and canonical hrf
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                            drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)


def test_convolve_regressors():
    # tests for convolve_regressors helper function
    conditions = ['c0', 'c1']
    onsets = [20, 40]
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets})
    # names not passed -> default names
    frame_times = np.arange(100)
    f, names = _convolve_regressors(paradigm, 'canonical', frame_times)
    assert_equal(names, ['c0', 'c1'])


def test_design_matrix1b():
    # idem test_design_matrix1, but different test
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)
    assert_equal(X.shape, (128, 7))


def test_design_matrix1c():
    # idem test_design_matrix1, but different test
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)
    assert_true((X[:, - 1] == 1).all())


def test_design_matrix1d():
    # idem test_design_matrix1, but different test
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)
    assert_true((np.isnan(X) == 0).all())


def test_design_matrix2():
    # idem test_design_matrix1 with a different drift term
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                        drift_model='cosine', period_cut=63)
    assert_equal(len(names), 7)  # was 8 with old cosine


def test_design_matrix3():
    # idem test_design_matrix1 with a different drift term
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                        drift_model='blank')
    assert_equal(len(names), 4)


def test_design_matrix4():
    # idem test_design_matrix1 with a different hrf model
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical With Derivative'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 10)


def test_design_matrix5():
    # idem test_design_matrix1 with a block paradigm
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = block_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)


def test_design_matrix6():
    # idem test_design_matrix1 with a block paradigm and the hrf derivative
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = block_paradigm()
    hrf_model = 'Canonical With Derivative'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 10)


def test_design_matrix7():
    # idem test_design_matrix1, but odd paradigm
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    conditions = [0, 0, 0, 1, 1, 1, 3, 3, 3]
    # no condition 'c2'
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets})
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                          drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)


def test_design_matrix8():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)


def test_design_matrix9():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                            drift_model='polynomial', drift_order=3,
                            fir_delays=range(1, 5))
    assert_equal(len(names), 16)


def test_design_matrix10():
    # Check that the first column o FIR design matrix is OK
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.name == 'c0'].astype(np.int)
    assert_true(np.all((X[onset + 1, 0] == 1)))


def test_design_matrix11():
    # check that the second column of the FIR design matrix is OK indeed
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.name == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 3, 2] == 1))


def test_design_matrix12():
    # check that the 11th column of a FIR design matrix is indeed OK
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.name == 'c2'].astype(np.int)
    assert_true(np.all(X[onset + 4, 11] == 1))


def test_design_matrix13():
    # Check that the fir_duration is well taken into account
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                          drift_model='polynomial', drift_order=3,
                          fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.name == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 1, 0] == 1))


def test_design_matrix14():
    # Check that the first column o FIR design matrix is OK after a 1/2
    # time shift
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128) + tr / 2
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.name == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 1, 0] > .9))


def test_design_matrix15():
    # basic test based on basic_paradigm, plus user supplied regressors
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    ax = np.random.randn(128, 4)
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3, add_regs=ax)
    assert_equal(len(names), 11)
    assert_equal(X.shape[1], 11)


def test_design_matrix16():
    # Check that additional regressors are put at the right place
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    ax = np.random.randn(128, 4)
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3, add_regs=ax)
    assert_almost_equal(X[:, 3: 7], ax)


def test_design_matrix17():
    # Test the effect of scaling on the events
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    ct = paradigm.onset[paradigm.name == 'c0'].astype(np.int) + 1
    assert_true((X[ct, 0] > 0).all())


def test_design_matrix18():
    # Test the effect of scaling on the blocks
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_block_paradigm()
    hrf_model = 'Canonical'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    ct = paradigm.onset[paradigm.name == 'c0'].astype(np.int) + 3
    assert_true((X[ct, 0] > 0).all())


def test_design_matrix19():
    # Test the effect of scaling on a FIR model
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, paradigm, hrf_model=hrf_model,
                            drift_model='polynomial', drift_order=3,
                            fir_delays=range(1, 5))
    idx = paradigm.onset[paradigm.name == 0].astype(np.int)
    assert_array_equal(X[idx + 1, 0], X[idx + 2, 1])


def test_design_matrix20():
    # Test for commit 10662f7
    frame_times = np.arange(0, 128)  # was 127 in old version of _cosine_drift
    paradigm = modulated_event_paradigm()
    X, names = design_matrix_light(
        frame_times, paradigm, hrf_model='canonical', drift_model='cosine')

    # check that the drifts are not constant
    assert_true(np.all(np.diff(X[:, -2]) != 0))


def test_fir_block():
    # tets FIR models on block designs
    bp = block_paradigm()
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    X, names = design_matrix_light(
        frame_times, bp, hrf_model='fir', drift_model='blank',
        fir_delays=range(0, 4))
    idx = bp['onset'][bp['name'] == 1].astype(np.int)
    assert_equal(X.shape, (128, 13))
    assert_true((X[idx, 4] == 1).all())
    assert_true((X[idx + 1, 5] == 1).all())
    assert_true((X[idx + 2, 6] == 1).all())
    assert_true((X[idx + 3, 7] == 1).all())


def test_csv_io():
    # test the csv io on design matrices
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    DM = make_design_matrix(frame_times, paradigm, hrf_model='Canonical',
                   drift_model='polynomial', drift_order=3)
    path = 'design_matrix.csv'
    with InTemporaryDirectory():
        DM.to_csv(path)
        DM2 = DataFrame().from_csv(path)

    _, matrix, names = check_design_matrix(DM)
    _, matrix_, names_ = check_design_matrix(DM2)
    assert_almost_equal(matrix, matrix_)
    assert_equal(names, names_)


def test_spm_1():
    # Check that the nistats design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frame_times = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets})
    X1 = make_design_matrix(frame_times, paradigm, drift_model='blank')
    _, matrix, _ = check_design_matrix(X1)
    spm_design_matrix = DESIGN_MATRIX['arr_0']
    assert_true(((spm_design_matrix - matrix) ** 2).sum() /
                (spm_design_matrix ** 2).sum() < .1)


def test_spm_2():
    # Check that the nistats design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frame_times = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    duration = 10 * np.ones(9)
    paradigm = DataFrame({'name': conditions,
                          'onset': onsets,
                          'duration': duration})
    X1 = make_design_matrix(frame_times, paradigm, drift_model='blank')
    spm_design_matrix = DESIGN_MATRIX['arr_1']
    _, matrix, _ = check_design_matrix(X1)
    assert_true(((spm_design_matrix - matrix) ** 2).sum() /
                (spm_design_matrix ** 2).sum() < .1)
