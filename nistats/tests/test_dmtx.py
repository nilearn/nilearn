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
#from os.path import join, dirname, walk
from ..experimental_paradigm import (EventRelatedParadigm, BlockParadigm)
from ..design_matrix import (dmtx_light, _convolve_regressors, dmtx_from_csv,
                             make_dmtx, _cosine_drift)

from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_almost_equal, dec, assert_array_equal

try:
    import matplotlib.pyplot
except ImportError:
    have_mpl = False
else:
    have_mpl = True

# load the spm file to test cosine basis
my_path = osp.dirname(osp.abspath(__file__))
full_path_dmtx_file = osp.join(my_path, 'spm_dmtx.npz')
DMTX = np.load(full_path_dmtx_file)

def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm =  EventRelatedParadigm(conditions, onsets)
    return paradigm

def modulated_block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 + 5 * np.random.rand(len(onsets))
    values = 1 + np.random.rand(len(onsets))
    paradigm = BlockParadigm(conditions, onsets, duration, values)
    return paradigm


def modulated_event_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    values = 1 + np.random.rand(len(onsets))
    paradigm = EventRelatedParadigm(conditions, onsets, values)
    return paradigm


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    duration = 5 * np.ones(9)
    paradigm = BlockParadigm (conditions, onsets, duration)
    return paradigm


@dec.skipif(not have_mpl)
def test_show_dmtx():
    # test that the show code indeed (formally) runs
    frametimes = np.linspace(0, 127 * 1.,128)
    DM = make_dmtx(frametimes, drift_model='polynomial', drift_order=3)
    ax = DM.show()
    assert (ax is not None)

def test_cosine_drift():
    # add something so that when the tests are launched from a different directory
    # we still find the file ' 'dctmtx_N_20_order_4.txt' ? 

    spm_drifts = DMTX['cosbf_dt_1_nt_20_hcut_0p1'] # np.loadtxt('dctmtx_N_20_order_4.txt')
    tim = np.arange(20)
    P = 10 # period is half the time, gives us an order 4
    nipy_drifts = _cosine_drift(P, tim) #
    assert_almost_equal(spm_drifts[:,1:], nipy_drifts[:,:-1])
        # nipy_drifts is placing the constant at the end [:,:-1]

def test_dmtx0():
    # Test design matrix creation when no paradigm is provided
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr,128)
    X, names= dmtx_light(frametimes, drift_model='polynomial',
                            drift_order=3)
    assert_equal(len(names), 4)


def test_dmtx0b():
    # Test design matrix creation when no paradigm is provided
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr,128)
    X, names= dmtx_light(frametimes, drift_model='polynomial',
                            drift_order=3)
    assert_almost_equal(X[:, 0], np.linspace(- 0.5, .5, 128))


def test_dmtx0c():
    # test design matrix creation when regressors are provided manually
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    ax = np.random.randn(128, 4)
    X, names= dmtx_light(frametimes, drift_model='polynomial',
                            drift_order=3, add_regs=ax)
    assert_almost_equal(X[:, 0], ax[:, 0])


def test_dmtx0d():
    # test design matrix creation when regressors are provided manually
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    ax = np.random.randn(128, 4)
    X, names= dmtx_light(frametimes, drift_model='polynomial',
                            drift_order=3, add_regs=ax)
    assert_equal(len(names), 8)
    assert_equal(X.shape[1], 8)

    
def test_dmtx1():
    # basic test based on basic_paradigm and canonical hrf
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                            drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)


def test_convolve_regressors():
    # tests for convolve_regressors helper function
    conditions = ['c0', 'c1']
    onsets = [20, 40]
    paradigm =  EventRelatedParadigm(conditions, onsets)
    # names not passed -> default names
    frametimes = np.arange(100)
    f, names = _convolve_regressors(paradigm, 'canonical', frametimes)
    assert_equal(names, ['c0', 'c1'])


def test_dmtx1b():
    # idem test_dmtx1, but different test
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)

    assert_equal(X.shape, (128, 7))


def test_dmtx1c():
    # idem test_dmtx1, but different test
    tr = 1.0
    frametimes = np.linspace(0, 127 *tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X,names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)
    assert_true((X[:, - 1] == 1).all())


def test_dmtx1d():
    # idem test_dmtx1, but different test
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X,names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='polynomial', drift_order=3)
    assert_true((np.isnan(X) == 0).all())
       
def test_dmtx2():
    # idem test_dmtx1 with a different drift term
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='cosine', hfcut=63)
    assert_equal(len(names), 7) # was 8 with old cosine

def test_dmtx3():
    # idem test_dmtx1 with a different drift term
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    X,names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='blank')
    assert_equal(len(names), 4)  

def test_dmtx4():
    # idem test_dmtx1 with a different hrf model
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical With Derivative'
    X, names= dmtx_light(frametimes, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 10)

def test_dmtx5():
    # idem test_dmtx1 with a block paradigm
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = block_paradigm()
    hrf_model = 'Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)

def test_dmtx6():
    # idem test_dmtx1 with a block paradigm and the hrf derivative
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = block_paradigm()
    hrf_model = 'Canonical With Derivative'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 10)

def test_dmtx7():
    # idem test_dmtx1, but odd paradigm
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    conditions = [0, 0, 0, 1, 1, 1, 3, 3, 3]
    # no condition 'c2'
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    paradigm = EventRelatedParadigm(conditions, onsets)
    hrf_model = 'Canonical'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                          drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)

def test_dmtx8():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    assert_equal(len(names), 7)

def test_dmtx9():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                            drift_model='polynomial', drift_order=3,
                            fir_delays=range(1, 5))
    assert_equal(len(names), 16)

def test_dmtx10():
    # Check that the first column o FIR design matrix is OK
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int)
    assert_true(np.all((X[onset + 1, 0] == 1)))


def test_dmtx11():
    # check that the second column of the FIR design matrix is OK indeed
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 3, 2] == 1))


def test_dmtx12():
    # check that the 11th column of a FIR design matrix is indeed OK
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr,128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.con_id == 'c2'].astype(np.int)
    assert_true(np.all(X[onset + 4, 11] == 1))


def test_dmtx13():
    # Check that the fir_duration is well taken into account
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                          drift_model='polynomial', drift_order=3,
                          fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 1, 0] == 1))


def test_dmtx14():
    # Check that the first column o FIR design matrix is OK after a 1/2
    # time shift
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128) + tr / 2
    paradigm = basic_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3,
                         fir_delays=range(1, 5))
    onset = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int)
    assert_true(np.all(X[onset + 1, 0] > .9))


def test_dmtx15():
    # basic test based on basic_paradigm, plus user supplied regressors 
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    ax = np.random.randn(128, 4)
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3, add_regs=ax)
    assert_equal(len(names), 11)
    assert_equal(X.shape[1], 11)

def test_dmtx16():
    # Check that additional regressors are put at the right place
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = basic_paradigm()
    hrf_model = 'Canonical'
    ax = np.random.randn(128, 4)
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3, add_regs=ax)
    assert_almost_equal(X[:, 3: 7], ax)


def test_dmtx17():
    # Test the effect of scaling on the events
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    hrf_model = 'Canonical'
    X, names = dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    ct = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int) + 1
    assert_true((X[ct, 0] > 0).all())


def test_dmtx18():
    # Test the effect of scaling on the blocks
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_block_paradigm()
    hrf_model = 'Canonical'
    X, names = dmtx_light(frametimes, paradigm, hrf_model=hrf_model,
                         drift_model='polynomial', drift_order=3)
    ct = paradigm.onset[paradigm.con_id == 'c0'].astype(np.int) + 3
    assert_true((X[ct, 0] > 0).all())


def test_dmtx19():
    # Test the effect of scaling on a FIR model
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    hrf_model = 'FIR'
    X, names = dmtx_light(frametimes, paradigm, hrf_model=hrf_model, 
                            drift_model='polynomial', drift_order=3,
                            fir_delays=range(1, 5))
    idx = paradigm.onset[paradigm.con_id == 0].astype(np.int)
    assert_array_equal(X[idx + 1, 0], X[idx + 2, 1])


def test_dmtx20():
    # Test for commit 10662f7
    frametimes = np.arange(0, 128) # was 127 in old version of _cosine_drift 
    paradigm = modulated_event_paradigm()
    X, names = dmtx_light(frametimes, paradigm, hrf_model='canonical',
        drift_model='cosine')

    # check that the drifts are not constant
    assert_true(np.all(np.diff(X[:, -2]) != 0))


def test_fir_block():
    # tets FIR models on block designs
    bp = block_paradigm()
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    X, names = dmtx_light(frametimes, bp, hrf_model='fir', drift_model='blank',
                          fir_delays=range(0, 4))
    idx = bp.onset[bp.con_id == 1].astype(np.int)
    assert_equal(X.shape, (128, 13))
    assert_true((X[idx, 4] == 1).all())
    assert_true((X[idx + 1, 5] == 1).all())
    assert_true((X[idx + 2, 6] == 1).all())
    assert_true((X[idx + 3, 7] == 1).all())


def test_csv_io():
    # test the csv io on design matrices
    tr = 1.0
    frametimes = np.linspace(0, 127 * tr, 128)
    paradigm = modulated_event_paradigm()
    DM = make_dmtx(frametimes, paradigm, hrf_model='Canonical',
                   drift_model='polynomial', drift_order=3)
    path = 'dmtx.csv'
    with InTemporaryDirectory():
        DM.write_csv(path)
        DM2 = dmtx_from_csv(path)
    assert_almost_equal(DM.matrix, DM2.matrix)
    assert_equal(DM.names, DM2.names)


def test_spm_1():
    # Check that the nipy design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frametimes = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    paradigm =  EventRelatedParadigm(conditions, onsets)
    X1 = make_dmtx(frametimes, paradigm, drift_model='blank')
    spm_dmtx = DMTX['arr_0']
    assert_true(((spm_dmtx - X1.matrix) ** 2).sum() / (spm_dmtx ** 2).sum()
                < .1)


def test_spm_2():
    # Check that the nipy design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frametimes = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    duration = 10 * np.ones(9)
    paradigm =  BlockParadigm(conditions, onsets, duration)
    X1 = make_dmtx(frametimes, paradigm, drift_model='blank')
    spm_dmtx = DMTX['arr_1']
    assert_true(((spm_dmtx - X1.matrix) ** 2).sum() / (spm_dmtx ** 2).sum()
                < .1)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
