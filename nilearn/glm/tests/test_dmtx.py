"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""
from os import path as osp

import numpy as np
import pandas as pd
import pytest

from nibabel.tmpdirs import InTemporaryDirectory
from numpy.testing import (
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_equal)

from nilearn.glm.first_level.design_matrix import (_convolve_regressors,
                                                   _cosine_drift,
                                                   check_design_matrix,
                                                   make_first_level_design_matrix,
                                                   make_second_level_design_matrix,
                                                   )

# load the spm file to test cosine basis
my_path = osp.dirname(osp.abspath(__file__))
full_path_design_matrix_file = osp.join(my_path, 'spm_dmtx.npz')
DESIGN_MATRIX = np.load(full_path_design_matrix_file)


def design_matrix_light(
    frame_times, events=None, hrf_model='glover',
    drift_model='cosine', high_pass=.01, drift_order=1, fir_delays=None,
    add_regs=None, add_reg_names=None, min_onset=-24, path=None
    ):
    """ Same as make_first_level_design_matrix,
    but only returns the computed matrix and associated name.
    """
    fir_delays = fir_delays if fir_delays else [0]
    dmtx = make_first_level_design_matrix(frame_times, events, hrf_model,
                                          drift_model, high_pass, drift_order,
                                          fir_delays,
                                          add_regs, add_reg_names, min_onset)
    _, matrix, names = check_design_matrix(dmtx)
    return matrix, names


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def modulated_block_paradigm():
    rng = np.random.RandomState(42)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 + 5 * rng.uniform(size=len(onsets))
    values = 1 + rng.uniform(size=len(onsets))
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
    values = 1 + rng.uniform(size=len(onsets))
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations,
                           'modulation': values})
    return events


def block_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 5 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def test_cosine_drift():
    # add something so that when the tests are launched
    # from a different directory
    spm_drifts = DESIGN_MATRIX['cosbf_dt_1_nt_20_hcut_0p1']
    frame_times = np.arange(20)
    high_pass_frequency = .1
    nistats_drifts = _cosine_drift(high_pass_frequency, frame_times)
    assert_almost_equal(spm_drifts[:, 1:], nistats_drifts[:, : -2])
    # nistats_drifts is placing the constant at the end [:, : - 1]


def test_design_matrix0():
    # Test design matrix creation when no experimental paradigm is provided
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    _, X, names = check_design_matrix(make_first_level_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3))
    assert len(names) == 4
    x = np.linspace(- 0.5, .5, 128)
    assert_almost_equal(X[:, 0], x)


def test_design_matrix0c():
    # test design matrix creation when regressors are provided manually
    rng = np.random.RandomState(42)
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    ax = rng.standard_normal(size=(128, 4))
    _, X, names = check_design_matrix(make_first_level_design_matrix(
        frame_times, drift_model='polynomial',
        drift_order=3, add_regs=ax))
    assert_almost_equal(X[:, 0], ax[:, 0])
    ax = rng.standard_normal(size=(127, 4))
    with pytest.raises(
        AssertionError,
        match="Incorrect specification of additional regressors:."
    ):
        make_first_level_design_matrix(frame_times, add_regs=ax)
    ax = rng.standard_normal(size=(128, 4))
    with pytest.raises(
        ValueError,
        match="Incorrect number of additional regressor names."
    ):
        make_first_level_design_matrix(frame_times,
                                       add_regs=ax,
                                       add_reg_names='')
    # with pandas Dataframe
    axdf = pd.DataFrame(ax)
    _, X1, names = check_design_matrix(make_first_level_design_matrix(
        frame_times, drift_model='polynomial',
        drift_order=3, add_regs=axdf))
    assert_almost_equal(X1[:, 0], ax[:, 0])
    assert_array_equal(names[:4],  np.arange(4))


def test_design_matrix0d():
    # test design matrix creation when regressors are provided manually
    rng = np.random.RandomState(42)
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    ax = rng.standard_normal(size=(128, 4))
    _, X, names = check_design_matrix(make_first_level_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3, add_regs=ax))
    assert len(names) == 8
    assert X.shape[1] == 8


def test_design_matrix10():
    # Check that the first column o FIR design matrix is OK
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    onset = events.onset[events.trial_type == 'c0'].astype(np.int)
    assert_array_almost_equal(X[onset + 1, 0], np.ones(3))


def test_convolve_regressors():
    # tests for convolve_regressors helper function
    conditions = ['c0', 'c1']
    onsets = [20, 40]
    duration = [1, 1]
    events = pd.DataFrame(
        {'trial_type': conditions, 'onset': onsets, 'duration': duration})
    # names not passed -> default names
    frame_times = np.arange(100)
    f, names = _convolve_regressors(events, 'glover', frame_times)
    assert names == ['c0', 'c1']


def test_design_matrix1():
    # basic test based on basic_paradigm and glover hrf
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 7
    assert X.shape == (128, 7)
    assert (X[:, - 1] == 1).all()
    assert (np.isnan(X) == 0).all()


def test_design_matrix2():
    # idem test_design_matrix1 with a different drift term
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='cosine', high_pass=1. / 63)
    assert len(names) == 8


def test_design_matrix3():
    # idem test_design_matrix1 with a different drift term
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model=None)
    assert len(names) == 4


def test_design_matrix4():
    # idem test_design_matrix1 with a different hrf model
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover + derivative'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 10


def test_design_matrix5():
    # idem test_design_matrix1 with a block experimental paradigm
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = block_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 7


def test_design_matrix6():
    """
    idem test_design_matrix1 with a block experimental paradigm
    and the hrf derivative
    """
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = block_paradigm()
    hrf_model = 'glover + derivative'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 10


def test_design_matrix7():
    # idem test_design_matrix1, but odd experimental paradigm
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    conditions = [0, 0, 0, 1, 1, 1, 3, 3, 3]
    durations = 1 * np.ones(9)
    # no condition 'c2'
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 7


def test_design_matrix8():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    assert len(names) == 7


def test_design_matrix9():
    # basic test based on basic_paradigm and FIR
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    assert len(names) == 16


def test_design_matrix11():
    # check that the second column of the FIR design matrix is OK indeed
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    onset = events.onset[events.trial_type == 'c0'].astype(np.int)
    assert_array_almost_equal(X[onset + 3, 2], np.ones(3))


def test_design_matrix12():
    # check that the 11th column of a FIR design matrix is indeed OK
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    onset = events.onset[events.trial_type == 'c2'].astype(np.int)
    assert_array_almost_equal(X[onset + 4, 11], np.ones(3))


def test_design_matrix13():
    # Check that the fir_duration is well taken into account
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    onset = events.onset[events.trial_type == 'c0'].astype(np.int)
    assert_array_almost_equal(X[onset + 1, 0], np.ones(3))


def test_design_matrix14():
    # Check that the first column o FIR design matrix is OK after a 1/2
    # time shift
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128) + tr / 2
    events = basic_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    onset = events.onset[events.trial_type == 'c0'].astype(np.int)
    assert np.all(X[onset + 1, 0] > .5)


def test_design_matrix15():
    # basic test based on basic_paradigm, plus user supplied regressors
    rng = np.random.RandomState(42)
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    ax = rng.standard_normal(size=(128, 4))
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   add_regs=ax)
    assert len(names) == 11
    assert X.shape[1] == 11


def test_design_matrix16():
    # Check that additional regressors are put at the right place
    rng = np.random.RandomState(42)
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    ax = rng.standard_normal(size=(128, 4))
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   add_regs=ax)
    assert_almost_equal(X[:, 3: 7], ax)


def test_design_matrix17():
    # Test the effect of scaling on the events
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = modulated_event_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    ct = events.onset[events.trial_type == 'c0'].astype(np.int) + 1
    assert (X[ct, 0] > 0).all()


def test_design_matrix18():
    # Test the effect of scaling on the blocks
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = modulated_block_paradigm()
    hrf_model = 'glover'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3)
    ct = events.onset[events.trial_type == 'c0'].astype(np.int) + 3
    assert (X[ct, 0] > 0).all()


def test_design_matrix19():
    # Test the effect of scaling on a FIR model
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = modulated_event_paradigm()
    hrf_model = 'FIR'
    X, names = design_matrix_light(frame_times, events, hrf_model=hrf_model,
                                   drift_model='polynomial', drift_order=3,
                                   fir_delays=range(1, 5))
    idx = events.onset[events.trial_type == 0].astype(np.int)
    assert_array_equal(X[idx + 1, 0], X[idx + 2, 1])


def test_design_matrix20():
    # Test for commit 10662f7
    frame_times = np.arange(0, 128)  # was 127 in old version of _cosine_drift
    events = modulated_event_paradigm()
    X, names = design_matrix_light(
        frame_times, events, hrf_model='glover', drift_model='cosine')

    # check that the drifts are not constant
    assert np.any(np.diff(X[:, -2]) != 0)


def test_design_matrix21():
    # basic test on repeated names of user supplied regressors
    rng = np.random.RandomState(42)
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = basic_paradigm()
    hrf_model = 'glover'
    ax = rng.standard_normal(size=(128, 4))
    with pytest.raises(ValueError):
        design_matrix_light(frame_times, events,
                            hrf_model=hrf_model, drift_model='polynomial',
                            drift_order=3, add_regs=ax,
                            add_reg_names=['aha'] * ax.shape[1])


def test_fir_block():
    # tets FIR models on block designs
    bp = block_paradigm()
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    X, names = design_matrix_light(
        frame_times, bp, hrf_model='fir', drift_model=None,
        fir_delays=range(0, 4))
    idx = bp['onset'][bp['trial_type'] == 1].astype(np.int)
    assert X.shape == (128, 13)
    assert (X[idx, 4] == 1).all()
    assert (X[idx + 1, 5] == 1).all()
    assert (X[idx + 2, 6] == 1).all()
    assert (X[idx + 3, 7] == 1).all()


def test_oversampling():
    events = basic_paradigm()
    frame_times = np.linspace(0, 127, 128)
    X1 = make_first_level_design_matrix(
        frame_times, events, drift_model=None)
    X2 = make_first_level_design_matrix(
        frame_times, events, drift_model=None, oversampling=50)
    X3 = make_first_level_design_matrix(
        frame_times, events, drift_model=None, oversampling=10)

    # oversampling = 50 by default so X2 = X1, X3 \neq X1, X3 close to X2
    assert_almost_equal(X1.values, X2.values)
    assert_almost_equal(X2.values, X3.values, 0)
    assert (np.linalg.norm(X2.values - X3.values)
            / np.linalg.norm(X2.values) > 1.e-4)

    # fir model, oversampling is forced to 1
    X4 = make_first_level_design_matrix(
        frame_times, events, hrf_model='fir', drift_model=None,
        fir_delays=range(0, 4), oversampling=1)
    X5 = make_first_level_design_matrix(
        frame_times, events, hrf_model='fir', drift_model=None,
        fir_delays=range(0, 4), oversampling=10)
    assert_almost_equal(X4.values, X5.values)


def test_high_pass():
    """ test that high-pass values lead to reasonable design matrices"""
    n_frames = 128
    tr = 2.0
    frame_times = np.arange(0, tr * n_frames, tr)
    X = make_first_level_design_matrix(
        frame_times, drift_model='Cosine', high_pass=1.)
    assert X.shape[1] == n_frames


def test_csv_io():
    # test the csv io on design matrices
    tr = 1.0
    frame_times = np.linspace(0, 127 * tr, 128)
    events = modulated_event_paradigm()
    DM = make_first_level_design_matrix(frame_times,
                                        events,
                                        hrf_model='glover',
                                        drift_model='polynomial',
                                        drift_order=3,
                                        )
    path = 'design_matrix.csv'
    with InTemporaryDirectory():
        DM.to_csv(path)
        DM2 = pd.read_csv(path, index_col=0)

    _, matrix, names = check_design_matrix(DM)
    _, matrix_, names_ = check_design_matrix(DM2)
    assert_almost_equal(matrix, matrix_)
    assert names == names_


def test_spm_1():
    # Check that the nistats design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frame_times = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    durations = 1 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    X1 = make_first_level_design_matrix(frame_times, events, drift_model=None)
    _, matrix, _ = check_design_matrix(X1)
    spm_design_matrix = DESIGN_MATRIX['arr_0']
    assert (((spm_design_matrix - matrix) ** 2).sum()
            / (spm_design_matrix ** 2).sum() < .1)


def test_spm_2():
    # Check that the nistats design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    frame_times = np.linspace(0, 99, 100)
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 50, 70, 10, 30, 80, 30, 40, 60]
    durations = 10 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    X1 = make_first_level_design_matrix(frame_times, events, drift_model=None)
    spm_design_matrix = DESIGN_MATRIX['arr_1']
    _, matrix, _ = check_design_matrix(X1)
    assert (((spm_design_matrix - matrix) ** 2).sum()
            / (spm_design_matrix ** 2).sum() < .1)


def _first_level_dataframe():
    names = ['con_01', 'con_02', 'con_01', 'con_02']
    subjects = ['01', '01', '02', '02']
    maps = ['', '', '', '']
    dataframe = pd.DataFrame({'map_name': names,
                              'subject_label': subjects,
                              'effects_map_path': maps})
    return dataframe


def test_create_second_level_design():
    subjects_label = ['02', '01']  # change order to test right output order
    regressors = [['01', 0.1], ['02', 0.75]]
    regressors = pd.DataFrame(regressors, columns=['subject_label', 'f1'])
    design = make_second_level_design_matrix(subjects_label, regressors)
    expected_design = np.array([[0.75, 1], [0.1, 1]])
    assert_array_equal(design, expected_design)
    assert len(design.columns) == 2
    assert len(design) == 2
