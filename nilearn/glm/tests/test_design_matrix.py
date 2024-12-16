"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from nilearn._utils.data_gen import basic_paradigm
from nilearn.glm.first_level.design_matrix import (
    _convolve_regressors,
    check_design_matrix,
    create_cosine_drift,
    make_first_level_design_matrix,
    make_second_level_design_matrix,
)

from ._testing import (
    block_paradigm,
    design_with_negative_onsets,
    modulated_block_paradigm,
    modulated_event_paradigm,
    spm_paradigm,
)

# load the spm file to test cosine basis
my_path = Path(__file__).resolve().parent
full_path_design_matrix_file = my_path / "spm_dmtx.npz"
DESIGN_MATRIX = np.load(full_path_design_matrix_file)


def design_matrix_light(
    frame_times,
    events=None,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=None,
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
):
    """Perform same as make_first_level_design_matrix, \
       but only returns the computed matrix and associated name.
    """
    fir_delays = fir_delays or [0]
    dmtx = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model,
        drift_model,
        high_pass,
        drift_order,
        fir_delays,
        add_regs,
        add_reg_names,
        min_onset,
    )
    _, matrix, names = check_design_matrix(dmtx)
    return matrix, names


@pytest.fixture
def n_frames():
    return 128


@pytest.fixture
def frame_times(n_frames):
    t_r = 1.0
    return np.linspace(0, (n_frames - 1) * t_r, n_frames)


def test_cosine_drift():
    # add something so that when the tests are launched
    # from a different directory
    spm_drifts = DESIGN_MATRIX["cosbf_dt_1_nt_20_hcut_0p1"]
    frame_times = np.arange(20)
    high_pass_frequency = 0.1
    nilearn_drifts = create_cosine_drift(high_pass_frequency, frame_times)
    assert_almost_equal(spm_drifts[:, 1:], nilearn_drifts[:, :-2])
    # nilearn_drifts is placing the constant at the end [:, : - 1]


def test_design_matrix_no_experimental_paradigm(frame_times):
    # Test design matrix creation when no experimental paradigm is provided
    _, X, names = check_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3
        )
    )
    assert len(names) == 4
    x = np.linspace(-0.5, 0.5, len(frame_times))
    assert_almost_equal(X[:, 0], x)


def test_design_matrix_regressors_provided_manually(rng, frame_times):
    # test design matrix creation when regressors are provided manually
    ax = rng.standard_normal(size=(len(frame_times), 4))
    _, X, names = check_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3, add_regs=ax
        )
    )
    assert_almost_equal(X[:, 0], ax[:, 0])
    assert len(names) == 8
    assert X.shape[1] == 8

    # with pandas Dataframe
    axdf = pd.DataFrame(ax)
    _, X1, names = check_design_matrix(
        make_first_level_design_matrix(
            frame_times, drift_model="polynomial", drift_order=3, add_regs=axdf
        )
    )
    assert_almost_equal(X1[:, 0], ax[:, 0])
    assert_array_equal(names[:4], np.arange(4))


def test_design_matrix_regressors_provided_manually_errors(rng, frame_times):
    ax = rng.standard_normal(size=(len(frame_times) - 1, 4))
    with pytest.raises(
        AssertionError,
        match="Incorrect specification of additional regressors:.",
    ):
        make_first_level_design_matrix(frame_times, add_regs=ax)

    ax = rng.standard_normal(size=(len(frame_times), 4))
    with pytest.raises(
        ValueError, match="Incorrect number of additional regressor names."
    ):
        make_first_level_design_matrix(
            frame_times, add_regs=ax, add_reg_names=""
        )


def test_convolve_regressors(frame_times):
    # tests for convolve_regressors helper function
    _, names = _convolve_regressors(basic_paradigm(), "glover", frame_times)
    assert names == ["c0", "c1", "c2"]


def test_design_matrix_basic_paradigm_glover_hrf(frame_times):
    X, _ = design_matrix_light(
        frame_times,
        events=basic_paradigm(),
        hrf_model="glover",
        drift_model="polynomial",
        drift_order=3,
    )
    assert (X[:, -1] == 1).all()
    assert (np.isnan(X) == 0).all()


@pytest.mark.parametrize(
    "events, hrf_model, drift_model, drift_order, high_pass, n_regressors",
    [
        (basic_paradigm(), "glover", None, 1, 0.01, 4),
        (
            basic_paradigm(),
            "glover",
            "cosine",
            1,
            1.0 / 63,
            8,
        ),
        (basic_paradigm(), "glover + derivative", "polynomial", 3, 0.01, 10),
        (block_paradigm(), "glover", "polynomial", 1, 0.01, 5),
        (block_paradigm(), "glover", "polynomial", 3, 0.01, 7),
        (block_paradigm(), "glover + derivative", "polynomial", 3, 0.01, 10),
    ],
)
def test_design_matrix(
    frame_times,
    n_frames,
    events,
    hrf_model,
    drift_model,
    drift_order,
    high_pass,
    n_regressors,
):
    X, names = design_matrix_light(
        frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        drift_order=drift_order,
        high_pass=high_pass,
    )
    assert len(names) == n_regressors
    assert X.shape == (n_frames, n_regressors)


def test_design_matrix_basic_paradigm_and_extra_regressors(rng, frame_times):
    # basic test based on basic_paradigm, plus user supplied regressors
    ax = rng.standard_normal(size=(len(frame_times), 4))
    X, names = design_matrix_light(
        frame_times,
        events=basic_paradigm(),
        hrf_model="glover",
        drift_model="polynomial",
        drift_order=3,
        add_regs=ax,
    )
    assert len(names) == 11
    assert X.shape[1] == 11
    # Check that additional regressors are put at the right place
    assert_almost_equal(X[:, 3:7], ax)


@pytest.mark.parametrize(
    "fir_delays, n_regressors", [(None, 7), (range(1, 5), 16)]
)
def test_design_matrix_fir_basic_paradigm(
    frame_times, fir_delays, n_regressors
):
    # basic test based on basic_paradigm and FIR
    X, names = design_matrix_light(
        frame_times,
        events=basic_paradigm(),
        hrf_model="FIR",
        drift_model="polynomial",
        drift_order=3,
        fir_delays=fir_delays,
    )
    assert len(names) == n_regressors
    assert X.shape == (len(frame_times), n_regressors)


def test_design_matrix_fir_block(frame_times):
    # test FIR models on block designs
    bp = block_paradigm()
    X, _ = design_matrix_light(
        frame_times,
        bp,
        hrf_model="fir",
        drift_model=None,
        fir_delays=range(4),
    )
    idx = bp["onset"][bp["trial_type"] == 1].astype(int)
    assert X.shape == (len(frame_times), 13)
    assert (X[idx, 4] == 1).all()
    assert (X[idx + 1, 5] == 1).all()
    assert (X[idx + 2, 6] == 1).all()
    assert (X[idx + 3, 7] == 1).all()


def test_design_matrix_fir_column_1_3_and_11(frame_times):
    # Check that 1rst, 3rd and 11th of FIR design matrix are OK
    events = basic_paradigm()
    hrf_model = "FIR"
    X, _ = design_matrix_light(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model="polynomial",
        drift_order=3,
        fir_delays=range(1, 5),
    )
    onset = events.onset[events.trial_type == "c0"].astype(int)
    assert_array_almost_equal(X[onset + 1, 0], np.ones(3))
    assert_array_almost_equal(X[onset + 3, 2], np.ones(3))

    onset = events.onset[events.trial_type == "c2"].astype(int)
    assert_array_almost_equal(X[onset + 4, 11], np.ones(3))


def test_design_matrix_fir_time_shift(frame_times):
    # Check that the first column of FIR design matrix is OK after a 1/2
    # time shift
    t_r = 1.0
    frame_times = frame_times + t_r / 2
    events = basic_paradigm()
    hrf_model = "FIR"
    X, _ = design_matrix_light(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model="polynomial",
        drift_order=3,
        fir_delays=range(1, 5),
    )
    ct = events.onset[events.trial_type == "c0"].astype(int)
    assert np.all(X[ct + 1, 0] > 0.5)


@pytest.mark.parametrize(
    "events, idx_offset",
    [(modulated_event_paradigm(), 1), (modulated_block_paradigm(), 3)],
)
def test_design_matrix_scaling(events, idx_offset, frame_times):
    X, _ = design_matrix_light(
        frame_times,
        events=events,
        hrf_model="glover",
        drift_model="polynomial",
        drift_order=3,
    )
    idx = events.onset[events.trial_type == "c0"].astype(int)
    ct = idx + idx_offset
    assert (X[ct, 0] > 0).all()


def test_design_matrix_scaling_fir_model(frame_times):
    # Test the effect of scaling on a FIR model
    events = modulated_event_paradigm()
    hrf_model = "FIR"
    X, _ = design_matrix_light(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model="polynomial",
        drift_order=3,
        fir_delays=range(1, 5),
    )
    idx = events.onset[events.trial_type == 0].astype(int)
    assert_array_equal(X[idx + 1, 0], X[idx + 2, 1])


def test_design_matrix20(n_frames):
    # Test for commit 10662f7
    frame_times = np.arange(
        0, n_frames
    )  # was 127 in old version of create_cosine_drift
    events = modulated_event_paradigm()
    X, _ = design_matrix_light(
        frame_times, events, hrf_model="glover", drift_model="cosine"
    )

    # check that the drifts are not constant
    assert np.any(np.diff(X[:, -2]) != 0)


def test_design_matrix_repeated_name_in_user_regressors(rng, frame_times):
    # basic test on repeated names of user supplied regressors
    events = basic_paradigm()
    hrf_model = "glover"
    ax = rng.standard_normal(size=(len(frame_times), 4))
    with pytest.raises(
        ValueError, match="Design matrix columns do not have unique names"
    ):
        design_matrix_light(
            frame_times,
            events,
            hrf_model=hrf_model,
            drift_model="polynomial",
            drift_order=3,
            add_regs=ax,
            add_reg_names=["aha"] * ax.shape[1],
        )


def test_oversampling(n_frames):
    events = basic_paradigm()
    frame_times = np.linspace(0, n_frames - 1, n_frames)
    X1 = make_first_level_design_matrix(frame_times, events, drift_model=None)
    X2 = make_first_level_design_matrix(
        frame_times, events, drift_model=None, oversampling=50
    )
    X3 = make_first_level_design_matrix(
        frame_times, events, drift_model=None, oversampling=10
    )

    # oversampling = 50 by default so X2 = X1, X3 \neq X1, X3 close to X2
    assert_almost_equal(X1.to_numpy(), X2.to_numpy())
    assert_almost_equal(X2.to_numpy(), X3.to_numpy(), 0)
    assert (
        np.linalg.norm(X2.to_numpy() - X3.to_numpy())
        / np.linalg.norm(X2.to_numpy())
        > 1.0e-4
    )

    # fir model, oversampling is forced to 1
    X4 = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model="fir",
        drift_model=None,
        fir_delays=range(4),
        oversampling=1,
    )
    X5 = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model="fir",
        drift_model=None,
        fir_delays=range(4),
        oversampling=10,
    )
    assert_almost_equal(X4.to_numpy(), X5.to_numpy())


def test_events_as_path(n_frames, tmp_path):
    events = basic_paradigm()
    frame_times = np.linspace(0, n_frames - 1, n_frames)

    events_file = tmp_path / "design.csv"
    events.to_csv(events_file)
    make_first_level_design_matrix(frame_times, events=events_file)
    make_first_level_design_matrix(frame_times, events=str(events_file))

    events_file = tmp_path / "design.tsv"
    events.to_csv(events_file, sep="\t")
    make_first_level_design_matrix(frame_times, events=events_file)
    make_first_level_design_matrix(frame_times, events=str(events_file))


def test_high_pass(n_frames):
    """Test that high-pass values lead to reasonable design matrices."""
    t_r = 2.0
    frame_times = np.arange(0, t_r * n_frames, t_r)
    X = make_first_level_design_matrix(
        frame_times, drift_model="Cosine", high_pass=1.0
    )
    assert X.shape[1] == n_frames


def test_csv_io(tmp_path, frame_times):
    # test the csv io on design matrices
    DM = make_first_level_design_matrix(
        frame_times,
        events=modulated_event_paradigm(),
        hrf_model="glover",
        drift_model="polynomial",
        drift_order=3,
    )
    path = tmp_path / "design_matrix.csv"
    DM.to_csv(path)
    DM2 = pd.read_csv(path, index_col=0)

    _, matrix, names = check_design_matrix(DM)
    _, matrix_, names_ = check_design_matrix(DM2)
    assert_almost_equal(matrix, matrix_)
    assert names == names_


@pytest.mark.parametrize(
    "block_duration, array", [(1, "arr_0"), (10, "arr_1")]
)
def test_compare_design_matrix_to_spm(block_duration, array):
    # Check that the nilearn design matrix is close enough to the SPM one
    # (it cannot be identical, because the hrf shape is different)
    events, frame_times = spm_paradigm(block_duration=block_duration)
    X1 = make_first_level_design_matrix(
        frame_times, events, drift_model=None, hrf_model="spm"
    )
    _, matrix, _ = check_design_matrix(X1)

    spm_design_matrix = DESIGN_MATRIX[array]

    assert ((spm_design_matrix - matrix) ** 2).sum() / (
        spm_design_matrix**2
    ).sum() < 0.1


def test_create_second_level_design():
    subjects_label = ["02", "01"]  # change order to test right output order
    regressors = [["01", 0.1], ["02", 0.75]]
    regressors = pd.DataFrame(regressors, columns=["subject_label", "f1"])
    design = make_second_level_design_matrix(subjects_label, regressors)
    expected_design = np.array([[0.75, 1.0], [0.1, 1.0]])
    assert_array_equal(design, expected_design)
    assert len(design.columns) == 2
    assert len(design) == 2


def test_designs_with_negative_onsets_warning(frame_times):
    with pytest.warns(
        UserWarning,
        match="Some stimulus onsets are earlier than",
    ):
        make_first_level_design_matrix(
            events=design_with_negative_onsets(), frame_times=frame_times
        )
