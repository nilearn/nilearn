import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nilearn.interfaces.fmriprep.load_confounds_scrub import (
    _get_outlier_cols,
    extract_outlier_regressors,
    optimize_scrub,
)


@pytest.mark.parametrize(
    "original_motion_outliers_index,expected_optimal",
    [
        (
            [3, 11, 13, 80, 97],
            np.array([0, 1, 2, 3, 11, 12, 13, 80, 97, 98, 99]),
        ),
        (
            [11, 13, 16, 44, 50, 80],  # middle volumes
            np.array([11, 12, 13, 14, 15, 16, 44, 50, 80]),
        ),
        ([4], np.array([0, 1, 2, 3, 4])),  # head volumes
        ([96], np.array([96, 97, 98, 99])),  # tail volumes
        ([5], np.array([5])),
    ],
)  # no optimization needed
def test_optimize_scrub(original_motion_outliers_index, expected_optimal):
    """Check the segment removal is acting correctly."""
    # simulated labels with 100 time frames and remove any segment under
    # 5 volumes
    optimised_index = optimize_scrub(original_motion_outliers_index, 100, 5)
    assert np.array_equal(optimised_index, expected_optimal)


def test_get_outlier_cols():
    """Check the non-steady state columns are detached."""
    col_names = ["confound_regressor"]
    non_steady_state = [f"non_steady_state_outlier{i:02d}" for i in range(3)]
    col_names += non_steady_state
    col_names = pd.Index(col_names)
    outlier_cols, confounds_cols = _get_outlier_cols(col_names)
    assert confounds_cols == ["confound_regressor"]
    assert outlier_cols == non_steady_state


def test_extract_outlier_regressors(rng):
    """Check outlier regressors of different types."""
    # Create a fake confound dataframe
    n_scans = 50
    fake_confounds = pd.DataFrame(
        rng.random((n_scans, 1)), columns=["confound_regressor"]
    )

    # scrubbed volume one-hot, overlap with non-steady-state
    idx_scrubbed = [2, 4, 34, 44]
    scrub_vol = pd.DataFrame(
        np.eye(n_scans)[:, idx_scrubbed],
        columns=[f"motion_outlier{i:02d}" for i in range(len(idx_scrubbed))],
    )
    # First three volumes are non-steady-state
    non_steady_vol = pd.DataFrame(
        np.eye(n_scans)[:, :3],
        columns=[f"non_steady_state_outlier{i:02d}" for i in range(3)],
    )

    # non-steady only
    non_steady_conf = pd.concat([fake_confounds, non_steady_vol], axis=1)
    sample_mask, confounds, outliers = extract_outlier_regressors(
        non_steady_conf
    )
    assert np.array_equal(sample_mask, np.arange(n_scans)[3:]) is True
    assert_frame_equal(outliers, non_steady_vol)
    assert_frame_equal(confounds, fake_confounds)

    # scrub only
    srub_conf = pd.concat([fake_confounds, scrub_vol], axis=1)
    make_mask = np.delete(np.arange(n_scans), idx_scrubbed)
    sample_mask, confounds, outliers = extract_outlier_regressors(srub_conf)
    assert np.array_equal(sample_mask, make_mask) is True
    assert_frame_equal(outliers, scrub_vol)
    assert_frame_equal(confounds, fake_confounds)

    # scrub and non-steady state
    all_conf = pd.concat([fake_confounds, non_steady_vol, scrub_vol], axis=1)
    make_mask = np.delete(np.arange(n_scans), idx_scrubbed)[2:]
    make_outliers = pd.concat([non_steady_vol, scrub_vol], axis=1)
    make_outliers = make_outliers.reindex(
        sorted(make_outliers.columns), axis=1
    )
    make_outliers = make_outliers.drop(columns="non_steady_state_outlier02")

    sample_mask, confounds, outliers = extract_outlier_regressors(all_conf)
    assert len(sample_mask) == 44
    assert np.array_equal(sample_mask, make_mask) is True
    assert_frame_equal(outliers, make_outliers)
    assert_frame_equal(confounds, fake_confounds)


@pytest.mark.parametrize(
    "outlier_type",
    ["motion_outlier", "non_steady_state_outlier"],
)
def test_warning_no_volumes_left(outlier_type):
    rng = np.random.default_rng()
    n_scans = 10
    fake_confounds = pd.DataFrame(
        rng.random((n_scans, 1)), columns=["confound_regressor"]
    )

    # scrubbed volume one-hot, overlap with non-steady-state
    idx_scrubbed = np.arange(n_scans)
    scrub_vol = pd.DataFrame(
        np.eye(n_scans)[:, idx_scrubbed],
        columns=[f"{outlier_type}{i:02d}" for i in range(len(idx_scrubbed))],
    )

    srub_conf = pd.concat([fake_confounds, scrub_vol], axis=1)

    with pytest.warns(
        RuntimeWarning,
        match="All volumes were marked as motion outliers.",
    ):
        sample_mask, _, _ = extract_outlier_regressors(srub_conf)
        assert sample_mask.size == 0
