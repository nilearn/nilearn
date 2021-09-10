import numpy as np
import pandas as pd
import pytest

from nilearn.load_confounds import scrub

from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("original_motion_outliers_index,expected_optimal",
                         [([3, 11, 13, 80, 97],
                           np.array([0, 1, 2, 3, 11, 12, 13, 80, 97, 98, 99])),
                          ([11, 13, 16, 44, 50, 80],  # Middle volumes only
                           np.array([11, 12, 13, 14, 15, 16, 44, 50, 80])),
                          ([4], np.array([0, 1, 2, 3, 4])), # head volumes only
                          ([96], np.array([96, 97, 98, 99])), # tail volumes only
                          ([5], np.array([5]))]) # no optimisation needed
def test_optimize_scrub(original_motion_outliers_index, expected_optimal):
    optimised_index = scrub._optimize_scrub(
        original_motion_outliers_index, 100)
    assert np.array_equal(optimised_index, expected_optimal)


def test_get_outlier_cols():
    col_names = ["confound_regressor"]
    non_steady_state = [f"non_steady_state_outlier{i:02d}" for i in range(3)]
    col_names += non_steady_state
    col_names = pd.Index(col_names)
    outlier_cols, confounds_cols = scrub._get_outlier_cols(col_names)
    assert confounds_cols == ["confound_regressor"]
    assert outlier_cols == non_steady_state


def test_extract_outlier_regressors():
    # Create a fake confound dataframe
    n_scans = 50
    fake_confounds = pd.DataFrame(np.random.rand(n_scans, 1),
                                  columns=["confound_regressor"])

    # scrubbed volume one-hot, overlap with non-steady-state
    idx_scrubbed = [2, 4, 34, 44]
    scrub_vol = pd.DataFrame(np.eye(n_scans)[:, idx_scrubbed],
                             columns=[f"motion_outlier{i:02d}"
                                      for i in range(len(idx_scrubbed))])
    # First three volumes are non-steady-state
    non_steady_vol = pd.DataFrame(np.eye(n_scans)[:, :3],
                                  columns=[f"non_steady_state_outlier{i:02d}"
                                           for i in range(3)])

    # non-steady only
    non_steady_conf = pd.concat([fake_confounds, non_steady_vol], axis=1)
    sample_mask, confounds, outliers = scrub._extract_outlier_regressors(
        non_steady_conf)
    assert np.array_equal(sample_mask, np.arange(n_scans)[3:]) is True
    assert_frame_equal(outliers, non_steady_vol)
    assert_frame_equal(confounds, fake_confounds)

    # scrub only
    srub_conf = pd.concat([fake_confounds, scrub_vol], axis=1)
    make_mask = np.delete(np.arange(n_scans), idx_scrubbed)
    sample_mask, confounds, outliers = scrub._extract_outlier_regressors(
        srub_conf)
    assert np.array_equal(sample_mask, make_mask) is True
    assert_frame_equal(outliers, scrub_vol)
    assert_frame_equal(confounds, fake_confounds)

    # scrub and non-steady state
    all_conf = pd.concat([fake_confounds, non_steady_vol, scrub_vol], axis=1)
    make_mask = np.delete(np.arange(n_scans), idx_scrubbed)[2:]
    make_outliers = pd.concat([non_steady_vol, scrub_vol], axis=1)
    make_outliers = make_outliers.reindex(sorted(make_outliers.columns),
                                          axis=1)
    make_outliers = make_outliers.drop(columns="non_steady_state_outlier02")

    sample_mask, confounds, outliers = scrub._extract_outlier_regressors(
        all_conf)
    assert len(sample_mask) == 44
    assert np.array_equal(sample_mask, make_mask) is True
    assert_frame_equal(outliers, make_outliers)
    assert_frame_equal(confounds, fake_confounds)
