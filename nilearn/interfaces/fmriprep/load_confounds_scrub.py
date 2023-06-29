"""Helper functions for _load_scrub and sample_mask functions."""
import numpy as np
import pandas as pd


def _optimize_scrub(motion_outliers_index, n_scans, scrub):
    """Remove continuous segments with fewer than a minimal segment length."""
    # Start by checking if the beginning continuous segment is fewer than
    # a minimal segment length (default to 5)
    if motion_outliers_index[0] < scrub:
        motion_outliers_index = np.asarray(
            list(range(motion_outliers_index[0])) + list(motion_outliers_index)
        )
    # Do the same for the ending segment of scans
    if n_scans - (motion_outliers_index[-1] + 1) < scrub:
        motion_outliers_index = np.asarray(
            list(motion_outliers_index)
            + list(range(motion_outliers_index[-1], n_scans))
        )
    # Now do everything in between
    fd_outlier_ind_diffs = np.diff(motion_outliers_index)
    short_segments_inds = np.where(
        np.logical_and(
            fd_outlier_ind_diffs > 1, fd_outlier_ind_diffs < (scrub + 1)
        )
    )[0]
    for ind in short_segments_inds:
        motion_outliers_index = np.asarray(
            list(motion_outliers_index)
            + list(
                range(
                    motion_outliers_index[ind] + 1,
                    motion_outliers_index[ind + 1],
                )
            )
        )
    motion_outliers_index = np.sort(np.unique(motion_outliers_index))
    return motion_outliers_index


def _extract_outlier_regressors(confounds):
    """Separate outlier regressors from other confounds variables."""
    outlier_cols, confounds_cols = _get_outlier_cols(confounds.columns)
    if outlier_cols:
        outliers = confounds.loc[:, outlier_cols]
        outliers = outliers.T.drop_duplicates().T
    else:
        outliers = pd.DataFrame()
    confounds = confounds.loc[:, confounds_cols]
    sample_mask = _outlier_to_sample_mask(outliers)
    return sample_mask, confounds, outliers


def _get_outlier_cols(confounds_columns):
    """Get outlier regressor column names."""
    outlier_cols = {
        col
        for col in confounds_columns
        if "motion_outlier" in col or "non_steady_state" in col
    }
    confounds_cols = set(confounds_columns) - outlier_cols
    return sorted(outlier_cols), sorted(confounds_cols)


def _outlier_to_sample_mask(outliers):
    """Generate sample mask from outlier regressors."""
    outliers_one_hot = outliers.copy()
    if outliers_one_hot.size == 0:  # Do not supply sample mask
        return None  # consistency with nilearn sample_mask
    outliers_one_hot = outliers_one_hot.sum(axis=1).values
    return np.where(outliers_one_hot == 0)[0]
