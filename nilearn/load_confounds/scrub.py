"""Helper functions for scrubbing and sampl_mask related functions."""
import numpy as np
import pandas as pd


def _optimize_scrub(fd_outliers, n_scans):
    """
    Perform optimized scrub. After scrub volumes, further remove
    continuous segments containing fewer than 5 volumes.
    Power, Jonathan D., et al. "Methods to detect, characterize, and remove
    motion artifact in resting state fMRI." Neuroimage 84 (2014): 320-341.
    """
    # Start by checking if the beginning continuous segment is fewer than
    # 5 volumes
    if fd_outliers[0] < 5:
        fd_outliers = np.asarray(
            list(range(fd_outliers[0])) + list(fd_outliers)
        )
    # Do the same for the ending segment of scans
    if n_scans - (fd_outliers[-1] + 1) < 5:
        fd_outliers = np.asarray(
            list(fd_outliers) + list(range(fd_outliers[-1], n_scans))
        )
    # Now do everything in between
    fd_outlier_ind_diffs = np.diff(fd_outliers)
    short_segments_inds = np.where(
        np.logical_and(fd_outlier_ind_diffs > 1, fd_outlier_ind_diffs < 6)
    )[0]
    for ind in short_segments_inds:
        fd_outliers = np.asarray(
            list(fd_outliers)
            + list(range(fd_outliers[ind] + 1, fd_outliers[ind + 1]))
        )
    fd_outliers = np.sort(np.unique(fd_outliers))
    return fd_outliers


def _extract_outlier_regressors(confounds):
    """Separate confounds and outlier regressors."""
    outlier_cols, confounds_col = _get_outlier_cols(confounds.columns)
    outliers = confounds[outlier_cols] if outlier_cols else pd.DataFrame()
    confounds = confounds[confounds_col]
    sample_mask = _outlier_to_sample_mask(outliers)
    return sample_mask, confounds, outliers


def _get_outlier_cols(confounds_columns):
    """Get outlier regressor column names."""
    outlier_cols = {
        col
        for col in confounds_columns
        if "motion_outlier" in col or "non_steady_state" in col
    }
    confounds_col = set(confounds_columns) - outlier_cols
    return outlier_cols, confounds_col


def _outlier_to_sample_mask(outlier_flag):
    """Generate sample mask from outlier regressors."""
    if outlier_flag.size == 0:  # Do not supply sample mask
        return None  # consistency with nilearn sample_mask
    outlier_flag = outlier_flag.sum(axis=1).values
    return np.where(outlier_flag == 0)[0]
