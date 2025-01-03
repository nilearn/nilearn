"""Helper functions for load_scrub and sample_mask functions."""

import warnings

import numpy as np
import pandas as pd


def optimize_scrub(motion_outliers_index, n_scans, scrub):
    """Remove continuous segments with fewer than a minimal segment length.

    Parameters
    ----------
    motion_outliers_index : numpy.ndarray
        Index array of shape (n_motion_outliers) indicating the volumes
        that are motion outliers.

    n_scans : int
        Number of volumes in the functional image.

    scrub : int, default=5
        Minimal segment length.

    Returns
    -------
    motion_outliers_index : numpy.ndarray
        Index of outlier volumes.
    """
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


def extract_outlier_regressors(confounds):
    """Separate outlier one-hot regressors from other confounds \
    variables and generate a sample mask, indicates the volumes kept.

    Parameters
    ----------
    confounds : pandas.DataFrame
        DataFrame of confounds.

    Returns
    -------
    sample_mask : numpy.ndarray
        Index array of shape (n_samples) indicating the volumes
        that are not outliers.

    confounds : pandas.DataFrame
        DataFrame of confounds without the one-hot encoders for
        outlier regressors.

    outliers : pandas.DataFrame
        DataFrame of outlier regressors.
    """
    outlier_cols, confounds_cols = _get_outlier_cols(confounds.columns)
    if outlier_cols:
        outliers = confounds.loc[:, outlier_cols]
        outliers = outliers.T.drop_duplicates().T
    else:
        outliers = pd.DataFrame()
    confounds = confounds.loc[:, confounds_cols]
    sample_mask = _outlier_to_sample_mask(outliers)

    if sample_mask is not None and sample_mask.size == 0:
        warnings.warn(
            category=RuntimeWarning,
            message="All volumes were marked as motion outliers. "
            "This would lead to all volumes in the time "
            "series to be scrubbed.",
            stacklevel=4,
        )
    return sample_mask, confounds, outliers


def _get_outlier_cols(confounds_columns):
    """Get outlier regressor column names.

    Parameters
    ----------
    confounds_columns : list
        List of confounds column names.

    Returns
    -------
    outlier_cols : list
        List of outlier regressor column names.

    confounds_cols : list
        List of confounds column names without outlier regressors.
    """
    outlier_cols = {
        col
        for col in confounds_columns
        if "motion_outlier" in col or "non_steady_state" in col
    }
    confounds_cols = set(confounds_columns) - outlier_cols
    return sorted(outlier_cols), sorted(confounds_cols)


def _outlier_to_sample_mask(outliers):
    """Generate sample mask from outlier regressors.

    Parameters
    ----------
    outliers : pandas.DataFrame
        DataFrame of outlier regressors. The shape should be
        (number of volumes, number of outlier regressors).

    Returns
    -------
    sample_mask : numpy.ndarray
        Index array of shape indicating the volumes that are not
        outliers. (number of volumes - number of outlier regressors, ).
    """
    outliers_one_hot = outliers.copy()
    if outliers_one_hot.size == 0:  # Do not supply sample mask
        return None  # consistency with nilearn sample_mask
    outliers_one_hot = outliers_one_hot.sum(axis=1).to_numpy()
    return np.where(outliers_one_hot == 0)[0]
