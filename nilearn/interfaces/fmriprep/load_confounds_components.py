"""Functions to noise components based on selected strategey.


The _load_* functions  of this module are indirectly used
in nilearn.interfaces.fmriprep._load_noise_component.

See an example below:

.. code-block:: python

    loaded_confounds = getattr(components, f"_load_{component}")(
        confounds_raw, **params
    )

"""

import numpy as np
import pandas as pd

from .load_confounds_compcor import find_compcor
from .load_confounds_scrub import optimize_scrub
from .load_confounds_utils import (
    MissingConfoundError,
    add_suffix,
    check_params,
    find_confounds,
)


def _load_motion(confounds_raw, motion):
    """Load the motion regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    motion : str
        Motion strategy to use. Options are "basic",
        "derivatives", "power2", or "full".

    Returns
    -------
    pandas.DataFrame
        DataFrame of motion regressors.

    Raises
    ------
    MissingConfoundError
        When motion regressors are not found or incomplete, raise error
        as motion is not a valid choice of strategy.
    """
    motion_params = add_suffix(
        ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
        motion,
    )
    motion_regressor_check = check_params(confounds_raw, motion_params)
    if isinstance(motion_regressor_check, list):
        raise MissingConfoundError(params=motion_regressor_check)

    if motion_regressor_check:
        return confounds_raw[motion_params]
    else:
        raise MissingConfoundError(keywords=["motion"])


def _load_high_pass(confounds_raw):
    """Load the high pass filter regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    Returns
    -------
    pandas.DataFrame
        DataFrame of high pass filter regressors.
        If not present in file, return an empty DataFrame.
    """
    high_pass_params = find_confounds(confounds_raw, ["cosine"])
    return (
        confounds_raw[high_pass_params] if high_pass_params else pd.DataFrame()
    )


def _load_wm_csf(confounds_raw, wm_csf):
    """Load the regressors derived from the white matter and CSF masks.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    wm_csf : str
        White matter and CSF strategy to use. Options are "basic",
        "derivatives", "power2", or "full".

    Returns
    -------
    pandas.DataFrame
        DataFrame of white matter and CSF regressors.

    Raises
    ------
    MissingConfoundError
        When white matter and CSF regressors are not found, raise error as
        wm_csf is not a valid choice of strategy.
    """
    wm_csf_params = add_suffix(["csf", "white_matter"], wm_csf)
    if check_params(confounds_raw, wm_csf_params):
        return confounds_raw[wm_csf_params]
    else:
        raise MissingConfoundError(keywords=["wm_csf"])


def _load_global_signal(confounds_raw, global_signal):
    """Load the regressors derived from the global signal.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    global_signal : str
        Global signal strategy to use. Options are "basic",
        "derivatives", "power2", or "full".

    Returns
    -------
    pandas.DataFrame
        DataFrame of global signal regressors.

    Raises
    ------
    MissingConfoundError
        When global signal regressors are not found, raise error as global
        signal is not a valid choice of strategy.
    """
    global_params = add_suffix(["global_signal"], global_signal)
    if check_params(confounds_raw, global_params):
        return confounds_raw[global_params]
    else:
        raise MissingConfoundError(keywords=["global_signal"])


def _load_compcor(confounds_raw, meta_json, compcor, n_compcor):
    """Load compcor regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    meta_json : dict
        Dictionary of confounds meta data from the confounds.json file.

    compcor : str
        Compcor strategy to use. Options are "temporal_anat", "temporal",
        "anat", or "combined".

    n_compcor : int or str
        Number of compcor components to retain. If "all", all components
        are retained.

    Returns
    -------
    pandas.DataFrame
        DataFrame of compcor regressors.

    Raises
    ------
    MissingConfoundError
        When compcor regressors are not found, raise error as compcor is
        not a valid choice of strategy.
    """
    compcor_cols = find_compcor(meta_json, compcor, n_compcor)
    if check_params(confounds_raw, compcor_cols):
        return confounds_raw[compcor_cols]
    else:
        raise MissingConfoundError(keywords=["compcor"])


def _load_ica_aroma(confounds_raw, ica_aroma):
    """Load the ICA-AROMA regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    ica_aroma : str
        ICA-AROMA strategy to use. Options are "full", "basic".

    Returns
    -------
    pandas.DataFrame
        DataFrame of ICA-AROMA regressors.
        When ica_aroma is "full", return an empty DataFrame as the
        ICA-AROMA regressors have been handled in the preprocessed bold
        image.

    Raises
    ------
    ValueError
        When ica_aroma is not "full" or "basic".
    """
    if ica_aroma == "full":
        return pd.DataFrame()
    elif ica_aroma == "basic":
        ica_aroma_params = find_confounds(confounds_raw, ["aroma"])
        if not ica_aroma_params:
            raise MissingConfoundError(keywords=["ica_aroma"])
        return confounds_raw[ica_aroma_params]
    else:
        raise ValueError(
            "Please select an option when using ICA-AROMA strategy."
            f"Current input: {ica_aroma}"
        )


def _load_scrub(confounds_raw, scrub, fd_threshold, std_dvars_threshold):
    """Remove volumes if FD and/or DVARS exceeds threshold.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    scrub : int, default=5
        Minimal segment length.
        Segment smaller than the given value will be removed.

    fd_threshold : float
        Threshold for the framewise displacement. Volumes with FD larger
        than the threshold will be removed.

    std_dvars_threshold : float
        Threshold for the standard deviation of DVARS.
        Volumes with DVARS larger than the threshold will be removed.

    Returns
    -------
    motion_outlier_regressors : pandas.DataFrame
        DataFrame of one-hot encoded motion outlier regressors.
    """
    n_scans = len(confounds_raw)
    # Get indices of fd outliers
    fd_outliers_index = np.where(
        confounds_raw["framewise_displacement"] > fd_threshold
    )[0]
    dvars_outliers_index = np.where(
        confounds_raw["std_dvars"] > std_dvars_threshold
    )[0]
    motion_outliers_index = np.sort(
        np.unique(np.concatenate((fd_outliers_index, dvars_outliers_index)))
    )
    # when motion outliers were detected, remove segments with too few
    # timeframes if desired
    if scrub > 0 and len(motion_outliers_index) > 0:
        motion_outliers_index = optimize_scrub(
            motion_outliers_index, n_scans, scrub
        )
    # Make one-hot encoded motion outlier regressors
    motion_outlier_regressors = pd.DataFrame(
        np.transpose(np.eye(n_scans)[motion_outliers_index]).astype(int)
    )
    column_names = [
        f"motion_outlier_{num}"
        for num in range(np.shape(motion_outlier_regressors)[1])
    ]
    motion_outlier_regressors.columns = column_names
    return motion_outlier_regressors


def _load_non_steady_state(confounds_raw):
    """Find non steady state regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        DataFrame of confounds.

    Returns
    -------
    pandas.DataFrame
        DataFrame of non steady state regressors generated by fMRIPrep.
        If none were found, return an empty DataFrame.
    """
    nss_outliers = find_confounds(confounds_raw, ["non_steady_state"])
    return confounds_raw[nss_outliers] if nss_outliers else pd.DataFrame()
