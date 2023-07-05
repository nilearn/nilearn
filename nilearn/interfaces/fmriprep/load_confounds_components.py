"""Functions to noise components based on selected strategey."""
import numpy as np
import pandas as pd

from .load_confounds_compcor import _find_compcor
from .load_confounds_scrub import _optimize_scrub
from .load_confounds_utils import (
    MissingConfound,
    _add_suffix,
    _check_params,
    _find_confounds,
)


def _load_motion(confounds_raw, motion):
    """Load the motion regressors."""
    motion_params = _add_suffix(
        ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
        motion,
    )
    motion_regressor_check = _check_params(confounds_raw, motion_params)
    if isinstance(motion_regressor_check, list):
        raise MissingConfound(params=motion_regressor_check)

    if motion_regressor_check:
        return confounds_raw[motion_params]
    else:
        raise MissingConfound(keywords=["motion"])


def _load_high_pass(confounds_raw):
    """Load the high pass filter regressors."""
    high_pass_params = _find_confounds(confounds_raw, ["cosine"])
    if high_pass_params:
        return confounds_raw[high_pass_params]
    else:
        return pd.DataFrame()


def _load_wm_csf(confounds_raw, wm_csf):
    """Load the regressors derived from the white matter and CSF masks."""
    wm_csf_params = _add_suffix(["csf", "white_matter"], wm_csf)
    if _check_params(confounds_raw, wm_csf_params):
        return confounds_raw[wm_csf_params]
    else:
        raise MissingConfound(keywords=["wm_csf"])


def _load_global_signal(confounds_raw, global_signal):
    """Load the regressors derived from the global signal."""
    global_params = _add_suffix(["global_signal"], global_signal)
    if _check_params(confounds_raw, global_params):
        return confounds_raw[global_params]
    else:
        raise MissingConfound(keywords=["global_signal"])


def _load_compcor(confounds_raw, meta_json, compcor, n_compcor):
    """Load compcor regressors."""
    compcor_cols = _find_compcor(meta_json, compcor, n_compcor)
    if _check_params(confounds_raw, compcor_cols):
        return confounds_raw[compcor_cols]
    else:
        raise MissingConfound(keywords=["compcor"])


def _load_ica_aroma(confounds_raw, ica_aroma):
    """Load the ICA-AROMA regressors."""
    if ica_aroma == "full":
        return pd.DataFrame()
    elif ica_aroma == "basic":
        ica_aroma_params = _find_confounds(confounds_raw, ["aroma"])
        if not ica_aroma_params:
            raise MissingConfound(keywords=["ica_aroma"])
        return confounds_raw[ica_aroma_params]
    else:
        raise ValueError(
            "Please select an option when using ICA-AROMA strategy."
            f"Current input: {ica_aroma}"
        )


def _load_scrub(confounds_raw, scrub, fd_threshold, std_dvars_threshold):
    """Remove volumes if FD and/or DVARS exceeds threshold."""
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
        motion_outliers_index = _optimize_scrub(
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
    """Find non steady state regressors."""
    nss_outliers = _find_confounds(confounds_raw, ["non_steady_state"])
    if nss_outliers:
        return confounds_raw[nss_outliers]
    else:
        return pd.DataFrame()
