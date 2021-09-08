"""Helper functions for the manipulation of confounds.

Authors: load_confounds team
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
import json
import re

from .scrub import _extract_outlier_regressors
from .._utils.load_confounds import _flag_single_gifti


img_file_patterns = {
    "aroma": "_desc-smoothAROMAnonaggr_bold",
    "nii.gz": "_space-.*_desc-preproc_bold.nii.gz",
    "dtseries.nii": "_space-.*_bold.dtseries.nii",
    "func.gii": "_space-.*_hemi-[LR]_bold.func.gii",
}

img_file_error = {
    "aroma": (
        "Input must be ~desc-smoothAROMAnonaggr_bold for full ICA-AROMA"
        " strategy."
    ),
    "nii.gz": "Invalid file type for the selected method.",
    "dtseries.nii": "Invalid file type for the selected method.",
    "func.gii": "need fMRIprep output with extension func.gii",
}


def _check_params(confounds_raw, params):
    """Check that specified parameters can be found in the confounds."""
    not_found_params = [
        par for par in params if par not in confounds_raw.columns
    ]
    if not_found_params:
        raise MissingConfound(params=not_found_params)
    return None


def _find_confounds(confounds_raw, keywords):
    """Find confounds that contain certain keywords."""
    list_confounds, missing_keys = [], []
    for key in keywords:
        key_found = [col for col in confounds_raw.columns if key in col]
        if key_found:
            list_confounds.extend(key_found)
        elif key != "non_steady_state":
            missing_keys.append(key)
    if missing_keys:
        raise MissingConfound(keywords=missing_keys)
    return list_confounds


def _sanitize_confounds(img_files):
    """Make sure the inputs are in the correct format."""
    # we want to support loading a single set of confounds, instead of a list
    # so we hack it
    if isinstance(img_files, list) and len(img_files) == 2:
        flag_single = _flag_single_gifti(img_files)
    else:  # single file
        flag_single = isinstance(img_files, str)

    if flag_single:
        img_files = [img_files]
    return img_files, flag_single


def _add_suffix(params, model):
    """
    Add suffixes to a list of parameters.
    Suffixes includes derivatives, power2 and full
    """
    params_full = params.copy()
    suffix = {
        "basic": {},
        "derivatives": {"derivative1"},
        "power2": {"power2"},
        "full": {"derivative1", "power2", "derivative1_power2"},
    }
    for par in params:
        for suff in suffix[model]:
            params_full.append(f"{par}_{suff}")
    return params_full


def _pca_motion(confounds_motion, n_components):
    """Reduce the motion paramaters using PCA."""
    n_available = confounds_motion.shape[1]
    if n_components > n_available:
        raise ValueError(
            (
                f"User requested n_motion={n_components} motion components, "
                f"but found only {n_available}."
            )
        )
    confounds_motion = confounds_motion.dropna()
    confounds_motion_std = scale(
        confounds_motion, axis=0, with_mean=True, with_std=True
    )
    pca = PCA(n_components=n_components)
    motion_pca = pd.DataFrame(pca.fit_transform(confounds_motion_std))
    motion_pca.columns = [
        "motion_pca_" + str(col + 1) for col in motion_pca.columns
    ]
    return motion_pca


def _get_file_name(nii_file):
    """Get the name of the raw confound file."""
    if isinstance(nii_file, list):  # catch gifti
        nii_file = nii_file[0]
    suffix = "_space-" + nii_file.split("space-")[1]
    # fmriprep has changed the file suffix between v20.1.1 and v20.2.0 with
    # respect to BEP 012.
    # cf. https://neurostars.org/t/naming-change-confounds-regressors-to-confounds-timeseries/17637 # noqa
    # Check file with new naming scheme exists or replace,
    # for backward compatibility.
    confounds_raw_candidates = [
        nii_file.replace(suffix, "_desc-confounds_timeseries.tsv"),
        nii_file.replace(suffix, "_desc-confounds_regressors.tsv"),
    ]

    confounds_raw = [
        cr for cr in confounds_raw_candidates if os.path.exists(cr)
    ]

    if not confounds_raw:
        raise ValueError("Could not find associated confound file.")
    elif len(confounds_raw) != 1:
        raise ValueError("Found more than one confound file.")
    else:
        return confounds_raw[0]


def _get_json(confounds_raw_path, flag_acompcor):
    """Load json data companion to the confounds tsv file."""
    # Load JSON file
    confounds_json = confounds_raw_path.replace("tsv", "json")
    try:
        with open(confounds_json, "rb") as f:
            confounds_json = json.load(f)
    except OSError:
        if flag_acompcor:
            raise ValueError(
                (
                    f"Could not find a json file {confounds_json}."
                    "This is necessary for anatomical CompCor."
                    "The CompCor component is only supported for fMRIprep "
                    "version >= 1.4.0."
                )
            )
    return confounds_json


def _get_file_raw(confounds_raw_path):
    """Load raw confounds as a pandas DataFrame and check if the version
    of fMRIprep (>=1.2.0) is supported based on header format (to-do)."""
    confounds_raw = pd.read_csv(
        confounds_raw_path, delimiter="\t", encoding="utf-8"
    )
    return confounds_raw


def _ext_validator(image_file, ext):
    """Check image is valid based on extention."""
    try:
        valid_img = all(
            bool(re.search(img_file_patterns[ext], img)) for img in image_file
        )
        error_message = img_file_error[ext]
    except KeyError:
        valid_img = False
        error_message = "Unsupported input."
    return valid_img, error_message


def _check_images(image_file, flag_full_aroma):
    """Validate input file and ICA AROMA related file."""
    if len(image_file) == 2:  # must be gifti
        valid_img, error_message = _ext_validator(image_file, "func.gii")
    elif flag_full_aroma:
        valid_img, error_message = _ext_validator([image_file], "aroma")
    else:
        ext = ".".join(image_file.split(".")[-2:])
        valid_img, error_message = _ext_validator([image_file], ext)
    if not valid_img:
        raise ValueError(error_message)


def _confounds_to_df(image_file, flag_acompcor, flag_full_aroma):
    """Load raw confounds and associated metadata."""
    _check_images(image_file, flag_full_aroma)
    confounds_raw_path = _get_file_name(image_file)
    confounds_json = _get_json(confounds_raw_path, flag_acompcor)
    confounds_raw = _get_file_raw(confounds_raw_path)
    return confounds_raw, confounds_json


def _prepare_output(confounds, demean):
    """Demean and create sample mask for the selected confounds."""
    sample_mask, confounds, outliers = _extract_outlier_regressors(confounds)
    if confounds.size != 0:  # ica_aroma = "full" generate empty output
        # Derivatives have NaN on the first row
        # Replace them by estimates at second time point,
        # otherwise nilearn will crash.
        mask_nan = np.isnan(confounds.values[0, :])
        confounds.iloc[0, mask_nan] = confounds.iloc[1, mask_nan]
        if demean:
            confounds = _demean_confounds(confounds, sample_mask)
    return sample_mask, confounds


def _demean_confounds(confounds, sample_mask):
    """Demean the confounds. The mean is calculated on non-outlier values."""
    confound_cols = confounds.columns
    if sample_mask is None:
        confounds = scale(confounds, axis=0, with_std=False)
    else:  # calculate the mean without outliers.
        confounds_mean = confounds.iloc[sample_mask, :].mean(axis=0)
        confounds -= confounds_mean
    return pd.DataFrame(confounds, columns=confound_cols)


class MissingConfound(Exception):
    """
    Exception raised when failing to find params in the confounds.

    Parameters
    ----------
        params : list of missing params
        keywords: list of missing keywords
    """

    def __init__(self, params=None, keywords=None):
        """Default values are empty lists."""
        self.params = params or []
        self.keywords = keywords or []
