"""Helper functions for the manipulation of fmriprep output confounds."""

import itertools
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from nilearn._utils.fmriprep_confounds import flag_single_gifti, is_camel_case
from nilearn.interfaces.bids import parse_bids_filename

from .load_confounds_scrub import extract_outlier_regressors

img_file_patterns = {
    "aroma": "_desc-smoothAROMAnonaggr_bold",
    "nii.gz": "(_space-.*)?_desc-preproc_bold.nii.gz",
    "dtseries.nii": "(_space-.*)?_bold.dtseries.nii",
    "tedana": "_desc-optcom_bold.nii.gz",
    "func.gii": "_hemi-[LR](_space-.*)?_bold.func.gii",
}

img_file_error = {
    "aroma": (
        "Input must be desc-smoothAROMAnonaggr_bold for full ICA-AROMA"
        " strategy."
    ),
    "nii.gz": "Invalid file type for the selected 'nii.gz' method.",
    "dtseries.nii": "Invalid file type for the selected 'dtseries.nii' method",
    "tedana": "Input must be the ~desc-optcom_bold.nii.gz file "
    "for tedana strategy. Other files like "
    "~desc-denoised_bold.nii.gz are not supported or "
    "have already been denoised.",
    "func.gii": "need fMRIprep output with extension func.gii",
}


def check_params_confounds(confounds_raw, params):
    """Check that specified parameters can be found in the confounds.

    Used for motion, wm_csf, global_signal, and compcor regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        Raw confounds loaded from the confounds file.

    params : :obj:`list` of :obj:`str`
        List of parameters constructed based on users choices.

    Returns
    -------
    bool or :obj:`list` of :obj:`str`
        True if all parameters are found in the confounds.
        False if none of the parameters are found in the confounds.
        List of parameters that are not found in the confounds
        if only some parameters are found.
    """
    not_found_params = [
        par for par in params if par not in confounds_raw.columns
    ]
    if len(not_found_params) == len(params):
        return False
    elif not_found_params:
        return not_found_params
    else:
        return True


def find_confounds(confounds_raw, keywords):
    """Find confounds that contain certain keywords.

    Used for cosine regressors and ICA-AROMA regressors.

    Parameters
    ----------
    confounds_raw : pandas.DataFrame
        Raw confounds loaded from the confounds file.

    keywords : :obj:`list` of :obj:`str`
        List of keywords to search for in the confounds.

    Returns
    -------
    list of :obj:`str`
        List of confounds that contain the keywords.
    """
    list_confounds = []
    for key in keywords:
        key_found = [col for col in confounds_raw.columns if key in col]
        if key_found:
            list_confounds.extend(key_found)
    return list_confounds


def sanitize_confounds(img_files):
    """Make sure the inputs are in the correct format.

    Parameters
    ----------
    img_files : :obj:`str` or :obj:`list` of :obj:`str`
        Path to the functional image file(s).

    Returns
    -------
    img_files : :obj:`list` of :obj:`str`
        List of functional image file(s).
    flag_single : bool
        True if the input is a single file, False if it is a :obj:`list` of
        files.
    """
    # we want to support loading a single set of confounds, instead of a list
    # so we hack it
    if len(img_files) == 1:
        return img_files, True
    # gifti has to be passed as pair
    if isinstance(img_files, list) and len(img_files) == 2:
        flag_single = flag_single_gifti(img_files)
    else:  # single file
        flag_single = isinstance(img_files, str)
    if flag_single:
        img_files = [img_files]
    return img_files, flag_single


def add_suffix(params, model):
    """Add derivative suffixes to a list of parameters.

    Used from motion, wm_csf, global_signal.

    Parameters
    ----------
    params : :obj:`list` of :obj:`str`
        List of parameters to add suffixes to.
    model : :obj:`str`
        Model to use. Options are "basic", "derivatives", "power2", or
        "full".

    Returns
    -------
    params_full : :obj:`list` of :obj:`str`
        List of parameters with suffixes added.
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


def _generate_confounds_file_candidates(nii_file, flag_tedana=False):
    """Generate confounds file candidates.

    Build a list of potential confounds filenames using all combinations of
    the entities in the image file.

    Parameters
    ----------
    nii_file : str
        Path to the functional image file.
    flag_tedana : bool, optional
        If True, also generate candidates with desc=ICA for TEDANA
        optimally combined output. Defaults to False.

    Returns
    -------
    filenames : list of str
        List of potential confounds filenames.
    """
    parsed_file = parse_bids_filename(nii_file)
    entities = parsed_file["entities"]

    variants = []

    # Standard confounds
    entities_fmriprep = deepcopy(entities)
    entities_fmriprep["desc"] = "confounds"
    variants.append(entities_fmriprep)

    if flag_tedana:
        # ICA  mixing and tedana
        entities_tedana = deepcopy(entities)
        entities_tedana["desc"] = ["ICA", "tedana"]
        variants.append(entities_tedana)

    filenames = []

    for variant in variants:
        # make sure “desc” is iterable
        desc_values = (
            [variant["desc"]]
            if isinstance(variant["desc"], str)
            else variant["desc"]
        )

        # Allows us to generate filenames separately for each
        # desc value (“confounds”, “ICA”, “tedana”).
        for desc_value in desc_values:
            variant_with_desc = variant.copy()
            variant_with_desc["desc"] = desc_value

            all_subsets = [
                list(itertools.combinations(variant_with_desc.keys(), n))
                for n in range(1, len(variant_with_desc.keys()) + 1)
            ]
            # Flatten the list of lists
            all_subsets = [list(x) for sublist in all_subsets for x in sublist]
            # https://stackoverflow.com/a/3724558/2589328
            unique_subsets = [list(x) for x in {tuple(x) for x in all_subsets}]
            # Require "desc"
            subset_with_desc = [
                subset for subset in unique_subsets if "desc" in subset
            ]

            # Prevents overwriting the list on each iteration;
            # instead, accumulates all filename variants.
            filenames.extend(
                "_".join(f"{k}-{variant_with_desc[k]}" for k in subset)
                for subset in subset_with_desc
            )

    return filenames


def _get_file_name(nii_file, flag_tedana=False):
    """Identify the confounds file associated with a functional image.

    Parameters
    ----------
    nii_file : str
        Path to the functional image file.

    flag_tedana : bool, optional
        If True, look for TEDANA confounds files. Defaults to False.

    Returns
    -------
    confound_file : str
        Path to the associated confounds file.
    """
    if isinstance(nii_file, list):  # catch gifti
        nii_file = nii_file[0]

    base_dir = Path(nii_file).parent

    filenames = _generate_confounds_file_candidates(
        nii_file, flag_tedana=flag_tedana
    )

    # fmriprep has changed the file suffix between v20.1.1 and v20.2.0 with
    # respect to BEP 012.
    # cf. https://neurostars.org/t/naming-change-confounds-regressors-to-confounds-timeseries/17637 # noqa: E501
    # Check file with new naming scheme exists or replace,
    # for backward compatibility.
    suffixes = ["_timeseries.tsv", "_regressors.tsv"]
    if flag_tedana:  # tedana has different suffixes
        suffixes = ["_mixing.tsv", "_metrics.tsv"]

    confound_file_candidates = []
    for suffix in suffixes:
        confound_file_candidates += [f + suffix for f in filenames]

    # Sort the potential filenames by decreasing length,
    # so earlier entries reflect more retained entities.
    # https://www.geeksforgeeks.org/python-sort-list-of-lists-by-the-size-of-sublists/
    confound_file_candidates = sorted(confound_file_candidates, key=len)[::-1]
    confound_file_candidates = [
        base_dir / crc for crc in confound_file_candidates
    ]
    found_files = [str(cr) for cr in confound_file_candidates if cr.is_file()]

    if not found_files:
        raise ValueError(
            "Could not find associated confound file. "
            "The functional derivatives should exist under the same parent "
            "directory."
        )
    elif len(found_files) != 1 and not flag_tedana:
        found_str = "\n\t".join(found_files)
        raise ValueError(f"Found more than one confound file:\n\t{found_str}")
    elif len(found_files) != 2 and flag_tedana:
        found_str = "\n\t".join(found_files)
        raise ValueError(
            f"Found {len(found_files)} confound files "
            f"(expected 2 for TEDANA):\n\t{found_str}\n\n"
            "TEDANA should produce exactly two confound files:\n"
            "- mixing.tsv\n"
            "- table_status.tsv"
        )
    elif flag_tedana:
        return found_files

    return found_files[0]


def get_confounds_file(image_file, flag_full_aroma, flag_tedana):
    """Return the confounds file associated with a functional image.

    Parameters
    ----------
    image_file : :obj:`str`
        Path to the functional image file.

    flag_full_aroma : :obj:`bool`
        True if the input is a full ICA-AROMA output, False otherwise.

    flag_tedata : :obj:`bool`
        True if the input is a TEDANA optimally combined output,
        False otherwise.

    Returns
    -------
    confounds_raw_path : :obj:`str`
        Path to the associated confounds file.
    """
    _check_images(image_file, flag_full_aroma, flag_tedana)
    confounds_raw_path = _get_file_name(image_file, flag_tedana=flag_tedana)
    return confounds_raw_path


def get_json(confounds_raw_path, flag_tedana=False):
    """Return json data companion file to the confounds tsv file."""
    if flag_tedana:
        # TEDANA does not have a json confound companion file
        return None
    # Load JSON file
    return str(confounds_raw_path).replace("tsv", "json")


def load_confounds_json(confounds_json, flag_acompcor):
    """Load json data companion to the confounds tsv file.

    Parameters
    ----------
    confounds_json : :obj:`str`
        Path to the json file.

    flag_acompcor : :obj:`bool`
        True if user selected anatomical compcor for denoising strategy,
        False otherwise.

    Returns
    -------
    confounds_json : dict
        Dictionary of confounds meta data from the confounds.json file.

    Raises
    ------
    ValueError
        If the json file is not found. This should not be the case for
        fMRIprep >= 1.4.0.
    """
    try:
        with Path(confounds_json).open("rb") as f:
            confounds_json = json.load(f)
    except OSError:
        if flag_acompcor:
            raise ValueError(
                f"Could not find associated json file {confounds_json}."
                "This is necessary for anatomical CompCor."
                "The CompCor component is only supported for fMRIprep "
                "version >= 1.4.0."
            )
    return confounds_json


def load_confounds_file_as_dataframe(confounds_raw_path, flag_tedana=False):
    """Load raw confounds as a pandas DataFrame.

    Meanwhile detect if the fMRIPrep version is supported.

    Parameters
    ----------
    confounds_raw_path : :obj:`str` or :obj:`list`
        Path to the confounds file or List of paths for tedana.

    flag_tedana : :obj:`bool`
        True if the input is a TEDANA optimally combined output, False
        otherwise.

    Returns
    -------
    confounds_raw : pandas.DataFrame
        Raw confounds loaded from the confounds file.
    """
    if flag_tedana:
        # TEDANA outputs are not camel case, but they have a different
        # header format.
        confounds_tedana_raw = {}
        for tedana_conf in ["mixing", "metrics"]:
            confounds_tedana_raw[tedana_conf] = pd.read_csv(
                next(
                    file for file in confounds_raw_path if tedana_conf in file
                ),
                delimiter="\t",
                encoding="utf-8",
            )
        if any(
            col.startswith("ICA_")
            for confounds_raw in confounds_tedana_raw.values()
            for col in confounds_raw
        ) or any(
            "Component" in confounds_raw.columns
            for confounds_raw in confounds_tedana_raw.values()
        ):
            return confounds_tedana_raw
        else:
            raise ValueError(
                "The confound file does not contain the expected columns for "
                "TEDANA output. Expected 'ICA_xx' for mixing.tsv and"
                "'Component' for the metrics.tsv columns."
            )
    confounds_raw = pd.read_csv(
        confounds_raw_path, delimiter="\t", encoding="utf-8"
    )
    # check if the version of fMRIprep (>=1.2.0) is supported based on
    # header format. 1.0.x and 1.1.x series uses camel case
    if any(is_camel_case(col_name) for col_name in confounds_raw.columns):
        raise ValueError(
            "The confound file contains header in camel case. "
            "This is likely the output from 1.0.x and 1.1.x series. "
            "We only support fmriprep outputs >= 1.2.0."
            f"{confounds_raw.columns}"
        )

    # even old version with no header will have the first row as header
    try:
        too_old = float(confounds_raw.columns[0])
    except ValueError:
        too_old = False

    if too_old:
        bad_file = pd.read_csv(
            confounds_raw_path, delimiter="\t", encoding="utf-8", header=None
        )
        raise ValueError(
            "The confound file contains no header."
            "Is this an old version fMRIprep output?"
            f"{bad_file.head()}"
        )

    return confounds_raw


def _ext_validator(image_file, ext):
    """Check image is valid based on extension.

    Parameters
    ----------
    image_file : str
        Path to the functional image file.

    ext : str
        Extension to check.

    Returns
    -------
    valid_img : bool
        True if the image is valid, False otherwise.

    error_message : str
        Error message to raise if the image is invalid.
    """
    try:
        valid_img = all(
            bool(re.search(img_file_patterns[ext], img)) for img in image_file
        )
        error_message = img_file_error[ext]
    except KeyError:
        valid_img = False
        error_message = "Unsupported input."
    return valid_img, error_message


def _check_images(image_file, flag_full_aroma, flag_tedana):
    """Validate input file and ICA AROMA related file.

    Parameters
    ----------
    image_file : str
        Path to the functional image file.

    flag_full_aroma : bool
        True if the input is a full ICA-AROMA output, False otherwise.

    flag_tedata : :obj:`bool`
        True if the input is a TEDANA optimally combined output

    Raises
    ------
    ValueError
        If the image is not valid.
    """
    if len(image_file) == 2:  # must be gifti
        valid_img, error_message = _ext_validator(image_file, "func.gii")
    elif flag_full_aroma:
        valid_img, error_message = _ext_validator([image_file], "aroma")
    elif flag_tedana:
        valid_img, error_message = _ext_validator([image_file], "tedana")
    else:
        ext = ".".join(image_file.split(".")[-2:])
        valid_img, error_message = _ext_validator([image_file], ext)
    if not valid_img:
        raise ValueError(error_message)


def prepare_output(confounds, demean):
    """Demean and create sample mask for the selected confounds.

    Parameters
    ----------
    confounds : pandas.DataFrame
        Confound regressors loaded based on user's choice.

    demean : :obj:`bool`
        True if the confounds should be demeaned, False otherwise.

    Returns
    -------
    sample_mask : None or numpy.ndarray
        When no volume removal is required, the value is None.
        Otherwise, the shape is \
            (number of scans - number of volumes removed, )
        The index of the niimgs along time/fourth dimension for valid
        volumes for subsequent analysis.

    confounds : pandas.DataFrame
        Demeaned confounds ready for subsequent analysis.
    """
    sample_mask, confounds, _ = extract_outlier_regressors(confounds)
    if confounds.size != 0:  # ica_aroma = "full" generate empty output
        # Derivatives have NaN on the first row
        # Replace them by estimates at second time point,
        # otherwise nilearn will crash.
        mask_nan = np.isnan(confounds.to_numpy()[0, :])
        confounds.iloc[0, mask_nan] = confounds.iloc[1, mask_nan]
        if demean:
            confounds = _demean_confounds(confounds, sample_mask)
    return sample_mask, confounds


def _demean_confounds(confounds, sample_mask):
    """Demean the confounds.

    The mean is calculated on non-outlier values.

    Parameters
    ----------
    confounds : pandas.DataFrame
        Confound regressors loaded based on user's choice.

    sample_mask : None or numpy.ndarray
        When no volume removal is required, the value is None.
        Otherwise, the shape is \
            (number of scans - number of volumes removed, )
        The index of the niimgs along time/fourth dimension for valid
        volumes for subsequent analysis.

    Returns
    -------
    confounds : pandas.DataFrame
        Demeaned confounds.
    """
    confound_cols = confounds.columns
    if sample_mask is None:
        confounds = scale(confounds, axis=0, with_std=False)
    else:  # calculate the mean without outliers.
        confounds_mean = confounds.iloc[sample_mask, :].mean(axis=0)
        confounds -= confounds_mean
    return pd.DataFrame(confounds, columns=confound_cols)


class MissingConfoundError(Exception):
    """
    Exception raised when failing to find params in the confounds.

    Parameters
    ----------
    params : :obj:`list` of missing params, default=[]

    keywords : :obj:`list` of missing keywords, default=[]
    """

    def __init__(self, params=None, keywords=None):
        """Set missing parameters and keywords."""
        self.params = params or []
        self.keywords = keywords or []
