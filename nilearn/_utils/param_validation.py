"""Utilities to check for valid parameters."""

import numbers
import warnings

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

from nilearn._utils import logger

from .niimg import _get_data

# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1827243.0


def check_threshold(threshold, data, percentile_func, name="threshold"):
    """Check if the given threshold is in correct format \
    and within the limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold : float or str
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with
        a percent sign like "99.7%".

    data : ndarray
        An array of the input masked data.

    percentile_func : function {scoreatpercentile, fastabspercentile}
        Percentile function for example scipy.stats.scoreatpercentile
        to calculate the score on the data.

    name : str, default='threshold'
        A string just used for representing
        the name of the threshold for a precise
        error message.

    Returns
    -------
    threshold : number
        Returns the score of the percentile on the data or
        returns threshold as it is
        if given threshold is not a string percentile.

    """
    if isinstance(threshold, str):
        message = (
            f'If "{name}" is given as string it '
            "should be a number followed by the percent "
            'sign, e.g. "25.3%"'
        )
        if not threshold.endswith("%"):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message,)
            raise

        threshold = percentile_func(data, percentile)
    elif isinstance(threshold, numbers.Real):
        # checks whether given float value exceeds the maximum
        # value of the image data
        value_check = abs(data).max()
        if abs(threshold) > value_check:
            warnings.warn(
                f"The given float value must not exceed {value_check}. "
                f"But, you have given threshold={threshold}.",
                category=UserWarning,
                stacklevel=3,
            )
    else:
        raise TypeError(
            f"{name} should be either a number "
            "or a string finishing with a percent sign"
        )
    return threshold


def get_mask_volume(mask_img):
    """Compute the volume of a brain mask in mm^3.

    Parameters
    ----------
    mask_img : nibabel image object
        Input image whose voxel dimensions are to be computed.

    Returns
    -------
    vol : float
        The computed volume.

    """
    affine = mask_img.affine
    prod_vox_dims = 1.0 * np.abs(np.linalg.det(affine[:3, :3]))
    return prod_vox_dims * _get_data(mask_img).astype(bool).sum()


def adjust_screening_percentile(screening_percentile, mask_img, verbose=0):
    """Adjust the screening percentile according to the MNI152 template.

    Parameters
    ----------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.

    mask_img : nibabel image object
        Input image whose voxel dimensions are to be computed.

    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    screening_percentile: float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection.

    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    mask_volume = get_mask_volume(mask_img)
    if mask_volume > 1.1 * MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is bigger than the volume of a standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=3,
        )
    elif mask_volume < 0.005 * MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is smaller than .5% of the volume "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=3,
        )

    if screening_percentile < 100.0:
        screening_percentile = screening_percentile * (
            MNI152_BRAIN_VOLUME / mask_volume
        )
        screening_percentile = min(screening_percentile, 100.0)
    # if screening_percentile is 100, we don't do anything

    logger.log(
        f"Mask volume = {mask_volume:g}mm^3 = {mask_volume / 1000.0:g}cm^3",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        "Standard brain volume "
        f"= {MNI152_BRAIN_VOLUME:g}mm^3 "
        f"= {MNI152_BRAIN_VOLUME / 1.0e3:g}cm^3",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Original screening-percentile: {original_screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Volume-corrected screening-percentile: {screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    return screening_percentile


def check_feature_screening(
    screening_percentile, mask_img, is_classification, verbose=0
):
    """Check feature screening method.

    Turns floats between 1 and 100 into SelectPercentile objects.

    Parameters
    ----------
    screening_percentile : float in the interval [0, 100]
        Percentile value for :term:`ANOVA` univariate feature selection.
        A value of 100 means 'keep all features'.
        This percentile is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.

    mask_img : nibabel image object
        Input image whose :term:`voxel` dimensions are to be computed.

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    selector : SelectPercentile instance
       Used to perform the :term:`ANOVA` univariate feature selection.

    """
    f_test = f_classif if is_classification else f_regression

    if screening_percentile == 100 or screening_percentile is None:
        return None
    elif not (0.0 <= screening_percentile <= 100.0):
        raise ValueError(
            "screening_percentile should be in the interval"
            f" [0, 100], got {screening_percentile:g}"
        )
    else:
        # correct screening_percentile according to the volume of the data mask
        screening_percentile_ = adjust_screening_percentile(
            screening_percentile, mask_img, verbose=verbose
        )

        return SelectPercentile(f_test, percentile=int(screening_percentile_))


def check_run_sample_masks(n_runs, sample_masks):
    """Check that number of sample_mask matches number of runs."""
    if not isinstance(sample_masks, (list, tuple, np.ndarray)):
        raise TypeError(
            f"sample_mask has an unhandled type: {sample_masks.__class__}"
        )

    if isinstance(sample_masks, np.ndarray):
        sample_masks = (sample_masks,)

    checked_sample_masks = [_convert_bool2index(sm) for sm in sample_masks]

    if len(checked_sample_masks) != n_runs:
        raise ValueError(
            f"Number of sample_mask ({len(checked_sample_masks)}) not "
            f"matching number of runs ({n_runs})."
        )
    return checked_sample_masks


def _convert_bool2index(sample_mask):
    """Convert boolean to index."""
    check_boolean = [
        type(i) is bool or type(i) is np.bool_ for i in sample_mask
    ]
    if all(check_boolean):
        sample_mask = np.where(sample_mask)[0]
    return sample_mask
