"""Utilities to check for valid parameters."""

import numbers
import sys
import warnings

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

import nilearn.typing as nilearn_typing
from nilearn._utils import logger
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import _get_data
from nilearn.surface import SurfaceImage

# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1882989.0


def check_threshold(
    threshold, data, percentile_func, name="threshold", two_sided=True
):
    """Check if the given threshold is in correct format and within the limit.

    If threshold is string, this function returns score of the data calculated
    based upon the given specific percentile function.

    Parameters
    ----------
    threshold : :obj:`float` or :obj:`str`
        Threshold that is used to set certain data values to zero.
        If threshold is float, it should be within the range of minimum and the
        maximum intensity of the data.
        If `two_sided` is True, threshold cannot be negative.
        If threshold is str, the given string should be within the range of
        "0%" to "100%".

    data : ndarray
        An array of the input masked data.

    percentile_func : function {scoreatpercentile, fast_abs_percentile}
        Percentile function for example scipy.stats.scoreatpercentile
        to calculate the score on the data.

    name : :obj:`str`, default='threshold'
        A string just used for representing the name of the threshold for a
        precise error message.

    two_sided : :obj:`bool`, default=True
        Whether the thresholding should yield both positive and negative
        part of the maps.

        .. versionadded:: 0.12.0

    Returns
    -------
    threshold : :obj:`float`
        Returns the score of the percentile on the data or returns threshold as
        it is if given threshold is not a string percentile.

    Raises
    ------
    ValueError
        If threshold is of type str but is not a non-negative number followed
        by the percent sign.
        If threshold is a negative float and `two_sided` is True.
    TypeError
        If threshold is neither float nor a string in correct percentile
        format.
    """
    percentile = False
    if isinstance(threshold, str):
        message = (
            f'If "{name}" is given as string it '
            "should be a number followed by the percent "
            'sign, e.g. "25.3%"'
        )
        if not threshold.endswith("%"):
            raise ValueError(message)
        try:
            threshold = float(threshold[:-1])
            percentile = True
        except ValueError as exc:
            exc.args += (message,)
            raise
    elif not isinstance(threshold, numbers.Real):
        raise TypeError(
            f"{name} should be either a number "
            "or a string finishing with a percent sign"
        )

    if threshold >= 0:
        data = abs(data) if two_sided else np.extract(data >= 0, data)

        if percentile:
            threshold = percentile_func(data, threshold)
        else:
            value_check = data.max()
            if threshold > value_check:
                warnings.warn(
                    f"The given float value must not exceed {value_check}. "
                    f"But, you have given threshold={threshold}.",
                    category=UserWarning,
                    stacklevel=find_stack_level(),
                )
    else:
        if two_sided:
            raise ValueError(
                f'"{name}" should not be a negative value when two_sided=True.'
            )
        if percentile:
            raise ValueError(
                f'"{name}" should not be a negative percentile value.'
            )
        data = np.extract(data <= 0, data)
        value_check = data.min()
        if threshold < value_check:
            warnings.warn(
                f"The given float value must not be less than "
                f"{value_check}. But, you have given "
                f"threshold={threshold}.",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )

    return threshold


def _get_mask_extent(mask_img):
    """Compute the extent of the provided brain mask.
    The extent is the volume of the mask in mm^3 if mask_img is a Nifti1Image
    or the number of vertices if mask_img is a SurfaceImage.

    Parameters
    ----------
    mask_img : Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    Returns
    -------
    mask_extent : float
        The computed volume in mm^3 (if mask_img is a Nifti1Image) or the
        number of vertices (if mask_img is a SurfaceImage).

    """
    if not hasattr(mask_img, "affine"):
        # sum number of True values in both hemispheres
        return (
            mask_img.data.parts["left"].sum()
            + mask_img.data.parts["right"].sum()
        )
    affine = mask_img.affine
    prod_vox_dims = 1.0 * np.abs(np.linalg.det(affine[:3, :3]))
    return prod_vox_dims * _get_data(mask_img).astype(bool).sum()


@fill_doc
def adjust_screening_percentile(screening_percentile, mask_img, verbose=0):
    """Adjust the screening percentile according to the MNI152 template or
    the number of vertices of the provided standard brain mesh.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img :  Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    %(verbose0)s

    Returns
    -------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection.

    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    # or the number of vertices of the reference mesh
    mask_extent = _get_mask_extent(mask_img)
    # if mask_img is a surface mesh, reference is the number of vertices
    # in the standard mesh otherwise it is the volume of the MNI152 brain
    # template
    reference_extent = (
        mask_img.mesh.n_vertices
        if isinstance(mask_img, SurfaceImage)
        else MNI152_BRAIN_VOLUME
    )
    if mask_extent > 1.1 * reference_extent:
        unit = "mm^3"
        if hasattr(mask_img, "mesh"):
            unit = "vertices"
        warnings.warn(
            f"Brain mask ({mask_extent} {unit}) is bigger than the standard "
            f"human brain ({reference_extent} {unit})."
            "This object is probably not tuned to be used on such data.",
            stacklevel=find_stack_level(),
        )
    elif mask_extent < 0.005 * reference_extent:
        warnings.warn(
            "Brain mask is smaller than .5% of the size of the standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=find_stack_level(),
        )

    if screening_percentile < 100.0:
        screening_percentile = screening_percentile * (
            reference_extent / mask_extent
        )
        screening_percentile = min(screening_percentile, 100.0)
    # if screening_percentile is 100, we don't do anything

    if hasattr(mask_img, "mesh"):
        log_mask = f"Mask n_vertices = {mask_extent:g}"
    else:
        log_mask = (
            f"Mask volume = {mask_extent:g}mm^3 = {mask_extent / 1000.0:g}cm^3"
        )
    logger.log(
        log_mask,
        verbose=verbose,
        msg_level=1,
    )
    if hasattr(mask_img, "mesh"):
        log_ref = f"Reference mesh n_vertices = {reference_extent:g}"
    else:
        log_ref = f"Standard brain volume = {MNI152_BRAIN_VOLUME:g}mm^3"
    logger.log(
        log_ref,
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Original screening-percentile: {original_screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Corrected screening-percentile: {screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    return screening_percentile


@fill_doc
def check_feature_screening(
    screening_percentile, mask_img, is_classification, verbose=0
):
    """Check feature screening method.

    Turns floats between 1 and 100 into SelectPercentile objects.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img : nibabel image object
        Input image whose :term:`voxel` dimensions are to be computed.

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    %(verbose0)s

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
        # correct screening_percentile according to the volume or the number of
        # vertices in the data mask
        effective_screening_percentile = adjust_screening_percentile(
            screening_percentile,
            mask_img,
            verbose=verbose,
        )

        if effective_screening_percentile == 100:
            warnings.warn(
                f"screening_percentile set to '100' despite "
                f"requesting '{screening_percentile=}'. "
                "\nAll elements in the mask will be included. "
                "\nThis usually occurs when the mask image "
                "is too small compared to full brain mask.",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )

        return SelectPercentile(
            f_test, percentile=int(effective_screening_percentile)
        )


def check_run_sample_masks(n_runs, sample_masks):
    """Check that number of sample_mask matches number of runs."""
    if not isinstance(sample_masks, (list, tuple, np.ndarray)):
        raise TypeError(
            f"sample_mask has an unhandled type: {sample_masks.__class__}"
        )

    if isinstance(sample_masks, np.ndarray):
        sample_masks = (sample_masks,)

    checked_sample_masks = [_convert_bool2index(sm) for sm in sample_masks]
    checked_sample_masks = [_cast_to_int32(sm) for sm in checked_sample_masks]

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


def _cast_to_int32(sample_mask):
    """Ensure the sample mask dtype is signed."""
    new_dtype = np.int32
    if np.min(sample_mask) < 0:
        msg = "sample_mask should not contain negative values."
        raise ValueError(msg)

    if highest := np.max(sample_mask) > np.iinfo(new_dtype).max:
        msg = f"Max value in sample mask is larger than \
            what can be represented by int32: {highest}."
        raise ValueError(msg)
    return np.asarray(sample_mask, new_dtype)


# dictionary that matches a given parameter / attribute name to a type
TYPE_MAPS = {
    "annotate": nilearn_typing.Annotate,
    "border_size": nilearn_typing.BorderSize,
    "bg_on_data": nilearn_typing.BgOnData,
    "colorbar": nilearn_typing.ColorBar,
    "connected": nilearn_typing.Connected,
    "data_dir": nilearn_typing.DataDir,
    "draw_cross": nilearn_typing.DrawCross,
    "detrend": nilearn_typing.Detrend,
    "high_pass": nilearn_typing.HighPass,
    "hrf_model": nilearn_typing.HrfModel,
    "keep_masked_labels": nilearn_typing.KeepMaskedLabels,
    "keep_masked_maps": nilearn_typing.KeepMaskedMaps,
    "low_pass": nilearn_typing.LowPass,
    "lower_cutoff": nilearn_typing.LowerCutoff,
    "memory": nilearn_typing.MemoryLike,
    "memory_level": nilearn_typing.MemoryLevel,
    "n_jobs": nilearn_typing.NJobs,
    "n_perm": nilearn_typing.NPerm,
    "opening": nilearn_typing.Opening,
    "radiological": nilearn_typing.Radiological,
    "random_state": nilearn_typing.RandomState,
    "resolution": nilearn_typing.Resolution,
    "resume": nilearn_typing.Resume,
    "screening_percentile": nilearn_typing.ScreeningPercentile,
    "smoothing_fwhm": nilearn_typing.SmoothingFwhm,
    "standardize_confounds": nilearn_typing.StandardizeConfounds,
    "t_r": nilearn_typing.Tr,
    "tfce": nilearn_typing.Tfce,
    "threshold": nilearn_typing.Threshold,
    "title": nilearn_typing.Title,
    "two_sided_test": nilearn_typing.TwoSidedTest,
    "target_affine": nilearn_typing.TargetAffine,
    "target_shape": nilearn_typing.TargetShape,
    "transparency": nilearn_typing.Transparency,
    "transparency_range": nilearn_typing.TransparencyRange,
    "url": nilearn_typing.Url,
    "upper_cutoff": nilearn_typing.UpperCutoff,
    "verbose": nilearn_typing.Verbose,
    "vmax": nilearn_typing.Vmax,
    "vmin": nilearn_typing.Vmin,
}


def check_params(fn_dict):
    """Check types of inputs passed to a function / method / class.

    This function checks the types of function / method parameters or type_map
    the attributes of the class.

    This function is made to check the types of the parameters
    described in ``nilearn._utils.docs``
    that are shared by many functions / methods / class
    and thus ensure a generic way to do input validation
    in several important points in the code base.

    In most cases this means that this function can be used
    on functions / classes that have the ``@fill_doc`` decorator,
    or whose doc string uses parameter templates
    (for example ``%(data_dir)s``).

    If the function cannot (yet) check any of the parameters / attributes,
    it will throw an error to say that its use is not needed.

    Typical usage:

    .. code-block:: python

        def some_function(param_1, param_2="a"):
            check_params(locals())
            ...

        Class MyClass:
            def __init__(param_1, param_2="a")
            ...

            def fit(X):
                # check attributes of the class instance
                check_params(self.__dict__)
                # check parameters passed to the method
                check_params(locals())

    """
    keys_to_check = set(TYPE_MAPS.keys()).intersection(set(fn_dict.keys()))
    # Send a message to dev if they are using this function needlessly.
    if not keys_to_check:
        raise ValueError(
            "No known parameter to check.\n"
            "You probably do not need to use 'check_params' here."
        )

    for k in keys_to_check:
        type_to_check = TYPE_MAPS[k]
        value = fn_dict[k]

        # TODO (python 3.10) update when dropping python 3.9
        error_msg = (
            f"'{k}' should be of type '{type_to_check}'.\nGot: '{type(value)}'"
        )
        if sys.version_info[1] > 9:
            if not isinstance(value, type_to_check):
                raise TypeError(error_msg)
        elif value is not None and not isinstance(value, type_to_check):
            raise TypeError(error_msg)


def check_reduction_strategy(strategy: str):
    """Check that the provided strategy is supported.

    Parameters
    ----------
    %(strategy)s
    """
    available_reduction_strategies = {
        "mean",
        "median",
        "sum",
        "minimum",
        "maximum",
        "standard_deviation",
        "variance",
    }

    if strategy not in available_reduction_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Valid strategies are {available_reduction_strategies}."
        )
