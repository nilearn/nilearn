"""Utilities to check for valid parameters."""

import numbers
import warnings
from collections.abc import Iterable
from typing import Any, Literal, get_args, get_origin

import numpy as np

import nilearn.typing as nilearn_typing
from nilearn._utils.logger import find_stack_level


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

        .. nilearn_versionadded:: 0.12.0

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


def check_run_sample_masks(n_runs, sample_masks):
    """Check that number of sample_mask matches number of runs."""
    check_is_of_allowed_type(
        sample_masks, (list, tuple, np.ndarray), "sample_masks"
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
    "cluster_threshold": nilearn_typing.ClusterThreshold,
    "connected": nilearn_typing.Connected,
    "copy_header": nilearn_typing.CopyHeader,
    "data_dir": nilearn_typing.DataDir,
    "draw_cross": nilearn_typing.DrawCross,
    "detrend": nilearn_typing.Detrend,
    "force_resample": nilearn_typing.ForceResample,
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
    "standardize": nilearn_typing.Standardize,
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

        if get_origin(type_to_check) is Literal:
            allowed_values = get_args(type_to_check)
            check_parameter_in_allowed(value, allowed_values, k)

        else:
            check_is_of_allowed_type(value, type_to_check, k)


def check_is_of_allowed_type(
    value: Any, type_to_check: tuple[Any] | Any, parameter_name: str
):
    if not isinstance(type_to_check, tuple):
        type_to_check = (type_to_check,)
    if not isinstance(value, type_to_check):
        type_to_check_str = ", ".join([str(x) for x in type_to_check])
        error_msg = (
            f"'{parameter_name}' must be of type(s): '{type_to_check_str}'.\n"
            f"Got: '{value.__class__.__name__}'"
        )
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
    check_parameter_in_allowed(
        strategy, available_reduction_strategies, "strategy"
    )


def check_parameter_in_allowed(
    parameter: Any, allowed: Iterable[Any], parameter_name: str
):
    if parameter not in allowed:
        raise ValueError(
            f"'{parameter_name}' must be one of {allowed}.\n"
            f"'{parameter}' was provided."
        )
