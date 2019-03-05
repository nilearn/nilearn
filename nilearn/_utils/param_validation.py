"""
Utilities to check for valid parameters
"""
import functools

import numpy as np
import warnings
import numbers

from sklearn.feature_selection import (SelectPercentile, f_regression,
                                       f_classif)

from .compat import _basestring


# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1827243.


def check_threshold(threshold, data, percentile_func, name='threshold'):
    """ Checks if the given threshold is in correct format and within the
    limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold: float or str
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with
        a percent sign like "99.7%".
    data: ndarray
        an array of the input masked data.
    percentile_func: function {scoreatpercentile, fastabspercentile}
        Percentile function for example scipy.stats.scoreatpercentile
        to calculate the score on the data.
    name: str, optional
        A string just used for representing the name of the threshold for a precise
        error message.

    Returns
    -------
    threshold: number
        returns the score of the percentile on the data or
        returns threshold as it is if given threshold is not a string percentile.
    """
    if isinstance(threshold, _basestring):
        message = ('If "{0}" is given as string it '
                   'should be a number followed by the percent '
                   'sign, e.g. "25.3%"').format(name)
        if not threshold.endswith('%'):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message, )
            raise

        threshold = percentile_func(data, percentile)
    elif isinstance(threshold, numbers.Real):
        # checks whether given float value exceeds the maximum
        # value of the image data
        value_check = abs(data).max()
        if abs(threshold) > value_check:
            warnings.warn("The given float value must not exceed {0}. "
                          "But, you have given threshold={1} ".format(value_check,
                                                                      threshold))
    else:
        raise TypeError('%s should be either a number '
                        'or a string finishing with a percent sign' % (name, ))
    return threshold


def _get_mask_volume(mask_img):
    """Computes the volume of a brain mask in mm^3

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
    prod_vox_dims = 1. * np.abs(np.linalg.det(affine[:3, :3]))
    return prod_vox_dims * mask_img.get_data().astype(np.bool).sum()


def _adjust_screening_percentile(screening_percentile, mask_img,
                                 verbose=0):
    """Adjusts the screening percentile according to the MNI152 template.

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

    verbose : int, optional (default 0)
        Verbosity level.

    Returns
    -------
    screening_percentile: float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection.
    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    mask_volume = _get_mask_volume(mask_img)
    if mask_volume > 1.1 * MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is bigger than the volume of a standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.", stacklevel=2)
    elif mask_volume < .005 * MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is smaller than .5% of the volume "
            "human brain. This object is probably not tuned to"
            "be used on such data.", stacklevel=2)

    if screening_percentile < 100.:
        screening_percentile = screening_percentile * (
            MNI152_BRAIN_VOLUME / mask_volume)
        screening_percentile = min(screening_percentile, 100.)
    # if screening_percentile is 100, we don't do anything

    if verbose > 1:
        print("Mask volume = %gmm^3 = %gcm^3" % (
            mask_volume, mask_volume / 1.e3))
        print("Standard brain volume = %gmm^3 = %gcm^3" % (
            MNI152_BRAIN_VOLUME, MNI152_BRAIN_VOLUME / 1.e3))
        print("Original screening-percentile: %g" % (
            original_screening_percentile))
        print("Volume-corrected screening-percentile: %g" % (
            screening_percentile))
    return screening_percentile


def check_feature_screening(screening_percentile, mask_img,
                            is_classification, verbose=0):
    """Check feature screening method. Turns floats between 1 and 100 into
    SelectPercentile objects.

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

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    verbose : int, optional (default 0)
        Verbosity level.

    Returns
    -------
    selector : SelectPercentile instance
       Used to perform the ANOVA univariate feature selection.
    """

    f_test = f_classif if is_classification else f_regression

    if screening_percentile == 100 or screening_percentile is None:
        return None
    elif not (0. <= screening_percentile <= 100.):
        raise ValueError(
            ("screening_percentile should be in the interval"
             " [0, 100], got %g" % screening_percentile))
    else:
        # correct screening_percentile according to the volume of the data mask
        screening_percentile_ = _adjust_screening_percentile(
            screening_percentile, mask_img, verbose=verbose)

        return SelectPercentile(f_test, int(screening_percentile_))


def replace_parameters(replacement_params, end_version, lib_name='Nilearn'):
    """
    Decorator to deprecate & replace specificied parameters
    in the decorated functions and methods.
    
    Add **kwargs as the last parameter in the decorated method/function.
    
    Parameters
    ----------
    replacement_params : Dict[string, string]
        Dict where the key-value pairs represent the old parameters
        and their corresponding new parameters.
        Example: {old_param1: new_param1, old_param2: new_param2,...}
        
    end_version : str
        Version when the deprecated parameters will cease functioning
        and no more warnings will be displayed.
        Default: None / 'future'
        Example: '0.6.0b', 'next'
        
    lib_name: str
        Name of the library to which the decoratee belongs.
        Default: 'Nilearn'
    """
    
    def _replace_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecated_params(replacement_params, end_version, lib_name, kwargs)
            kwargs = _transfer_deprecated_param_vals(replacement_params, kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    return _replace_params


def _warn_deprecated_params(replacement_params, end_version, lib_name, kwargs):
    """ For the decorator replace_parameters(),
        raises warnings about deprecated parameters.
    """
    if end_version is None or end_version == 'future':
        lib_end_ver = 'a future {} version'.format(lib_name)
    elif end_version == 'next':
        lib_end_ver = 'the next {} version'.format(lib_name)
    else:
        lib_end_ver = '{} version {}'.format(lib_name, end_version)
    used_deprecated_params = set(kwargs).intersection(replacement_params)
    for deprecated_param_ in used_deprecated_params:
        replacement_param = replacement_params[deprecated_param_]
        param_deprecation_msg = (
            'The parameter "{}" will be removed in {}. '
            'Please use the parameter "{}" instead.'.format(deprecated_param_,
                                                            lib_end_ver,
                                                            replacement_param,
                                                            )
        )
        warnings.filterwarnings('always', message=param_deprecation_msg)
        warnings.warn(category=DeprecationWarning,
                      message=param_deprecation_msg,
                      stacklevel=3)


def _transfer_deprecated_param_vals(replacement_params, kwargs):
    """ For the decorator replace_parameters(), reassigns new parameters
    the values passed to their corresponding deprecated parameters.
    """
    for old_param, new_param in replacement_params.items():
        old_param_val = kwargs.setdefault(old_param, None)
        if old_param_val is not None:
            kwargs[new_param] = old_param_val
        kwargs.pop(old_param)
    return kwargs
