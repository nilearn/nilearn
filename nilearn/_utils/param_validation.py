"""
Utilities to check for valid parameters
"""

import numpy as np
import warnings
import numbers

from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.feature_selection import (SelectPercentile, f_regression,
                                       f_classif)

from .compat import _basestring
from ..input_data import NiftiMasker, MultiNiftiMasker
from .._utils.compat import _basestring


# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1827243.


def check_threshold(threshold, data, percentile_func, name='threshold'):
    """ Checks if the given threshold is in correct format and within the limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold: float or str
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with a
        percent sign like "99.7%".
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


def check_masking(mask, target_affine=None, target_shape=None,
                  smoothing_fwhm=None, standardize=True,
                  mask_strategy='epi', memory=None, memory_level=1):
    """Setup a nifti masker.
    Parameters
    ----------
    mask : filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a NiftiMasker.

    target_affine : 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. An important use-case
        of this parameter is for downsampling the input data to a coarser
        resolution (to speed of the model fit). Please see the related
        documentation for details.

    target_shape : 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    smoothing_fwhm : float, optional (default None)
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize : bool, optional (default True):
        If set, then the data (X, y) are centered to have mean zero along
        axis 0. This is here because nearly all linear models will want
        their data to be centered.

    mask_strategy: {'background' or 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    Returns
    -------
    masker : NiftiMasker or MultiNiftiMasker instance
        Mask to be used on data.
    """
    # mask is an image, not a masker
    if isinstance(mask, _basestring) or (mask is None):
        masker = NiftiMasker(mask_img=mask,
                             smoothing_fwhm=smoothing_fwhm,
                             target_affine=target_affine,
                             target_shape=target_shape,
                             standardize=standardize,
                             mask_strategy=mask_strategy,
                             memory=memory,
                             memory_level=memory_level)
    # mask is a masker object
    elif isinstance(mask, (NiftiMasker, MultiNiftiMasker)):
        try:
            masker = clone(mask)
            if hasattr(mask, 'mask_img_'):
                mask_img = mask.mask_img_
                masker.set_params(mask_img=mask_img)
                masker.fit()
        except TypeError as e:
            # Workaround for a joblib bug: in joblib 0.6, a Memory object
            # with cachedir = None cannot be cloned.
            masker_memory = mask.memory
            if masker_memory.cachedir is None:
                mask.memory = None
                masker = clone(mask)
                mask.memory = masker_memory
                masker.memory = Memory(cachedir=None)
            else:
                # The error was raised for another reason
                raise e

        for param_name in ['target_affine', 'target_shape',
                           'smoothing_fwhm', 'mask_strategy',
                           'memory', 'memory_level']:
            if getattr(mask, param_name) is not None:
                warnings.warn('Parameter %s of the masker overriden'
                              % param_name)
                masker.set_params(**{param_name: getattr(mask, param_name)})
        if hasattr(mask, 'mask_img_'):
            warnings.warn('The mask_img_ of the masker will be copied')
    return masker


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
    vox_dims = mask_img.get_header().get_zooms()[:3]
    return 1. * np.prod(vox_dims) * mask_img.get_data().astype(np.bool).sum()


def adjust_screening_percentile(screening_percentile, mask_img,
                                verbose=0):
    """Adjusts the screening percentile according to the MNI mask template.
    Parameters
    ----------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection. A value of
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.  If '100' is given,
        all the features are used, regardless of the number of voxels.

    mask_img : nibabel image object
        Input image whose voxel dimensions are to be computed.

    verbose : int, optional (default 0)
        Verbosity level.

    Retruns
    -------
    screening_percentile: float in the interval [0, 100]
        Percentile value for ANOVa univariate feature selection.
    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    mask_volume = _get_mask_volume(mask_img)
    if mask_volume > MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is bigger than the volume of a standard "
            "human brain. SpaceNet is probably not tuned to "
            "be used on such data.", stacklevel=2)
    elif mask_volume < .005 * MNI152_BRAIN_VOLUME:
        warnings.warn(
            "Brain mask is smaller than .5% of the volume "
            "human brain. SpaceNet is probably not tuned to"
            "be used on such data.", stacklevel=2)

    if screening_percentile < 100:
        screening_percentile = screening_percentile * (
            MNI152_BRAIN_VOLUME / mask_volume)
        screening_percentile = min(screening_percentile, 100)
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
        100 means 'keep all features'. This percentile is is expressed
        w.r.t the volume of a standard (MNI152) brain, and so is corrected
        at runtime by premultiplying it with the ratio of the volume of the
        mask of the data and volume of a standard brain.  If '100' is given,
        all the features are used, regardless of the number of voxels.

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
        screening_percentile_ = adjust_screening_percentile(
            screening_percentile, mask_img, verbose=verbose)

        return SelectPercentile(f_test, int(screening_percentile_))

