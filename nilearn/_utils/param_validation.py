"""
Utilities to check for valid parameters
"""
import numpy as np
import warnings
import numbers

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
    vox_dims = mask_img.get_header().get_zooms()[:3]
    return 1. * np.prod(vox_dims) * mask_img.get_data().astype(np.bool).sum()


def _adjust_screening_percentile(screening_percentile, mask_img,
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
