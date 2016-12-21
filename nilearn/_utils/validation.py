"""
Utilities to check for valid instances
"""
import warnings

from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.feature_selection import (SelectPercentile, f_regression,
                                       f_classif)

from .param_validation import adjust_screening_percentile
from .compat import _basestring
from ..input_data import NiftiMasker, MultiNiftiMasker


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
