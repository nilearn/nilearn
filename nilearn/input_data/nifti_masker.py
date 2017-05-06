"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

from copy import copy as copy_object

from sklearn.externals.joblib import Memory

from .base_masker import BaseMasker, filter_and_extract
from .. import _utils
from .. import image
from .. import masking
from .._utils import CacheMixin
from .._utils.class_inspect import get_params
from .._utils.niimg_conversions import _check_same_fov
from .._utils.compat import get_affine


class _ExtractionFunctor(object):
    func_name = 'nifti_masker_extractor'

    def __init__(self, mask_img_):
        self.mask_img_ = mask_img_

    def __call__(self, imgs):
        return masking.apply_mask(imgs, self.mask_img_), get_affine(imgs)


def filter_and_mask(imgs, mask_img_, parameters,
                    memory_level=0, memory=Memory(cachedir=None),
                    verbose=0,
                    confounds=None,
                    copy=True):
    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)

    # Check whether resampling is truly necessary. If so, crop mask
    # as small as possible in order to speed up the process

    if not _check_same_fov(imgs, mask_img_):
        parameters = copy_object(parameters)
        # now we can crop
        mask_img_ = image.crop_img(mask_img_, copy=False)
        parameters['target_shape'] = mask_img_.shape
        parameters['target_affine'] = get_affine(mask_img_)

    data, affine = filter_and_extract(imgs, _ExtractionFunctor(mask_img_),
                                      parameters,
                                      memory_level=memory_level,
                                      memory=memory,
                                      verbose=verbose,
                                      confounds=confounds, copy=copy)

    # For _later_: missing value removal or imputing of missing data
    # (i.e. we want to get rid of NaNs, if smoothing must be done
    # earlier)
    # Optionally: 'doctor_nan', remove voxels with NaNs, other option
    # for later: some form of imputation
    return data


class NiftiMasker(BaseMasker, CacheMixin):
    """Applying a mask to extract time-series from Niimg-like objects.

    NiftiMasker is useful when preprocessing (detrending, standardization,
    resampling, etc.) of in-mask voxels is necessary. Use case: working with
    time series of resting-state or task maps.

    Parameters
    ----------
    mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask for the data. If not given, a mask is computed in the fit step.
        Optional parameters (mask_args and mask_strategy) can be set to
        fine tune the mask extraction.

    sessions : numpy array, optional
        Add a session level to the preprocessing. Each session will be
        detrended independently. Must be a 1D array of n_samples elements.

    smoothing_fwhm : float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend : boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background' or 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    sample_mask : Any type compatible with numpy-array indexing
        Masks the niimgs along time/fourth dimension. This complements
        3D masking by the mask_img argument. This masking step is applied
        before data preprocessing at the beginning of NiftiMasker.transform.
        This is useful to perform data subselection as part of a scikit-learn
        pipeline.

    memory : instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ----------
    `mask_img_` : nibabel.Nifti1Image
        The mask of the data, or the computed one.

    `affine_` : 4x4 numpy array
        Affine of the transformed image.

    See also
    --------
    nilearn.masking.compute_background_mask
    nilearn.masking.compute_epi_mask
    nilearn.image.resample_img
    nilearn.masking.apply_mask
    nilearn.signal.clean
    """

    def __init__(self, mask_img=None, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0
                 ):
        # Mask is provided or computed
        self.mask_img = mask_img

        self.sessions = sessions
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.sample_mask = sample_mask

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        self._shelving = False

    def _check_fitted(self):
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # y=None is for scikit-learn compatibility (unused here).

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print("[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(imgs)[:200]))

        # Compute the mask if not given by the user
        if self.mask_img is None:
            mask_args = (self.mask_args if self.mask_args is not None
                         else {})
            if self.mask_strategy == 'background':
                compute_mask = masking.compute_background_mask
            elif self.mask_strategy == 'epi':
                compute_mask = masking.compute_epi_mask
            else:
                raise ValueError("Unknown value of mask_strategy '%s'. "
                                 "Acceptable values are 'background' and "
                                 "'epi'." % self.mask_strategy)
            if self.verbose > 0:
                print("[%s.fit] Computing the mask" % self.__class__.__name__)
            self.mask_img_ = self._cache(compute_mask, ignore=['verbose'])(
                imgs, verbose=max(0, self.verbose - 1), **mask_args)
        else:
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print("[%s.fit] Resampling mask" % self.__class__.__name__)
        self.mask_img_ = self._cache(image.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=False)
        if self.target_affine is not None:
            self.affine_ = self.target_affine
        else:
            self.affine_ = get_affine(self.mask_img_)
        # Load data in memory
        self.mask_img_.get_data()
        if self.verbose > 10:
            print("[%s.fit] Finished fit" % self.__class__.__name__)
        return self

    def transform_single_imgs(self, imgs, confounds=None, copy=True):
        """Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs: 3D/4D Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each voxel inside the mask.
            shape: (number of scans, number of voxels)
        """

        # Ignore the mask-computing params: they are not useful and will
        # just invalid the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = get_params(self.__class__, self,
                            ignore=['mask_img', 'mask_args', 'mask_strategy'])

        data = self._cache(filter_and_mask,
                           ignore=['verbose', 'memory', 'memory_level',
                                   'copy'],
                           shelve=self._shelving)(
            imgs, self.mask_img_, params,
            memory_level=self.memory_level,
            memory=self.memory,
            verbose=self.verbose,
            confounds=confounds,
            copy=copy
        )

        return data
