"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

from sklearn.externals.joblib import Memory

from .. import masking
from .. import image
from .. import _utils
from .._utils import CacheMixin
from .base_masker import BaseMasker


class NiftiMasker(BaseMasker, CacheMixin):
    """Nifti data loader with preprocessing

    Parameters
    ----------
    mask_img : Niimg-like object, optional
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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

    low_pass : False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : False or float, optional
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
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

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
                 mask_args=None,
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

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # y=None is for scikit-learn compatibility (unused here).

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data from %s" % (
                            self.__class__.__name__,
                            _utils._repr_niimgs(imgs)[:200])

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
                    "Acceptable values are 'background' and 'epi'.")
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            imgs = _utils.check_niimgs(imgs, accept_3d=True)
            self.mask_img_ = self._cache(compute_mask,
                              func_memory_level=1,
                              ignore=['verbose'])(
                imgs,
                verbose=max(0, self.verbose - 1),
                **mask_args)
        else:
            self.mask_img_ = _utils.check_niimg(self.mask_img,
                                                ensure_3d=True)

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.fit] Resampling mask" % self.__class__.__name__
        self.mask_img_ = self._cache(image.resample_img, func_memory_level=1)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=False)
        if self.target_affine is not None:
            self.affine_ = self.target_affine
        else:
            self.affine_ = self.mask_img_.get_affine()
        # Load data in memory
        self.mask_img_.get_data()
        if self.verbose > 10:
            print "[%s.fit] Finished fit" % self.__class__.__name__
        return self

    def transform(self, imgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        return self.transform_single_imgs(
            imgs, confounds)
