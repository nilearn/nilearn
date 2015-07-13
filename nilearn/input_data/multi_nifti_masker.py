"""
Transformer used to apply basic transformations on multi subject MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings
import collections

from sklearn.externals.joblib import Memory

from .. import masking
from .. import image
from .. import _utils
from .._utils import CacheMixin
from .base_masker import BaseMasker
from .._utils.compat import _basestring


class MultiNiftiMasker(BaseMasker, CacheMixin):
    """Class for masking of Niimg-like objects.

    MultiNiftiMasker is useful when dealing with image sets from multiple
    subjects. Use case: integrates well with decomposition by MultiPCA and
    CanICA (multi-subject models)

    Parameters
    ==========
    mask_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters can be set using mask_args and mask_strategy to
        fine tune the mask extraction.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
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

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ==========
    mask_img_: nibabel.Nifti1Image object
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    affine_: 4x4 numpy.ndarray
        Affine of the transformed image. If affine is different across
        subjects, contains the affine of the first subject on which other
        subject data have been resampled.

    See Also
    ========
    nilearn.image.resample_img: image resampling
    nilearn.masking.compute_epi_mask: mask computation
    nilearn.masking.apply_mask: mask application on image
    nilearn.signal.clean: confounds removal and general filtering of signals
    """

    def __init__(self, mask_img=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):
        # Mask is provided or computed
        self.mask_img = mask_img

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
        self.n_jobs = n_jobs
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

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print("[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(imgs)[:200]))
        # Compute the mask if not given by the user
        if self.mask_img is None:
            if self.verbose > 0:
                print("[%s.fit] Computing mask" % self.__class__.__name__)
            data = []
            if not isinstance(imgs, collections.Iterable) \
                    or isinstance(imgs, _basestring):
                raise ValueError("[%s.fit] For multiple processing, you should"
                                 " provide a list of data "
                                 "(e.g. Nifti1Image objects or filenames)."
                                 "%r is an invalid input"
                                 % (self.__class__.__name__, imgs))

            mask_args = (self.mask_args if self.mask_args is not None
                         else {})
            if self.mask_strategy == 'background':
                compute_mask = masking.compute_multi_background_mask
            elif self.mask_strategy == 'epi':
                compute_mask = masking.compute_multi_epi_mask
            else:
                raise ValueError("Unknown value of mask_strategy '%s'. "
                    "Acceptable values are 'background' and 'epi'.")

            self.mask_img_ = self._cache(compute_mask,
                        ignore=['n_jobs', 'verbose', 'memory'])(
                            imgs,
                            target_affine=self.target_affine,
                            target_shape=self.target_shape,
                            n_jobs=self.n_jobs,
                            memory=self.memory,
                            verbose=max(0, self.verbose - 1),
                        **mask_args)
        else:
            if imgs is not None:
                warnings.warn('[%s.fit] Generation of a mask has been'
                             ' requested (imgs != None) while a mask has'
                             ' been provided at masker creation. Given mask'
                             ' will be used.' % self.__class__.__name__)
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

        # If resampling is requested, resample the mask as well.
        # Resampling: allows the user to change the affine, the shape or both.
        if self.verbose > 0:
            print("[%s.transform] Resampling mask" % self.__class__.__name__)
        self.mask_img_ = self._cache(image.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            interpolation='nearest', copy=False)
        if self.target_affine is not None:
            self.affine_ = self.target_affine
        else:
            self.affine_ = self.mask_img_.get_affine()
        # Load data in memory
        self.mask_img_.get_data()
        return self

    def transform(self, imgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        Returns
        -------
        data: {list of numpy arrays}
            preprocessed images
        """
        self._check_fitted()

        if not hasattr(imgs, '__iter__')\
                    or isinstance(imgs, _basestring):
                return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds, n_jobs=self.n_jobs)
