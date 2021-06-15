"""
Transformer used to apply basic transformations on multi subject MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import collections.abc
import itertools
import warnings

from joblib import Memory, Parallel, delayed

from .. import _utils
from .. import image
from .. import masking
from .._utils import CacheMixin
from .._utils.class_inspect import get_params
from .._utils.niimg_conversions import _iter_check_niimg
from .nifti_masker import NiftiMasker, filter_and_mask
from nilearn.image import get_data


class MultiNiftiMasker(NiftiMasker, CacheMixin):
    """Class for masking of Niimg-like objects.

    MultiNiftiMasker is useful when dealing with image sets from multiple
    subjects. Use case: integrates well with decomposition by MultiPCA and
    CanICA (multi-subject models)

    Parameters
    ----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters can be set using mask_args and mask_strategy to
        fine tune the mask extraction.

    smoothing_fwhm : float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of
        the spatial smoothing to apply to the signal.

    standardize : {False, True, 'zscore', 'psc'}, optional
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.
        Default=False.

    standardize_confounds : boolean, optional
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension.
        Default=True.

    high_variance_confounds : boolean, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.

    detrend : boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details. Default=False.

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : None or float, optional
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

    mask_strategy : {'background', 'epi' or 'template'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, 'epi' if they
        are raw EPI images, or you could use 'template' which will
        extract the gray matter part of your data by resampling the MNI152
        brain mask for your data's field of view.
        Depending on this value, the mask will be computed from
        masking.compute_background_mask, masking.compute_epi_mask or
        masking.compute_brain_mask. Default is 'background'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    memory : instance of joblib.Memory or string, optional
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching. Default=0.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on. Default=1.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        Default=0.

    Attributes
    ----------
    `mask_img_` : nibabel.Nifti1Image object
        The mask of the data.

    `affine_` : 4x4 numpy.ndarray
        Affine of the transformed image.

    See Also
    --------
    nilearn.image.resample_img: image resampling
    nilearn.masking.compute_epi_mask: mask computation
    nilearn.masking.apply_mask: mask application on image
    nilearn.signal.clean: confounds removal and general filtering of signals

    """

    def __init__(self, mask_img=None, smoothing_fwhm=None,
                 standardize=False, standardize_confounds=True, detrend=False,
                 high_variance_confounds=False, low_pass=None, high_pass=None,
                 t_r=None, target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 dtype=None, memory=Memory(location=None),
                 memory_level=0, n_jobs=1, verbose=0):
        # Mask is provided or computed
        self.mask_img = mask_img

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.dtype = dtype

        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs

        self.verbose = verbose

        self._shelving = False

    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        imgs : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        """

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print("[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(imgs, shorten=False)))
        # Compute the mask if not given by the user
        if self.mask_img is None:
            if self.verbose > 0:
                print("[%s.fit] Computing mask" % self.__class__.__name__)
            if not isinstance(imgs, collections.abc.Iterable) \
                    or isinstance(imgs, str):
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
            elif self.mask_strategy == 'template':
                compute_mask = masking.compute_multi_gray_matter_mask
            else:
                raise ValueError("Unknown value of mask_strategy '%s'. "
                                 "Acceptable values are 'background', 'epi' "
                                 "and 'template'.")

            self.mask_img_ = self._cache(
                compute_mask, ignore=['n_jobs', 'verbose', 'memory'])(
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
            self.affine_ = self.mask_img_.affine
        # Load data in memory
        get_data(self.mask_img_)
        return self

    def transform_imgs(self, imgs_list, confounds=None, sample_mask=None,
                       copy=True, n_jobs=1):
        """Prepare multi subject data in parallel

        Parameters
        ----------

        imgs_list : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            List of imgs file to prepare. One item per subject.

        confounds : list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files or pandas DataFrames). Must be of same length than imgs_list.

        sample_mask : list of sample_mask, optional
            List of sample_mask (1D arrays) if scrubbing motion outliers.
            Must be of same length than imgs_list.

                .. versionadded:: 0.8.0

        copy : boolean, optional
            If True, guarantees that output array has no memory in common with
            input array. Default=True.

        n_jobs : integer, optional
            The number of cpus to use to do the computation. -1 means
            'all cpus'. Default=1.

        Returns
        -------
        region_signals : list of 2D numpy.ndarray
            List of signal for each element per subject.
            shape: list of (number of scans, number of elements)

        """

        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)
        target_fov = None
        if self.target_affine is None:
            # Force resampling on first image
            target_fov = 'first'

        niimg_iter = _iter_check_niimg(imgs_list, ensure_ndim=None,
                                       atleast_4d=False,
                                       target_fov=target_fov,
                                       memory=self.memory,
                                       memory_level=self.memory_level,
                                       verbose=self.verbose)

        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))

        # Ignore the mask-computing params: they are not useful and will
        # just invalidate the cache for no good reason
        # target_shape and target_affine are conveyed implicitly in mask_img
        params = get_params(self.__class__, self,
                            ignore=['mask_img', 'mask_args', 'mask_strategy',
                                    'copy'])

        func = self._cache(filter_and_mask,
                           ignore=['verbose', 'memory', 'memory_level',
                                   'copy'],
                           shelve=self._shelving)
        data = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs, self.mask_img_, params,
                          memory_level=self.memory_level,
                          memory=self.memory,
                          verbose=self.verbose,
                          confounds=cfs,
                          copy=copy,
                          dtype=self.dtype
                          )
            for imgs, cfs in zip(niimg_iter, confounds))
        return data

    def transform(self, imgs, confounds=None, sample_mask=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        imgs : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data to be preprocessed

        confounds : CSV file path or 2D array or pandas DataFrame, optional
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        sample_mask : list of sample_mask, optional
            List of sample_mask (1D arrays) if scrubbing motion outliers.
            Must be of same length than imgs_list.

                .. versionadded:: 0.8.0

        Returns
        -------
        data : {list of numpy arrays}
            preprocessed images

        """
        self._check_fitted()
        if not hasattr(imgs, '__iter__') \
                or isinstance(imgs, str):
            return self.transform_single_imgs(imgs)
        return self.transform_imgs(imgs, confounds=confounds,
                                   sample_mask=sample_mask,
                                   n_jobs=self.n_jobs)
