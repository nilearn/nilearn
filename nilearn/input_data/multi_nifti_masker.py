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


class MultiNiftiMasker(BaseMasker, CacheMixin):
    """Nifti data loader with preprocessing for multiple subjects

    Parameters
    ==========
    mask: filename or NiImage, optional
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters detailed below (mask_connected...) can be set to
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

    mask_connected: boolean, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_opening: int, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_lower_cutoff: float, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_upper_cutoff: float, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

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

    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ==========
    mask_img_: Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    affine_: 4x4 numpy.ndarray
        Affine of the transformed NiImages. If affine is different across
        subjects, contains the affine of the first subject on which other
        subject data have been resampled.

    See Also
    ========
    nilearn.image.resample_img: image resampling
    nilearn.masking.compute_epi_mask: mask computation
    nilearn.masking.apply_mask: mask application on image
    nilearn.signal.clean: confounds removal and general filtering of signals
    """

    def __init__(self, mask=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_connected=True, mask_opening=2,
                 mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):
        # Mask is provided or computed
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff

        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, niimgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(niimgs)[:200])
        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing mask" % self.__class__.__name__
            data = []
            if not isinstance(niimgs, collections.Iterable) \
                    or isinstance(niimgs, basestring):
                raise ValueError("[%s.fit] For multiple processing, you should"
                                 " provide a list of data "
                                 "(e.g. Nifti1Image objects or filenames)."
                                 "%r is an invalid input"
                                 % (self.__class__.__name__, niimgs))
            for niimg in niimgs:
                # Note that data is not loaded into memory at this stage
                # if niimg is a string
                data.append(_utils.check_niimgs(niimg, accept_3d=True))

            self.mask_img_ = self._cache(
                        masking.compute_multi_epi_mask,
                        memory_level=1,
                        ignore=['n_jobs', 'verbose', 'memory'])(
                            niimgs,
                            connected=self.mask_connected,
                            opening=self.mask_opening,
                            lower_cutoff=self.mask_lower_cutoff,
                            upper_cutoff=self.mask_upper_cutoff,
                            target_affine=self.target_affine,
                            target_shape=self.target_shape,
                            n_jobs=self.n_jobs,
                            memory=self.memory,
                            verbose=(self.verbose - 1))
        else:
            if niimgs is not None:
                warnings.warn('[%s.fit] Generation of a mask has been'
                             ' requested (niimgs != None) while a mask has'
                             ' been provided at masker creation. Given mask'
                             ' will be used.' % self.__class__.__name__)
            self.mask_img_ = _utils.check_niimg(self.mask)

        # If resampling is requested, resample the mask as well.
        # Resampling: allows the user to change the affine, the shape or both.
        if self.verbose > 0:
            print "[%s.transform] Resampling mask" % self.__class__.__name__
        self.mask_img_ = self._cache(image.resample_img,
                                    memory_level=1)(
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
        return self

    def transform(self, niimgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        niimgs: nifti-like images
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signal.clean. Please see the
            corresponding documentation for details.

        Returns
        -------
        data: {list of numpy arrays}
            preprocessed images
        """
        if not hasattr(niimgs, '__iter__')\
                    or isinstance(niimgs, basestring):
                return self.transform_single_niimgs(niimgs)
        return self.transform_niimgs(niimgs, confounds, n_jobs=self.n_jobs)
