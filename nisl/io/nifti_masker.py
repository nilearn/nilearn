"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings
from sklearn.externals.joblib import Memory

from .. import masking
from .. import resampling
from ..utils import CacheMixin
from .. import utils
from .base_masker import BaseMasker


class NiftiMasker(BaseMasker, CacheMixin):
    """Nifti data loader with preprocessing

    Parameters
    ----------
    mask: filename or NiImage, optional
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters detailed below (mask_connected...) can be set to
        fine tune the mask extraction.

    sessions: numpy array, optional
        Add a session level to the preprocessing. Each session will be
        detrended independently. Must be a 1D array of n_samples elements.

    smooth: False or float, optional
        If smooth is not False, it gives the size, in voxel of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1.

    detrend: boolean, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    mask_connected: boolean, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_opening: boolean, optional
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

    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ----------
    `mask_img_`: Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `affine_`: 4x4 numpy array
        Affine of the transformed NiImages.

    See also
    --------
    nisl.masking.compute_epi_mask
    nisl.resampling.resample_img
    nisl.masking.apply_mask
    nisl.signals.clean
    """
    def __init__(self, mask=None, sessions=None, smooth=False,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_connected=True, mask_opening=False,
                 mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
                 memory_level=0, memory=Memory(cachedir=None),
                 verbose=0
                 ):
        # Mask is compulsory or computed
        self.mask = mask
        self.sessions = sessions
        self.smooth = smooth
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
        self.verbose = verbose

    def fit(self, niimgs=None, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # y=None is for scikit-learn compatibility (unused here).

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data from %s" % (
                            self.__class__.__name__,
                            utils._repr_niimgs(niimgs)[:200])

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            niimgs = utils.check_niimgs(niimgs, accept_3d=True)
            self.mask_img_ = self._cache(masking.compute_epi_mask,
                              memory_level=1,
                              ignore=['verbose'])(
                niimgs,
                connected=self.mask_connected,
                opening=self.mask_opening,
                lower_cutoff=self.mask_lower_cutoff,
                upper_cutoff=self.mask_upper_cutoff,
                verbose=(self.verbose - 1))
        else:
            if niimgs is not None:
                warnings.warn('[%s.fit] Generation of a mask has been'
                             ' requested (niimgs != None) while a mask has'
                             ' been provided at masker creation. Given mask'
                             ' will be used.' % self.__class__.__name__)
            self.mask_img_ = utils.check_niimg(self.mask)

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling mask" % self.__class__.__name__
        self.mask_img_ = self._cache(resampling.resample_img, memory_level=1)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=(self.target_affine is not None and
                  self.target_shape is not None))

        return self

    def transform(self, niimgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        niimgs: nifti like images
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signals.clean. Please see the related
            documentation for details
        """
        return self.transform_single_niimgs(
            niimgs, self.sessions, confounds)
