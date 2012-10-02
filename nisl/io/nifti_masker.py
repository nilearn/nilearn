"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

from nibabel import Nifti1Image

from .. import masking
from .. import resampling
from .. import signals
from .. import utils


class NiftiMasker(BaseEstimator, TransformerMixin):
    """Nifti data loader with preprocessing

    Parameters
    ----------
    mask: 3D numpy matrix, optional
        Mask of the data. If not given, a mask is computed in the fit step.

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

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    transpose: boolean, optional
        If true, data are transposed after filtering and inverse_transform
        considered input data as transpose too.
        
    smooth: False or float, optional
        If smooth is not False, it gives the size, in voxel of the
        spatial smoothing to apply to the signal.

    confounds: CSV file path or 2D matrix
        This parameter is passed to signals.clean. Please see the related
        documentation for details

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
        
    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed

    See also
    --------
    nisl.masking.compute_epi_mask
    resampling.resample_img
    masking.apply_mask
    signals.clean
    """

    def __init__(self, sessions=None, mask=None, mask_connected=True,
            mask_opening=False, mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
            smooth=False, confounds=None, detrend=False,
            target_affine=None, target_shape=None, low_pass=None,
            high_pass=None, t_r=None, transpose=False,
            memory=Memory(cachedir=None, verbose=0),
            transform_memory=Memory(cachedir=None, verbose=0), verbose=0):
        # Mask is compulsory or computed
        self.mask = mask
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.transform_memory = transform_memory
        self.verbose = verbose
        self.sessions_ = sessions
        self.transpose = transpose

    def fit(self, niimgs, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        memory = self.memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data" % self.__class__.__name__
        niimgs = utils.check_niimgs(niimgs, accept_3d=True)

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            mask = memory.cache(masking.compute_epi_mask)(
                                niimgs.get_data(),
                                connected=self.mask_connected,
                                opening=self.mask_opening,
                                lower_cutoff=self.mask_lower_cutoff,
                                upper_cutoff=self.mask_upper_cutoff,
                                verbose=(self.verbose -1))
            self.mask_ = Nifti1Image(mask.astype(np.int), niimgs.get_affine())
        else:
			self.mask_ = utils.check_niimg(self.mask)

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling mask" % self.__class__.__name__
        self.mask_ = memory.cache(resampling.resample_img)(self.mask_,
                target_affine=self.target_affine,
                target_shape=self.target_shape,
                copy=(self.target_affine is not None and
                      self.target_shape is not None))

        return self

    def transform(self, niimgs):
        memory = self.transform_memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.transform] Loading data" % self.__class__.__name__
        niimgs = utils.check_niimgs(niimgs)

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling" % self.__class__.__name__
        niimgs = memory.cache(resampling.resample_img)(niimgs,
                    target_affine=self.target_affine,
                    target_shape=self.target_shape)

        # Get series from data with optional smoothing
        if self.verbose > 0:
            print "[%s.transform] Masking and smoothing" \
                % self.__class__.__name__
        data = masking.apply_mask(niimgs, self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 0:
            print "[%s.transform] Cleaning signal" % self.__class__.__name__
        if self.sessions_ is None:
            data = memory.cache(signals.clean)(data,
                    confounds=self.confounds, low_pass=self.low_pass,
                    high_pass=self.high_pass, t_r=self.t_r,
                    detrend=self.detrend, normalize=False)
        else:
            for s in np.unique(self.sessions_):
                if self.confounds is not None:
                    session_confounds = self.confounds[self.sessions_ == s]
                    data[self.sessions_ == s] = \
                        memory.cache(signals.clean)(
                                data=data[self.sessions_ == s],
                                confounds=session_confounds,
                                low_pass=self.low_pass,
                                high_pass=self.high_pass, t_r=self.t_r,
                                detrend=self.detrend, normalize=False)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        # data is in format voxel x time_series. We inverse it
        data = np.rollaxis(data, -1)

        self.affine_ = niimgs.get_affine()
        if self.transpose:
            data = data.T
        return data

    def inverse_transform(self, X):
        if self.transpose:
            X = X.T
        mask = self.mask_.get_data().astype(np.bool)
        if len(X.shape) > 1:
            # we build the data iteratively to avoid MemoryError
            data = []
            for x in X:
                img = np.zeros(mask.shape)
                img[mask] = x
                data.append(img[..., np.newaxis])
            data = np.concatenate(data, axis=-1)
        else:
            data = np.zeros(mask.shape)
            data[mask] = X
        return Nifti1Image(data, self.affine_)
