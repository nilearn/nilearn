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


def _to_nifti(X, affine):
    if isinstance(X, np.ndarray):
        return Nifti1Image(X, affine)
    for index, x in enumerate(X):
        X[index] = _to_nifti(x, affine)
    return X


class BaseMasker(BaseEstimator, TransformerMixin):
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

    normalize: booelan, optional
        If normalize is True, the time-series are normalized: their
        variance is put to 1.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

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

    def transform_single_niimgs(self, niimgs, sessions=None,
                                confounds=None, copy=True):
        memory = self.transform_memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.transform] Loading data from %s" % (
                        self.__class__.__name__,
                        utils._repr_niimgs(niimgs)[:200])

        # If we have a string (filename), we won't need to copy, as
        # there will be no side effect
        if isinstance(niimgs, basestring):
            copy = False

        niimgs = utils.check_niimgs(niimgs)

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 1:
            print "[%s.transform] Resampling" % self.__class__.__name__
        niimgs = memory.cache(resampling.resample_img)(
                            niimgs,
                            target_affine=self.target_affine,
                            target_shape=self.target_shape,
                            copy=copy)

        # Get series from data with optional smoothing
        if self.verbose > 1:
            print "[%s.transform] Masking and smoothing" \
                % self.__class__.__name__
        data = masking.apply_mask(niimgs, self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 1:
            print "[%s.transform] Cleaning signal" % self.__class__.__name__
        if sessions is None:
            data = signals.clean(data,
                confounds=confounds, low_pass=self.low_pass,
                high_pass=self.high_pass, t_r=self.t_r,
                detrend=self.detrend, normalize=self.normalize)
        else:
            for s in np.unique(sessions):
                if confounds is not None:
                    confounds = confounds[sessions == s]
                data[sessions == s] = signals.clean(
                        data[sessions == s],
                        confounds=confounds,
                        low_pass=self.low_pass,
                        high_pass=self.high_pass, t_r=self.t_r,
                        detrend=self.detrend,
                        normalize=self.normalize)

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
        mask = utils.check_niimg(self.mask_)
        if not self.transpose:
            data = X
        else:
            data = X.T
        unmasked = masking.unmask(data, mask.get_data().astype(np.bool),
                                  transpose=True)

        return _to_nifti(unmasked, mask.get_affine())
