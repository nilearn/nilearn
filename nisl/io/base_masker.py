"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nibabel import Nifti1Image

from .. import masking
from .. import resampling
from .. import signals
from .. import utils
from ..utils import CacheMixin


def _to_nifti(X, affine):
    if isinstance(X, np.ndarray):
        return Nifti1Image(X, affine)
    for index, x in enumerate(X):
        X[index] = _to_nifti(x, affine)
    return X


class BaseMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Base class for NiftiMaskers
    """

    def transform_single_niimgs(self, niimgs, sessions=None,
                                confounds=None, copy=True):
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fit. '
                "You must call fit() before calling transform()."
                % self.__class__.__name__)

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
        niimgs = self._cache(resampling.resample_img, memory_level=2)(
            niimgs,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=copy)

        # Get series from data with optional smoothing
        if self.verbose > 1:
            print "[%s.transform] Masking and smoothing" \
                % self.__class__.__name__
        data = masking.apply_mask(niimgs, self.mask_img_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering
        # Confounds removing (from csv file or numpy array)
        # Normalizing

        if self.verbose > 1:
            print "[%s.transform] Cleaning signal" % self.__class__.__name__
        if sessions is None:
            data = self._cache(signals.clean, memory_level=2)(
                data,
                confounds=confounds, low_pass=self.low_pass,
                high_pass=self.high_pass, t_r=self.t_r,
                detrend=self.detrend,
                standardize=self.standardize)
        else:
            for s in np.unique(sessions):
                if confounds is not None:
                    confounds = confounds[sessions == s]
                data[:, sessions == s] = self._cache(signals.clean,
                                                    memory_level=2)(
                    data[:, sessions == s],
                    confounds=confounds,
                    low_pass=self.low_pass,
                    high_pass=self.high_pass, t_r=self.t_r,
                    detrend=self.detrend,
                    standardize=self.standardize)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        self.affine_ = niimgs.get_affine()
        return data

    def fit_transform(self, X, y=None, confounds=None, **fit_params):
        """Fit to data, then transform it

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X, confounds=confounds)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params) \
                .transform(X, confounds=confounds)

    def inverse_transform(self, X):
        mask_img = utils.check_niimg(self.mask_img_)
        data = X

        return masking.unmask(data, mask_img)
