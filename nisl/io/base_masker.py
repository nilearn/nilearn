"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory
from nibabel import Nifti1Image

from .. import masking
from .. import resampling
from .. import signal
from .. import utils
from ..utils import CacheMixin, cache


def _to_nifti(X, affine):
    if isinstance(X, np.ndarray):
        return Nifti1Image(X, affine)
    for index, x in enumerate(X):
        X[index] = _to_nifti(x, affine)
    return X


def _transform_single_niimgs(niimgs, mask_img_, sessions=None,
                            smoothing_fwhm=None,
                            standardize=True, detrend=False,
                            low_pass=None, high_pass=None, t_r=None,
                            target_affine=None, target_shape=None,
                            ref_memory_level=0,
                            memory=Memory(cachedir=None),
                            verbose=0,
                            confounds=None,
                            class_name='',
                            copy=True):
    # Load data (if filenames are given, load them)
    if verbose > 0:
        print "[%s.transform] Loading data from %s" % (
            class_name,
            utils._repr_niimgs(niimgs)[:200])

    # If we have a string (filename), we won't need to copy, as
    # there will be no side effect
    if isinstance(niimgs, basestring):
        copy = False

    niimgs = utils.check_niimgs(niimgs)

    # Resampling: allows the user to change the affine, the shape or both
    if verbose > 1:
        print "[%s.transform] Resampling" % class_name
    niimgs = cache(resampling.resample_img, memory, ref_memory_level,
                   memory_level=2)(
            niimgs,
            target_affine=target_affine,
            target_shape=target_shape,
            copy=copy)

    # Get series from data with optional smoothing
    if verbose > 1:
        print "[%s.transform] Masking and smoothing" \
            % class_name
    data = masking.apply_mask(niimgs, mask_img_,
                              smoothing_fwhm=smoothing_fwhm)

    # Temporal
    # ========
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing

    if verbose > 1:
        print "[%s.transform] Cleaning signal" % class_name
    if sessions is None:
        data = cache(signal.clean, memory, ref_memory_level, memory_level=2)(
            data,
            confounds=confounds, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r,
            detrend=detrend,
            standardize=standardize)
    else:
        for s in np.unique(sessions):
            if confounds is not None:
                confounds = confounds[sessions == s]
            data[:, sessions == s] = \
                cache(signal.clean, memory, ref_memory_level, memory_level=2)(
                    data[:, sessions == s],
                    confounds=confounds,
                    low_pass=low_pass,
                    high_pass=high_pass, t_r=t_r,
                    detrend=detrend,
                    standardize=standardize
                )

    # For _later_: missing value removal or imputing of missing data
    # (i.e. we want to get rid of NaNs, if smoothing must be done
    # earlier)
    # Optionally: 'doctor_nan', remove voxels with NaNs, other option
    # for later: some form of imputation

    return data, niimgs.get_affine()


class BaseMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Base class for NiftiMaskers
    """

    def transform_single_niimgs(self, niimgs, sessions=None,
                                confounds=None, copy=True):
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fit. '
                "You must call fit() before calling transform()."
                % self.__class__.__name__)

        data, affine = \
            self._cache(_transform_single_niimgs, memory_level=1)(
                niimgs, self.mask_img_,
                smoothing_fwhm=self.smoothing_fwhm,
                standardize=self.standardize, detrend=self.detrend,
                low_pass=self.low_pass, high_pass=self.high_pass, t_r=self.t_r,
                target_affine=self.target_affine,
                target_shape=self.target_shape,
                ref_memory_level=self.memory_level,
                memory=self.memory,
                verbose=self.verbose,
                confounds=confounds,
                class_name=self.__class__.__name__,
                copy=copy
            )

        self.affine_ = affine
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
            if self.mask is None:
                return self.fit(X, **fit_params
                                ).transform(X, confounds=confounds)
            else:
                return self.fit(**fit_params).transform(X, confounds=confounds)
        else:
            # fit method of arity 2 (supervised transformation)
            if self.mask is None:
                return self.fit(X, y, **fit_params
                                ).transform(X, confounds=confounds)
            else:
                warnings.warn('[%s.fit] Generation of a mask has been'
                              ' requested (y != None) while a mask has'
                              ' been provided at masker creation. Given mask'
                              ' will be used.' % self.__class__.__name__)
                return self.fit(**fit_params).transform(X, confounds=confounds)

    def inverse_transform(self, X):
        mask_img = utils.check_niimg(self.mask_img_)
        data = X

        return masking.unmask(data, mask_img)
