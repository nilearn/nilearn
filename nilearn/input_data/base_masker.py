"""
Transformer used to apply basic transformations on MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings

import numpy as np
import itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed

from .. import masking
from .. import image
from .. import signal
from .. import _utils
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.class_inspect import enclosing_scope_name, get_params
from .._utils.compat import _basestring, izip
from nilearn._utils.niimg_conversions import _iter_check_niimg


def filter_and_extract(imgs, extraction_function,
                       smoothing_fwhm, t_r,
                       standardize, detrend, low_pass, high_pass,
                       confounds, memory, memory_level,
                       sessions=None,
                       target_shape=None,
                       target_affine=None,
                       copy=True,
                       sample_mask=None, verbose=0):
    """Extract representative time series using given function.

    Parameters
    ----------
    imgs: images
        Images to be masked

    extraction_function: function
        Function used to extract the time series from 4D data.

    For all other parameters refer to NiftiMasker documentation
    """
    # Since the calling class can be any *Nifti*Masker, we look for exact type
    if verbose > 0:
        class_name = enclosing_scope_name(stack_level=3)

    # If we have a string (filename), we won't need to copy, as
    # there will be no side effect
    if isinstance(imgs, _basestring):
        copy = False

    if verbose > 0:
        print("[%s] Loading data from %s" % (
            class_name,
            _utils._repr_niimgs(imgs)[:200]))
    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)
    if sample_mask is not None:
        imgs = image.index_img(imgs, sample_mask)

    if target_shape is not None or target_affine is not None:
        if verbose > 0:
            print("[%s] Resampling images" % class_name)
        imgs = cache(
            image.resample_img, memory, func_memory_level=2,
            memory_level=memory_level, ignore=['copy'])(
                imgs, interpolation="continuous",
                target_shape=target_shape,
                target_affine=target_affine,
                copy=copy)

    if smoothing_fwhm is not None:
        if verbose > 0:
            print("[%s] Smoothing images" % class_name)
        imgs = cache(
            image.smooth_img, memory, func_memory_level=2,
            memory_level=memory_level)(
                imgs, smoothing_fwhm)

    if verbose > 0:
        print("[%s] Extracting region signals" % class_name)
    region_signals, aux = cache(extraction_function, memory,
                                func_memory_level=2,
                                memory_level=memory_level)(imgs)

    # Temporal
    # ========
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing

    if verbose > 0:
        print("[%s] Cleaning extracted signals" % class_name)
    region_signals = cache(
        signal.clean, memory=memory, func_memory_level=2,
        memory_level=memory_level)(
            region_signals, detrend=detrend, standardize=standardize, t_r=t_r,
            low_pass=low_pass, high_pass=high_pass,
            confounds=confounds, sessions=sessions)

    return region_signals, aux


class BaseMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Base class for NiftiMaskers
    """

    def transform_single_imgs(self, imgs, confounds=None, copy=True):

        self._check_fitted()
        params = self._get_params()

        data, _ = self._cache(self.filter_and_mask,
                              ignore=['verbose', 'memory', 'copy'])(
                                    imgs, params,
                                    memory_level=self.memory_level,
                                    memory=self.memory,
                                    verbose=self.verbose,
                                    confounds=confounds,
                                    copy=copy
        )
        return data

    def transform_imgs(self, imgs_list, confounds=None, copy=True, n_jobs=1):
        ''' Prepare multi subject data in parallel

        Parameters
        ----------

        imgs_list: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            List of imgs file to prepare. One item per subject.

        confounds: list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        copy: boolean, optional
            If True, guarantees that output array has no memory in common with
            input array.

        n_jobs: integer, optional
            The number of cpus to use to do the computation. -1 means
            'all cpus'.
        '''

        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)
        params = get_params(self.__class__, self)

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

        func = self._cache(self.filter_and_mask,
                           ignore=['verbose', 'memory', 'copy'])
        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))
        for name in ('mask_img', 'mask_args', 'mask_strategy'):
            params.pop(name, None)
        data = Parallel(n_jobs=n_jobs)(delayed(func)(
                                imgs, self.mask_img_,
                                parameters=params,
                                memory_level=self.memory_level,
                                memory=self.memory,
                                verbose=self.verbose,
                                confounds=confounds,
                                copy=copy)
                        for imgs, confounds in izip(niimg_iter, confounds))
        return list(zip(*data))[0]

    def fit_transform(self, X, y=None, confounds=None, **fit_params):
        """Fit to data, then transform it

        Parameters
        ----------
        X : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.

        y : numpy array of shape [n_samples]
            Target values.

        confounds: list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            if self.mask_img is None:
                return self.fit(X, **fit_params
                                ).transform(X, confounds=confounds)
            else:
                return self.fit(**fit_params).transform(X, confounds=confounds)
        else:
            # fit method of arity 2 (supervised transformation)
            if self.mask_img is None:
                return self.fit(X, y, **fit_params
                                ).transform(X, confounds=confounds)
            else:
                warnings.warn('[%s.fit] Generation of a mask has been'
                              ' requested (y != None) while a mask has'
                              ' been provided at masker creation. Given mask'
                              ' will be used.' % self.__class__.__name__)
                return self.fit(**fit_params).transform(X, confounds=confounds)

    def inverse_transform(self, X):
        img = self._cache(masking.unmask)(X, self.mask_img_)
        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        try:
            img._header._structarr = np.array(img._header._structarr).copy()
        except:
            pass
        return img

    def _check_fitted(self):
        if not hasattr(self, "mask_img_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)
