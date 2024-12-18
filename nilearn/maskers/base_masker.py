"""Transformer used to apply basic transformations on :term:`fMRI` data."""

# Author: Gael Varoquaux, Alexandre Abraham

import abc
import contextlib
import warnings

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import high_variance_confounds

from .. import _utils, image, masking, signal
from .._utils import logger, stringify_path
from .._utils.cache_mixin import CacheMixin, cache


def _filter_and_extract(
    imgs,
    extraction_function,
    parameters,
    memory_level=0,
    memory=None,
    verbose=0,
    confounds=None,
    sample_mask=None,
    copy=True,
    dtype=None,
):
    """Extract representative time series using given function.

    Parameters
    ----------
    imgs : 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    extraction_function : function
        Function used to extract the time series from 4D data. This function
        should take images as argument and returns a tuple containing a 2D
        array with masked signals along with a auxiliary value used if
        returning a second value is needed.
        If any other parameter is needed, a functor or a partial
        function must be provided.

    For all other parameters refer to NiftiMasker documentation

    Returns
    -------
    signals : 2D numpy array
        Signals extracted using the extraction function. It is a scikit-learn
        friendly 2D array with shape n_samples x n_features.

    """
    if memory is None:
        memory = Memory(location=None)
    # If we have a string (filename), we won't need to copy, as
    # there will be no side effect
    imgs = stringify_path(imgs)
    if isinstance(imgs, str):
        copy = False

    logger.log(
        f"Loading data from {_utils.repr_niimgs(imgs, shorten=False)}",
        verbose=verbose,
        stack_level=2,
    )

    # Convert input to niimg to check shape.
    # This must be repeated after the shape check because check_niimg will
    # coerce 5D data to 4D, which we don't want.
    temp_imgs = _utils.check_niimg(imgs)

    # Raise warning if a 3D niimg is provided.
    if temp_imgs.ndim == 3:
        warnings.warn(
            "Starting in version 0.12, 3D images will be transformed to "
            "1D arrays. "
            "Until then, 3D images will be coerced to 2D arrays, with a "
            "singleton first dimension representing time.",
            DeprecationWarning,
        )

    imgs = _utils.check_niimg(
        imgs, atleast_4d=True, ensure_ndim=4, dtype=dtype
    )

    target_shape = parameters.get("target_shape")
    target_affine = parameters.get("target_affine")
    if target_shape is not None or target_affine is not None:
        logger.log("Resampling images", stack_level=2)

        imgs = cache(
            image.resample_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
            ignore=["copy"],
        )(
            imgs,
            interpolation="continuous",
            target_shape=target_shape,
            target_affine=target_affine,
            copy=copy,
            copy_header=True,
            force_resample=False,  # set to True in 0.13.0
        )

    smoothing_fwhm = parameters.get("smoothing_fwhm")
    if smoothing_fwhm is not None:
        logger.log("Smoothing images", verbose=verbose, stack_level=2)
        imgs = cache(
            image.smooth_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
        )(imgs, parameters["smoothing_fwhm"])

    logger.log("Extracting region signals", verbose=verbose, stack_level=2)
    region_signals, aux = cache(
        extraction_function,
        memory,
        func_memory_level=2,
        memory_level=memory_level,
    )(imgs)

    # Temporal
    # --------
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing
    logger.log("Cleaning extracted signals", verbose=verbose, stack_level=2)
    runs = parameters.get("runs", None)
    region_signals = cache(
        signal.clean,
        memory=memory,
        func_memory_level=2,
        memory_level=memory_level,
    )(
        region_signals,
        detrend=parameters["detrend"],
        standardize=parameters["standardize"],
        standardize_confounds=parameters["standardize_confounds"],
        t_r=parameters["t_r"],
        low_pass=parameters["low_pass"],
        high_pass=parameters["high_pass"],
        confounds=confounds,
        sample_mask=sample_mask,
        runs=runs,
        **parameters["clean_kwargs"],
    )

    return region_signals, aux


class BaseMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Base class for NiftiMaskers."""

    @abc.abstractmethod
    def transform_single_imgs(
        self, imgs, confounds=None, sample_mask=None, copy=True
    ):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        copy : Boolean, default=True
            Indicates whether a copy is returned or not.

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        raise NotImplementedError()

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags()

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags()
        return tags

    def fit(self, imgs=None, y=None):
        """Present only to comply with sklearn estimators checks."""
        ...

    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        self._check_fitted()

        if confounds is None and not self.high_variance_confounds:
            return self.transform_single_imgs(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        # Compute high variance confounds if requested
        all_confounds = []
        if self.high_variance_confounds:
            hv_confounds = self._cache(high_variance_confounds)(imgs)
            all_confounds.append(hv_confounds)
        if confounds is not None:
            if isinstance(confounds, list):
                all_confounds += confounds
            else:
                all_confounds.append(confounds)

        return self.transform_single_imgs(
            imgs, confounds=all_confounds, sample_mask=sample_mask
        )

    def fit_transform(
        self, X, y=None, confounds=None, sample_mask=None, **fit_params
    ):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : Niimg-like object
            See :ref:`extracting_data`.

        y : numpy array of shape [n_samples], optional
            Target values.

        confounds : list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        sample_mask : list of sample_mask, optional
            List of sample_mask (1D arrays) if scrubbing motion outliers.
            Must be of same length than imgs_list.

                .. versionadded:: 0.8.0

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
                return self.fit(X, **fit_params).transform(
                    X, confounds=confounds, sample_mask=sample_mask
                )

            return self.fit(**fit_params).transform(
                X, confounds=confounds, sample_mask=sample_mask
            )

        # fit method of arity 2 (supervised transformation)
        if self.mask_img is None:
            return self.fit(X, y, **fit_params).transform(
                X, confounds=confounds, sample_mask=sample_mask
            )

        warnings.warn(
            f"[{self.__class__.__name__}.fit] "
            "Generation of a mask has been"
            " requested (y != None) while a mask has"
            " been provided at masker creation. Given mask"
            " will be used."
        )
        return self.fit(**fit_params).transform(
            X, confounds=confounds, sample_mask=sample_mask
        )

    def inverse_transform(self, X):
        """Transform the 2D data matrix back to an image in brain space.

        This step only performs spatial unmasking,
        without inverting any additional processing performed by ``transform``,
        such as temporal filtering or smoothing.

        Parameters
        ----------
        X : 1D/2D :obj:`numpy.ndarray`
            Signal for each element in the mask.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.
            See :ref:`extracting_data`.

        Returns
        -------
        img : Transformed image in brain space.

        """
        self._check_fitted()

        img = self._cache(masking.unmask)(X, self.mask_img_)
        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        with contextlib.suppress(Exception):
            img._header._structarr = np.array(img._header._structarr).copy()
        return img

    def _check_fitted(self):
        if not hasattr(self, "mask_img_"):
            raise ValueError(
                f"It seems that {self.__class__.__name__} "
                "has not been fitted. "
                "You must call fit() before calling transform()."
            )


class _BaseSurfaceMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Class from which all surface maskers should inherit."""

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags
