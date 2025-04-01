"""Transformer used to apply basic transformations on :term:`fMRI` data."""

import abc
import contextlib
import warnings

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils import repr_niimgs
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level, log
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import (
    concat_imgs,
    high_variance_confounds,
    resample_img,
    smooth_img,
)
from nilearn.masking import unmask
from nilearn.signal import clean


def filter_and_extract(
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

    log(
        f"Loading data from {repr_niimgs(imgs, shorten=False)}",
        verbose=verbose,
    )

    # Convert input to niimg to check shape.
    # This must be repeated after the shape check because check_niimg will
    # coerce 5D data to 4D, which we don't want.
    temp_imgs = check_niimg(imgs)

    # Raise warning if a 3D niimg is provided.
    if temp_imgs.ndim == 3:
        warnings.warn(
            "Starting in version 0.12, 3D images will be transformed to "
            "1D arrays. "
            "Until then, 3D images will be coerced to 2D arrays, with a "
            "singleton first dimension representing time.",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )

    imgs = check_niimg(imgs, atleast_4d=True, ensure_ndim=4, dtype=dtype)

    target_shape = parameters.get("target_shape")
    target_affine = parameters.get("target_affine")
    if target_shape is not None or target_affine is not None:
        log("Resampling images")

        imgs = cache(
            resample_img,
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
        log("Smoothing images", verbose=verbose)

        imgs = cache(
            smooth_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
        )(imgs, parameters["smoothing_fwhm"])

    log("Extracting region signals", verbose=verbose)

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

    log("Cleaning extracted signals", verbose=verbose)

    runs = parameters.get("runs", None)
    region_signals = cache(
        clean,
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

        confounds : CSV file or array-like, default=None
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, \
            default=None
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to clean.

                .. versionadded:: 0.8.0

        copy : :obj:`bool`, default=True
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

            return tags(masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True)
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

        confounds : CSV file or array-like, default=None
            This parameter is passed to clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, \
            default=None
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
        check_is_fitted(self)

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

        y : numpy array of shape [n_samples], default=None
            Target values.

        confounds : :obj:`list` of confounds, default=None
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        sample_mask : :obj:`list` of sample_mask, default=None
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
            " will be used.",
            stacklevel=find_stack_level(),
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
        check_is_fitted(self)

        img = self._cache(unmask)(X, self.mask_img_)
        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        with contextlib.suppress(Exception):
            img._header._structarr = np.array(img._header._structarr).copy()
        return img


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

            return tags(surf_img=True, niimg_like=False, masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            surf_img=True, niimg_like=False, masker=True
        )
        return tags

    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : 1D/2D Surface image
            Images to process.

        confounds : CSV file or array-like, default=None
            This parameter is passed to clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, \
            default=None
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to clean.

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)

        """
        check_is_fitted(self)

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = concat_imgs(imgs)

        check_compatibility_mask_and_images(self.mask_img_, imgs)

        if self.smoothing_fwhm is not None:
            warnings.warn(
                "Parameter smoothing_fwhm "
                "is not yet supported for surface data",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            self.smoothing_fwhm = None

        if self.reports:
            self._reporting_data["images"] = imgs

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

    @abc.abstractmethod
    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single surface image.

        Parameters
        ----------
        imgs : 1D/2D Surface image
            Images to process.

        confounds : CSV file or array-like, default=None
            This parameter is passed to clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, \
            default=None
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to clean.

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each element.
            shape: (number of scans, number of elements)

        """
        raise NotImplementedError()
