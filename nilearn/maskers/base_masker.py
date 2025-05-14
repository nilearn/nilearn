"""Transformer used to apply basic transformations on :term:`fMRI` data."""

import abc
import contextlib
import copy
import itertools
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import (
    rename_parameters,
    stringify_path,
)
from nilearn._utils.logger import find_stack_level, log
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg import repr_niimgs, safe_get_data
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils.numpy_conversions import csv_to_array
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import (
    concat_imgs,
    high_variance_confounds,
    new_img_like,
    resample_img,
    smooth_img,
)
from nilearn.masking import load_mask_img, unmask
from nilearn.signal import clean
from nilearn.surface.surface import at_least_2d
from nilearn.surface.utils import check_polymesh_equal


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


def prepare_confounds_multimaskers(masker, imgs_list, confounds):
    """Check and prepare confounds for multimaskers."""
    if confounds is None:
        confounds = list(itertools.repeat(None, len(imgs_list)))
    elif len(confounds) != len(imgs_list):
        raise ValueError(
            f"number of confounds ({len(confounds)}) unequal to "
            f"number of images ({len(imgs_list)})."
        )

    if masker.high_variance_confounds:
        for i, img in enumerate(imgs_list):
            hv_confounds = masker._cache(high_variance_confounds)(img)

            if confounds[i] is None:
                confounds[i] = hv_confounds
            elif isinstance(confounds[i], list):
                confounds[i] += hv_confounds
            elif isinstance(confounds[i], np.ndarray):
                confounds[i] = np.hstack([confounds[i], hv_confounds])
            elif isinstance(confounds[i], pd.DataFrame):
                confounds[i] = np.hstack(
                    [confounds[i].to_numpy(), hv_confounds]
                )
            elif isinstance(confounds[i], (str, Path)):
                c = csv_to_array(confounds[i])
                if np.isnan(c.flat[0]):
                    # There may be a header
                    c = csv_to_array(confounds[i], skip_header=1)
                confounds[i] = np.hstack([c, hv_confounds])
            else:
                confounds[i].append(hv_confounds)

    return confounds


@fill_doc
class BaseMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Base class for NiftiMaskers."""

    @abc.abstractmethod
    @fill_doc
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

        %(confounds)s

        %(sample_mask)s

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

    def _load_mask(self, imgs):
        """Load and validate mask if one passed at init.

        Returns
        -------
        mask_img_ : None or 3D binary nifti
        """
        if self.mask_img is None:
            # in this case
            # (Multi)Niftimasker will infer one from imaged to fit
            # other nifti maskers are OK with None
            return None

        repr = repr_niimgs(self.mask_img, shorten=(not self.verbose))
        msg = f"loading mask from {repr}"
        log(msg=msg, verbose=self.verbose)

        # ensure that the mask_img_ is a 3D binary image
        tmp = check_niimg(self.mask_img, atleast_4d=True)
        mask = safe_get_data(tmp, ensure_finite=True)
        mask = mask.astype(bool).all(axis=3)
        mask_img_ = new_img_like(self.mask_img, mask)

        # Just check that the mask is valid
        load_mask_img(mask_img_)
        if imgs is not None:
            check_compatibility_mask_and_images(self.mask_img, imgs)

        return mask_img_

    @fill_doc
    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        %(confounds)s

        %(sample_mask)s

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

    @rename_parameters(replacement_params={"X": "imgs"}, end_version="0.13.2")
    @fill_doc
    def fit_transform(
        self, imgs, y=None, confounds=None, sample_mask=None, **fit_params
    ):
        """Fit to data, then transform it.

        Parameters
        ----------
        imgs : Niimg-like object
            See :ref:`extracting_data`.

        y : numpy array of shape [n_samples], default=None
            Target values.

        %(confounds)s

        %(sample_mask)s

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
                return self.fit(imgs, **fit_params).transform(
                    imgs, confounds=confounds, sample_mask=sample_mask
                )

            return self.fit(**fit_params).transform(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        # fit method of arity 2 (supervised transformation)
        if self.mask_img is None:
            return self.fit(imgs, y, **fit_params).transform(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        warnings.warn(
            f"[{self.__class__.__name__}.fit] "
            "Generation of a mask has been"
            " requested (y != None) while a mask was"
            " given at masker creation. Given mask"
            " will be used.",
            stacklevel=find_stack_level(),
        )
        return self.fit(**fit_params).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
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

        self._check_signal_shape(X)

        img = self._cache(unmask)(X, self.mask_img_)
        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        with contextlib.suppress(Exception):
            img._header._structarr = np.array(img._header._structarr).copy()
        return img

    def _check_signal_shape(self, signals: np.ndarray):
        if signals.shape[-1] != self.n_elements_:
            raise ValueError(
                "Input to 'inverse_transform' has wrong shape.\n"
                f"Last dimension should be {self.n_elements_}.\n"
                f"Got {signals.shape[-1]}."
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

            return tags(surf_img=True, niimg_like=False, masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            surf_img=True, niimg_like=False, masker=True
        )
        return tags

    def _load_mask(self, imgs):
        """Load and validate mask if one passed at init.

        Returns
        -------
        mask_img_ : None or 1D binary SurfaceImage
        """
        if self.mask_img is None:
            return None

        mask_img_ = copy.deepcopy(self.mask_img)

        logger.log(
            msg=f"loading mask from {mask_img_.__repr__()}",
            verbose=self.verbose,
        )

        mask_img_ = at_least_2d(mask_img_)
        mask = {}
        for part, v in mask_img_.data.parts.items():
            mask[part] = v
            non_finite_mask = np.logical_not(np.isfinite(mask[part]))
            if non_finite_mask.any():
                warnings.warn(
                    "Non-finite values detected. "
                    "These values will be replaced with zeros.",
                    stacklevel=find_stack_level(),
                )
                mask[part][non_finite_mask] = 0
            mask[part] = mask[part].astype(bool).all(axis=1)

        mask_img_ = new_img_like(self.mask_img, mask)

        # Just check that the mask is valid
        load_mask_img(mask_img_)
        if imgs is not None:
            check_compatibility_mask_and_images(mask_img_, imgs)
            if not isinstance(imgs, Iterable):
                imgs = [imgs]
            for x in imgs:
                check_polymesh_equal(mask_img_.mesh, x.mesh)

        return mask_img_

    @rename_parameters(
        replacement_params={"img": "imgs"}, end_version="0.13.2"
    )
    @fill_doc
    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : :obj:`~nilearn.surface.SurfaceImage` object or \
              iterable of :obj:`~nilearn.surface.SurfaceImage`
            Images to process.

        %(confounds)s

        %(sample_mask)s

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
        """Extract signals from a single surface image."""
        # implemented in children classes
        raise NotImplementedError()

    @rename_parameters(
        replacement_params={"img": "imgs"}, end_version="0.13.2"
    )
    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`~nilearn.surface.SurfaceImage` object or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres. The data for each hemisphere \
            is of shape (n_vertices_per_hemisphere, n_timepoints).

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        %(confounds)s

        %(sample_mask)s


        Returns
        -------
        signals: :obj:`numpy.ndarray`
            Signal for each region as provided
            in the mask, label or maps image.
            The shape will vary depending on the masker type:
            - SurfaceMasker: (n_timepoints, n_vertices_in_mask)
            - SurfaceLabelsMasker: (n_timepoints, n_regions)
            - SurfaceMapssMasker: (n_timepoints, n_maps)
        """
        del y
        return self.fit(imgs).transform(imgs, confounds, sample_mask)

    def _check_signal_shape(self, signals: np.ndarray):
        if signals.shape[-1] != self.n_elements_:
            raise ValueError(
                "Input to 'inverse_transform' has wrong shape.\n"
                f"Last dimension should be {self.n_elements_}.\n"
                f"Got {signals.shape[-1]}."
            )
