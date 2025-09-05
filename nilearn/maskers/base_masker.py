"""Transformer used to apply basic transformations on :term:`fMRI` data."""

import abc
import contextlib
import itertools
import warnings
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.utils.validation import check_array

from nilearn._utils import logger
from nilearn._utils.bids import (
    generate_atlas_look_up_table,
    sanitize_look_up_table,
)
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import (
    rename_parameters,
    stringify_path,
)
from nilearn._utils.logger import find_stack_level
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
from nilearn.image.image import get_indices_from_image
from nilearn.masking import load_mask_img, unmask
from nilearn.signal import clean
from nilearn.surface.surface import SurfaceImage, at_least_2d, check_surf_img
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

    mask_logger("load_data", imgs, verbose)

    # Convert input to niimg to check shape.
    # This must be repeated after the shape check because check_niimg will
    # coerce 5D data to 4D, which we don't want.
    temp_imgs = check_niimg(imgs)

    imgs = check_niimg(imgs, atleast_4d=True, ensure_ndim=4, dtype=dtype)

    target_shape = parameters.get("target_shape")
    target_affine = parameters.get("target_affine")
    if target_shape is not None or target_affine is not None:
        logger.log("Resampling images")

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
        logger.log("Smoothing images", verbose=verbose)

        imgs = cache(
            smooth_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
        )(imgs, parameters["smoothing_fwhm"])

    mask_logger("extracting", verbose=verbose)

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

    mask_logger("cleaning", verbose=verbose)

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

    if temp_imgs.ndim == 3:
        region_signals = region_signals.squeeze()

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


def mask_logger(step, img=None, verbose=0):
    """Log similar messages for all maskers."""
    repr = None
    if img is not None:
        repr = img.__repr__()
        if verbose > 1:
            repr = repr_niimgs(img, shorten=True)
        elif verbose > 2:
            repr = repr_niimgs(img, shorten=False)

    messages = {
        "cleaning": "Cleaning extracted signals",
        "compute_mask": "Computing mask",
        "extracting": "Extracting region signals",
        "fit_done": "Finished fit",
        "inverse_transform": "Computing image from signals",
        "load_data": f"Loading data from {repr}",
        "load_mask": f"Loading mask from {repr}",
        "load_regions": f"Loading regions from {repr}",
        "resample_mask": "Resamping mask",
        "resample_regions": "Resampling regions",
    }

    if step not in messages:
        raise ValueError(f"Unknown step: {step}")

    if step in ["load_mask", "load_data"] and repr is None:
        return

    logger.log(messages[step], verbose=verbose)


@fill_doc
class BaseMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Base class for NiftiMaskers."""

    @abc.abstractmethod
    @fill_doc
    def transform_single_imgs(
        self, imgs, confounds=None, sample_mask=None, copy=True
    ):
        """Extract signals from a single niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.

        %(confounds)s

        %(sample_mask)s

            .. versionadded:: 0.8.0

        copy : :obj:`bool`, default=True
            Indicates whether a copy is returned or not.

        Returns
        -------
        %(signals_transform_nifti)s

        """
        raise NotImplementedError()

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True)
        return tags

    def fit(self, imgs=None, y=None):
        """Present only to comply with sklearn estimators checks."""

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

        mask_logger("load_mask", img=self.mask_img, verbose=self.verbose)

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
            If a 3D niimg is provided, a 1D array is returned.

        %(confounds)s

        %(sample_mask)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_nifti)s
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

    # TODO (nilearn >= 0.13.0)
    @fill_doc
    @rename_parameters(replacement_params={"X": "imgs"}, end_version="0.13.0")
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
        %(signals_transform_nifti)s

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

    @fill_doc
    def inverse_transform(self, X):
        """Transform the data matrix back to an image in brain space.

        This step only performs spatial unmasking,
        without inverting any additional processing performed by ``transform``,
        such as temporal filtering or smoothing.

        Parameters
        ----------
        %(x_inv_transform)s

        Returns
        -------
        %(img_inv_transform_nifti)s

        """
        check_is_fitted(self)

        # do not run sklearn_check as they may cause some failure
        # with some GLM inputs
        X = self._check_array(X, sklearn_check=False)

        mask_logger("inverse_transform", verbose=self.verbose)

        img = self._cache(unmask)(X, self.mask_img_)
        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        with contextlib.suppress(Exception):
            img._header._structarr = np.array(img._header._structarr).copy()
        return img

    def _check_array(
        self, signals: np.ndarray, sklearn_check: bool = True
    ) -> np.ndarray:
        """Check array to inverse transform.

        Parameters
        ----------
        signals : :obj:`numpy.ndarray`

        sklearn_check : :obj:`bool`
            Run scikit learn check on input
        """
        signals = np.atleast_1d(signals)

        if sklearn_check:
            signals = check_array(signals, ensure_2d=False)

        assert signals.ndim <= 2

        expected_shape = (
            (self.n_elements_,)
            if signals.ndim == 1
            else (signals.shape[0], self.n_elements_)
        )

        if signals.shape != expected_shape:
            raise ValueError(
                "Input to 'inverse_transform' has wrong shape.\n"
                f"Expected {expected_shape}.\n"
                f"Got {signals.shape}."
            )

        return signals

    def set_output(self, *, transform=None):
        """Set the output container when ``"transform"`` is called.

        .. warning::

            This has not been implemented yet.
        """
        raise NotImplementedError()

    def _sanitize_cleaning_parameters(self):
        """Make sure that cleaning parameters are passed via clean_args.

        TODO (nilearn >= 0.13.0) remove
        """
        if hasattr(self, "clean_kwargs"):
            if self.clean_kwargs:
                tmp = [", ".join(list(self.clean_kwargs))]
                # TODO (nilearn >= 0.13.0)
                warnings.warn(
                    f"You passed some kwargs to {self.__class__.__name__}: "
                    f"{tmp}. "
                    "This behavior is deprecated "
                    "and will be removed in version >0.13.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )
                if self.clean_args:
                    raise ValueError(
                        "Passing arguments via 'kwargs' "
                        "is mutually exclusive with using 'clean_args'"
                    )
            self.clean_kwargs_ = {
                k[7:]: v
                for k, v in self.clean_kwargs.items()
                if k.startswith("clean__")
            }


class _BaseSurfaceMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Class from which all surface maskers should inherit."""

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False, masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            surf_img=True, niimg_like=False, masker=True
        )
        return tags

    def _check_imgs(self, imgs) -> None:
        if not (
            isinstance(imgs, SurfaceImage)
            or (
                hasattr(imgs, "__iter__")
                and all(isinstance(x, SurfaceImage) for x in imgs)
            )
        ):
            raise TypeError(
                "'imgs' should be a SurfaceImage or "
                "an iterable of SurfaceImage."
                f"Got: {imgs.__class__.__name__}"
            )

    def _load_mask(self, imgs):
        """Load and validate mask if one passed at init.

        Returns
        -------
        mask_img_ : None or 1D binary SurfaceImage
        """
        if self.mask_img is None:
            return None

        mask_img_ = deepcopy(self.mask_img)

        mask_logger("load_mask", img=mask_img_, verbose=self.verbose)

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
                check_surf_img(x)
                check_polymesh_equal(mask_img_.mesh, x.mesh)

        return mask_img_

    # TODO (nilearn >= 0.13.0)
    @rename_parameters(
        replacement_params={"img": "imgs"}, end_version="0.13.0"
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
        %(signals_transform_surface)s
        """
        check_is_fitted(self)
        self._check_imgs(imgs)

        return_1D = isinstance(imgs, SurfaceImage) and len(imgs.shape) < 2

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = concat_imgs(imgs)
        check_surf_img(imgs)

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
            signals = self.transform_single_imgs(
                imgs, confounds=confounds, sample_mask=sample_mask
            )
            return signals.squeeze() if return_1D else signals

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

        signals = self.transform_single_imgs(
            imgs, confounds=all_confounds, sample_mask=sample_mask
        )

        return signals.squeeze() if return_1D else signals

    @abc.abstractmethod
    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single surface image."""
        # implemented in children classes
        raise NotImplementedError()

    # TODO (nilearn >= 0.13.0)
    @rename_parameters(
        replacement_params={"img": "imgs"}, end_version="0.13.0"
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
        %(signals_transform_surface)s
        """
        del y
        return self.fit(imgs).transform(imgs, confounds, sample_mask)

    def _check_array(
        self, signals: np.ndarray, sklearn_check: bool = True
    ) -> np.ndarray:
        """Check array to inverse transform.

        Parameters
        ----------
        signals : :obj:`numpy.ndarray`

        sklearn_check : :obj:`bool`
            Run scikit learn check on input
        """
        signals = np.atleast_2d(signals)

        if sklearn_check:
            signals = check_array(signals, ensure_2d=False)

        if signals.shape[-1] != self.n_elements_:
            raise ValueError(
                "Input to 'inverse_transform' has wrong shape.\n"
                f"Last dimension should be {self.n_elements_}.\n"
                f"Got {signals.shape[-1]}."
            )

        return signals

    def set_output(self, *, transform=None):
        """Set the output container when ``"transform"`` is called.

        .. warning::

            This has not been implemented yet.
        """
        raise NotImplementedError()


def generate_lut(labels_img, background_label, lut=None, labels=None):
    """Generate a look up table if one was not provided.

    Also sanitize its content if necessary.

    Parameters
    ----------
    labels_img : Nifti1Image | SurfaceImage

    background_label : int | float

    lut : Optional[str, Path, pd.DataFrame]

    labels : Optional[list[str]]
    """
    labels_present = get_indices_from_image(labels_img)
    add_background_to_lut = (
        None if background_label not in labels_present else background_label
    )

    if lut is not None:
        if isinstance(lut, (str, Path)):
            lut = pd.read_table(lut, sep=None, engine="python")

    elif labels:
        lut = generate_atlas_look_up_table(
            function=None,
            name=deepcopy(labels),
            index=labels_img,
            background_label=add_background_to_lut,
        )

    else:
        lut = generate_atlas_look_up_table(
            function=None,
            index=labels_img,
            background_label=add_background_to_lut,
        )

    assert isinstance(lut, pd.DataFrame)

    # passed labels or lut may not include background label
    # because of poor data standardization
    # so we need to update the lut accordingly
    mask_background_index = lut["index"] == background_label
    if (mask_background_index).any():
        # Ensure background is the first row with name "Background"
        # Shift the 'name' column down by one
        # if background row was not named properly
        first_rows = lut[mask_background_index]
        other_rows = lut[~mask_background_index]
        lut = pd.concat([first_rows, other_rows], ignore_index=True)

        mask_background_name = lut["name"] == "Background"
        if not (mask_background_name).any():
            lut["name"] = lut["name"].shift(1)

        lut.loc[0, "name"] = "Background"

    else:
        first_row = {
            "name": "Background",
            "index": background_label,
            "color": "FFFFFF",
        }
        first_row = {
            col: first_row[col] if col in lut else np.nan
            for col in lut.columns
        }
        lut = pd.concat([pd.DataFrame([first_row]), lut], ignore_index=True)

    return (
        sanitize_look_up_table(lut, atlas=labels_img)
        .sort_values("index")
        .reset_index(drop=True)
    )
