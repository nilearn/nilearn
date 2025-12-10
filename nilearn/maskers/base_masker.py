"""Transformer used to apply basic transformations on :term:`fMRI` data."""

import abc
import contextlib
import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from joblib import Memory
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.utils.validation import check_array

from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg import repr_niimgs, safe_get_data
from nilearn._utils.param_validation import check_parameter_in_allowed
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.exceptions import NotImplementedWarning
from nilearn.image import (
    check_niimg,
    concat_imgs,
    high_variance_confounds,
    new_img_like,
    resample_img,
    smooth_img,
)
from nilearn.maskers._mixin import _ReportingMixin
from nilearn.masking import load_mask_img, unmask
from nilearn.signal import clean
from nilearn.surface.surface import SurfaceImage, at_least_2d, check_surf_img
from nilearn.surface.utils import check_polymesh_equal

STANDARIZE_WARNING_MESSAGE = (
    "The 'zscore' strategy incorrectly "
    "uses population std to calculate sample zscores. "
    "The new strategy 'zscore_sample' corrects this "
    "behavior by using the sample std. "
    "In release 0.14.0, the 'zscore' option will be removed "
    "and using standardize=True will fall back "
    "to 'zscore_sample'."
    "To avoid this warning, please use 'zscore_sample' instead."
)


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
    sklearn_output_config=None,
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
    signals : 1D or 2D numpy array
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
        logger.log("Resampling images", verbose=verbose)

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

    # if we need to output to numpy and input was a 3D img
    # we return 1D array
    if temp_imgs.ndim == 3 and sklearn_output_config is None:
        region_signals = region_signals.squeeze()

    return region_signals, aux


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
        "resample_mask": "Resampling mask",
        "resample_regions": "Resampling regions",
    }

    check_parameter_in_allowed(step, messages.keys(), "step")

    if step in ["load_mask", "load_data"] and repr is None:
        return

    logger.log(messages[step], verbose=verbose)


def check_displayed_maps(
    displayed_maps: Any, var_name: str = "displayed_maps"
) -> None:
    """Check type displayed_maps parameter for report generation."""
    incorrect_type = not isinstance(
        displayed_maps, (list, np.ndarray, int, str)
    )
    incorrect_string = (
        isinstance(displayed_maps, str) and displayed_maps != "all"
    )
    not_integer = (
        isinstance(displayed_maps, np.ndarray)
        and (np.array(displayed_maps).dtype != int)
    ) or (
        isinstance(displayed_maps, list)
        and not all(isinstance(i, int) for i in displayed_maps)
    )
    if incorrect_type or incorrect_string or not_integer:
        input_type = displayed_maps.__class__.__name__
        if isinstance(displayed_maps, (np.ndarray, list)):
            input_type = (
                f"{displayed_maps.__class__.__name__} "
                f"of { {i.__class__.__name__ for i in displayed_maps} }"
            )
        raise TypeError(
            f"Parameter '{var_name}' of "
            "``generate_report()`` should be either 'all' or "
            "a positive 'int', or a list/array of ints. "
            f"You provided a {input_type}."
        )


def sanitize_displayed_maps(
    estimator,
    displayed_maps: Any,
    n_maps: int,
    var_name: str = "map",
) -> tuple[Any, list[int]]:
    """Check and sanitize displayed_maps parameter for report generation.

    Eventually adjust displayed_maps and add warning messages to estimator.

    First coerce displayed_maps to a list of integers.
    Then check that all requested maps are available in the masker.
    """
    if isinstance(displayed_maps, str) and displayed_maps == "all":
        displayed_maps = n_maps

    if isinstance(displayed_maps, int):
        if n_maps < displayed_maps:
            msg = (
                "`generate_report()` received "
                f"'displayed_{var_name}s={displayed_maps}' to be displayed. "
                f"But masker only has {n_maps} {var_name}(s). "
                f"'displayed_{var_name}s' was set to {n_maps}."
            )
            estimator._report_content["warning_messages"].append(msg)

            displayed_maps = n_maps

        displayed_maps = list(range(displayed_maps))

    if isinstance(displayed_maps, np.ndarray):
        displayed_maps = displayed_maps.tolist()

    available_maps = list(range(n_maps))

    # we can not rely on using set as we must preserve order
    unavailable_maps = [x for x in displayed_maps if x not in available_maps]
    displayed_maps = [x for x in displayed_maps if x in available_maps]

    if unavailable_maps:
        msg = (
            "`generate_report()` received "
            f"'displayed_{var_name}s={list(displayed_maps)}' to be displayed. "
            f"Report cannot display the following {var_name} "
            f"{unavailable_maps} because "
            f"masker only has {n_maps} {var_name}(s)."
        )
        estimator._report_content["warning_messages"].append(msg)

    return estimator, displayed_maps


@fill_doc
class BaseMasker(
    _ReportingMixin,
    TransformerMixin,
    CacheMixin,
    BaseEstimator,
):
    """Base class for NiftiMaskers."""

    _estimator_type = "masker"  # TODO (sklearn >= 1.8) remove

    _template_name = "body_masker.jinja"

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

            .. nilearn_versionadded:: 0.8.0

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
        tags.input_tags = InputTags()
        tags.estimator_type = "masker"
        return tags

    @property
    def _n_features_out(self):
        """Needed by sklearn machinery for set_ouput."""
        return self.n_elements_

    @abc.abstractmethod
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

            .. nilearn_versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_nifti)s
        """
        check_is_fitted(self)

        if (self.standardize == "zscore") or (self.standardize is True):
            # TODO (nilearn >= 0.14.0) remove or adapt warning
            warnings.warn(
                category=FutureWarning,
                message=STANDARIZE_WARNING_MESSAGE,
                stacklevel=find_stack_level(),
            )

        if confounds is None and not self.high_variance_confounds:
            # TODO (Nilearn >= 0.14.0) remove ignore FutureWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
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

        # TODO (Nilearn >= 0.14.0) remove ignore FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return self.transform_single_imgs(
                imgs, confounds=all_confounds, sample_mask=sample_mask
            )

    @fill_doc
    def fit_transform(
        self, imgs, y=None, confounds=None, sample_mask=None, **fit_params
    ):
        """Fit to data, then transform it.

        Parameters
        ----------
        imgs : Niimg-like object
            See :ref:`extracting_data`.

        %(y_dummy)s

        %(confounds)s

        %(sample_mask)s

            .. nilearn_versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_nifti)s

        """
        return self.fit(imgs, y, **fit_params).transform(
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


class _BaseSurfaceMasker(
    _ReportingMixin, TransformerMixin, CacheMixin, BaseEstimator
):
    """Class from which all surface maskers should inherit."""

    _estimator_type = "masker"  # TODO (sklearn >= 1.8) remove

    _template_name = "body_surface_masker.jinja"

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

            return tags(surf_img=True, niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        tags.estimator_type = "masker"
        return tags

    @property
    def _n_features_out(self):
        """Needed by sklearn machinery for set_ouput."""
        return self.n_elements_

    def _check_imgs(self, imgs) -> None:
        """Check that imgs is a SurfaceImage or an iterable of SurfaceImage."""
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

    @abc.abstractmethod
    def fit(self, imgs=None, y=None):
        """Present only to comply with sklearn estimators checks."""

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
                NotImplementedWarning,
                stacklevel=find_stack_level(),
            )
            self.smoothing_fwhm = None

        if (self.standardize == "zscore") or (self.standardize is True):
            # TODO (nilearn >= 0.14.0) remove or adapt warning
            warnings.warn(
                category=FutureWarning,
                message=STANDARIZE_WARNING_MESSAGE,
                stacklevel=find_stack_level(),
            )

        if self.reports:
            self._reporting_data["images"] = imgs

        if confounds is None and not self.high_variance_confounds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
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

        # TODO (Nilearn >= 0.14.0) remove ignore FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            signals = self.transform_single_imgs(
                imgs, confounds=all_confounds, sample_mask=sample_mask
            )

        sklearn_output_config = getattr(self, "_sklearn_output_config", None)

        return (
            signals.squeeze()
            if return_1D and sklearn_output_config is not None
            else signals
        )

    @abc.abstractmethod
    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single surface image."""
        # implemented in children classes
        raise NotImplementedError()

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

    def _generate_figure(
        self,
        img=None,
        bg_map=None,
        roi_map=None,
        vmin=None,
        vmax=None,
        threshold=None,
    ):
        """Create figure for all reports."""
        import matplotlib.pyplot as plt

        from nilearn.plotting import plot_surf, plot_surf_contours

        hemi_view = {
            "left": ["lateral", "medial"],
            "right": ["lateral", "medial"],
            "both": ["anterior", "posterior"],
        }

        fig, axes = plt.subplots(
            len(next(iter(hemi_view.values()))),
            len(hemi_view.keys()),
            subplot_kw={"projection": "3d"},
            figsize=(20, 20),
            layout="constrained",
        )

        axes = np.atleast_2d(axes)

        for j, (hemi, views) in enumerate(hemi_view.items()):
            for i, view in enumerate(views):
                if img:
                    plot_surf(
                        surf_map=img,
                        bg_map=bg_map,
                        hemi=hemi,
                        view=view,
                        figure=fig,
                        axes=axes[i, j],
                        cmap=self.cmap,
                        vmin=vmin,
                        vmax=vmax,
                        threshold=threshold,
                        bg_on_data=True,
                    )

                if roi_map:
                    colors = self._set_contour_colors(self)

                    plot_surf_contours(
                        roi_map=roi_map,
                        hemi=hemi,
                        view=view,
                        figure=fig,
                        axes=axes[i, j],
                        colors=colors,
                    )

        plt.close()
        return fig

    def _set_contour_colors(self, hemi):
        """Set the colors for the contours in the report."""
        del hemi
