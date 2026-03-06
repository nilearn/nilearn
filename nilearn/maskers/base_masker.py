"""Transformer used to apply basic transformations on :term:`fMRI` data."""

import abc
import contextlib
import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from joblib import Memory
from sklearn.base import TransformerMixin
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.utils.validation import check_array

from nilearn._base import NilearnBaseEstimator
from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import stringify_path
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.niimg import ensure_finite_data, repr_niimgs, safe_get_data
from nilearn._utils.param_validation import (
    check_parameter_in_allowed,
    check_params,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.exceptions import NotImplementedWarning
from nilearn.image.image import (
    check_niimg,
    check_volume_for_fit,
    concat_imgs,
    high_variance_confounds,
    new_img_like,
    smooth_img,
)
from nilearn.image.resampling import resample_img
from nilearn.maskers._mixin import _ReportingMixin
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


def mask_logger(step, img=None, verbose=0) -> None:
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
class _BaseMasker(
    _ReportingMixin,
    TransformerMixin,
    CacheMixin,
    NilearnBaseEstimator,
):
    """Base class for NiftiMaskers."""

    _estimator_type = "masker"  # TODO (sklearn >= 1.8) remove

    @property
    def _n_features_out(self):
        """Needed by sklearn machinery for set_ouput."""
        return self.n_elements_

    @abc.abstractmethod
    def _check_imgs(self, imgs) -> None:
        """Check if the images specified are not empty and of correct type for
        this masker.
        """
        raise NotImplementedError()


@fill_doc
class BaseMasker(_BaseMasker):
    """Base class for NiftiMaskers."""

    _template_name = "body_masker.jinja"

    @fill_doc
    def fit(self, imgs=None, y=None):
        """Compute the mask corresponding to the data.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects or None, default=None
            See :ref:`extracting_data`.
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        %(y_dummy)s
        """
        del y
        check_params(self.__dict__)

        if imgs is not None:
            self._check_imgs(imgs)

        # Reset warning message
        # in case where the masker was previously fitted
        self._report_content["warning_messages"] = []

        self.clean_args_ = {} if self.clean_args is None else self.clean_args

        self._fit_cache()

        self.mask_img_ = self._load_mask(imgs)

        return self._fit(imgs)

    @abc.abstractmethod
    def _fit(self, imgs):
        """Compute the mask corresponding to the data.

        Should be implement in inheriting classes.
        """
        raise NotImplementedError()

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

    def _get_masker_params(self, ignore: None | list[str] = None, deep=False):
        """Get parameters for this masker.

        Very similar to the BaseEstimator.get_params() from sklearn
        but allows to avoid returning some keys.

        Parameters
        ----------
        ignore : None or list of strings
            Names of the parameters that are not returned.

        deep : bool, default=True
            If True, will return the parameters for this estimator
            and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            The dict of parameters.

        """
        _ignore = {"memory", "memory_level", "verbose", "copy", "n_jobs"}
        if ignore is not None:
            _ignore.update(ignore)

        params = {
            k: v
            for k, v in super().get_params(deep=deep).items()
            if k not in _ignore
        }

        return params

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

    def _check_imgs(self, imgs) -> None:
        check_volume_for_fit(imgs)

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
        self._check_imgs(imgs)

        if self.standardize in [True, False]:
            # TODO (nilearn >= 0.15.0) remove warning
            warnings.warn(
                category=FutureWarning,
                message=(
                    "boolean values for 'standardize' "
                    "will be deprecated in nilearn 0.15.0.\n"
                    "Use 'zscore_sample' instead of 'True' or "
                    "use 'None' instead of 'False'."
                ),
                stacklevel=find_stack_level(),
            )

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

    def _check_array(self, signals, sklearn_check: bool = True) -> np.ndarray:
        """Check array to inverse transform.

        Parameters
        ----------
        signals : array like (numpy array, pandas or polars DataFrame)

        sklearn_check : :obj:`bool`
            Run scikit learn check on input
        """
        if hasattr(signals, "to_numpy"):
            # convert pandas or polars dataframe to numpy
            signals = signals.to_numpy().squeeze()

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


class _BaseSurfaceMasker(_BaseMasker):
    """Class from which all surface maskers should inherit."""

    _template_name = "body_surface_masker.jinja"

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
            ensure_finite_data(mask[part])
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

        if self.standardize in [True, False]:
            # TODO (nilearn >= 0.15.0) remove warning
            warnings.warn(
                category=FutureWarning,
                message=(
                    "boolean values for 'standardize' "
                    "will be deprecated in nilearn 0.15.0.\n"
                    "Use 'zscore_sample' instead of 'True' or "
                    "use 'None' instead of 'False'."
                ),
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

        %(y_dummy)s

        %(confounds)s

        %(sample_mask)s


        Returns
        -------
        %(signals_transform_surface)s
        """
        del y
        return self.fit(imgs).transform(imgs, confounds, sample_mask)

    def _check_array(self, signals, sklearn_check: bool = True) -> np.ndarray:
        """Check array to inverse transform.

        Parameters
        ----------
        signals : array like (numpy array, pandas or polars DataFrame)

        sklearn_check : :obj:`bool`
            Run scikit learn check on input
        """
        if hasattr(signals, "to_numpy"):
            # convert pandas or polars dataframe to numpy
            signals = signals.to_numpy().squeeze()

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

    def _clean(
        self, region_signals: np.ndarray, confounds, sample_mask
    ) -> np.ndarray:
        """Clean extracted signal before \
            returning it at the end of transform.
        """
        mask_logger("cleaning", verbose=self.verbose)
        region_signals = self._cache(clean, func_memory_level=2)(
            region_signals,
            detrend=self.detrend,
            standardize=self.standardize,
            standardize_confounds=self.standardize_confounds,
            t_r=self.t_r,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            confounds=confounds,
            sample_mask=sample_mask,
            **self.clean_args_,
        )
        return region_signals
