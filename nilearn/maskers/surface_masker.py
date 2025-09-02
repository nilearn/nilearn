"""Masker for surface objects."""

from __future__ import annotations

from copy import deepcopy
from warnings import warn

import numpy as np
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn import DEFAULT_SEQUENTIAL_CMAP, signal
from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import (
    constrained_layout_kwargs,
    rename_parameters,
)
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.param_validation import check_params
from nilearn.image import concat_imgs, mean_img
from nilearn.maskers.base_masker import _BaseSurfaceMasker, mask_logger
from nilearn.surface.surface import SurfaceImage, at_least_2d, check_surf_img
from nilearn.surface.utils import check_polymesh_equal


@fill_doc
class SurfaceMasker(_BaseSurfaceMasker):
    """Extract data from a :obj:`~nilearn.surface.SurfaceImage`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    mask_img : :obj:`~nilearn.surface.SurfaceImage` or None, default=None

    %(smoothing_fwhm)s
        This parameter is not implemented yet.

    %(standardize_maskers)s

    %(standardize_confounds)s

    %(detrend)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out.

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="inferno"
        Only relevant for the report figures.

    %(clean_args)s

    Attributes
    ----------
    %(clean_args_)s

    mask_img_ : A 1D binary :obj:`~nilearn.surface.SurfaceImage`
        The mask of the data, or the one computed from ``imgs`` passed to fit.
        If a ``mask_img`` is passed at masker construction,
        then ``mask_img_`` is the resulting binarized version of it
        where each vertex is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

    n_elements_ : :obj:`int` or None
        number of vertices included in mask

    """

    def __init__(
        self,
        mask_img=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=None,
        memory_level=1,
        verbose=0,
        reports=True,
        cmap=DEFAULT_SEQUENTIAL_CMAP,
        clean_args=None,
    ):
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.reports = reports
        self.cmap = cmap
        self.clean_args = clean_args
        # content to inject in the HTML template
        self._report_content = {
            "description": (
                "This report shows the input surface image overlaid "
                "with the outlines of the mask. "
                "We recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "n_vertices": {},
            # unused but required in HTML template
            "number_of_regions": None,
            "summary": None,
            "warning_message": None,
            "n_elements": 0,
            "coverage": 0,
        }
        # data necessary to construct figure for the report
        self._reporting_data = None

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "mask_img_")
            and hasattr(self, "n_elements_")
            and self.mask_img_ is not None
            and self.n_elements_ is not None
        )

    def _fit_mask_img(self, img):
        """Get mask passed during init or compute one from input image.

        Parameters
        ----------
        img : SurfaceImage object or :obj:`list` of SurfaceImage or None
        """
        self.mask_img_ = self._load_mask(img)

        if self.mask_img_ is not None:
            if img is not None:
                warn(
                    f"[{self.__class__.__name__}.fit] "
                    "Generation of a mask has been"
                    " requested (y != None) while a mask was"
                    " given at masker creation. Given mask"
                    " will be used.",
                    stacklevel=find_stack_level(),
                )
            return

        if img is None:
            raise ValueError(
                "Parameter 'imgs' must be provided to "
                f"{self.__class__.__name__}.fit() "
                "if no mask is passed to mask_img."
            )

        mask_logger("compute_mask", verbose=self.verbose)

        img = deepcopy(img)
        if not isinstance(img, list):
            img = [img]
        img = concat_imgs(img)

        img = at_least_2d(img)

        check_surf_img(img)

        mask_data = {}
        for part, v in img.data.parts.items():
            # mask out vertices with NaN or infinite values
            mask_data[part] = np.isfinite(v.astype("float32")).all(axis=1)
            if not mask_data[part].all():
                warn(
                    "Non-finite values detected in the input image. "
                    "The computed mask will mask out these vertices.",
                    stacklevel=find_stack_level(),
                )
        self.mask_img_ = SurfaceImage(mesh=img.mesh, data=mask_data)

    # TODO (nilearn >= 0.13.0)
    @rename_parameters(
        replacement_params={"img": "imgs"}, end_version="0.13.0"
    )
    @fill_doc
    def fit(self, imgs=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage` or None, \
              default = None
            Mesh and data for both hemispheres.

        %(y_dummy)s

        Returns
        -------
        SurfaceMasker object
        """
        del y
        check_params(self.__dict__)
        if imgs is not None:
            self._check_imgs(imgs)

        self._fit_cache()

        self._fit_mask_img(imgs)
        assert self.mask_img_ is not None

        start, stop = 0, 0
        self._slices = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            stop = start + mask.sum()
            self._slices[part_name] = start, stop
            start = stop
        self.n_elements_ = int(stop)

        if self.reports:
            self._report_content["n_elements"] = self.n_elements_
            for part in self.mask_img_.data.parts:
                self._report_content["n_vertices"][part] = (
                    self.mask_img_.mesh.parts[part].n_vertices
                )
            self._report_content["coverage"] = (
                self.n_elements_ / self.mask_img_.mesh.n_vertices * 100
            )
            self._reporting_data = {
                "mask": self.mask_img_,
                "images": imgs,
            }

        if self.clean_args is None:
            self.clean_args_ = {}
        else:
            self.clean_args_ = self.clean_args

        mask_logger("fit_done", verbose=self.verbose)

        return self

    @fill_doc
    def transform_single_imgs(
        self,
        imgs,
        confounds=None,
        sample_mask=None,
    ):
        """Extract signals from fitted surface object.

        Parameters
        ----------
        imgs : imgs : :obj:`~nilearn.surface.SurfaceImage` object or \
              iterable of :obj:`~nilearn.surface.SurfaceImage`
            Images to process.
            Mesh and data for both hemispheres/parts.

        %(confounds)s

        %(sample_mask)s

        Returns
        -------
        %(signals_transform_surface)s

        """
        check_is_fitted(self)

        check_compatibility_mask_and_images(self.mask_img_, imgs)

        check_polymesh_equal(self.mask_img_.mesh, imgs.mesh)

        if self.reports:
            self._reporting_data["images"] = imgs

        mask_logger("extracting", verbose=self.verbose)

        output = np.empty((1, self.n_elements_))
        if len(imgs.shape) == 2:
            output = np.empty((imgs.shape[1], self.n_elements_))
        for part_name, (start, stop) in self._slices.items():
            mask = self.mask_img_.data.parts[part_name].ravel()
            output[:, start:stop] = imgs.data.parts[part_name][mask].T

        mask_logger("cleaning", verbose=self.verbose)

        parameters = get_params(self.__class__, self, ignore=["mask_img"])

        parameters["clean_args"] = self.clean_args_

        # signal cleaning here
        output = self._cache(signal.clean, func_memory_level=2)(
            output,
            detrend=parameters["detrend"],
            standardize=parameters["standardize"],
            standardize_confounds=parameters["standardize_confounds"],
            t_r=parameters["t_r"],
            low_pass=parameters["low_pass"],
            high_pass=parameters["high_pass"],
            confounds=confounds,
            sample_mask=sample_mask,
            **parameters["clean_args"],
        )

        return output

    @fill_doc
    def inverse_transform(self, signals):
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        %(signals_inv_transform)s

        Returns
        -------
        %(img_inv_transform_surface)s
        """
        check_is_fitted(self)

        return_1D = signals.ndim < 2

        # do not run sklearn_check as they may cause some failure
        # with some GLM inputs
        signals = self._check_array(signals, sklearn_check=False)

        mask_logger("inverse_transform", verbose=self.verbose)

        data = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            data[part_name] = np.zeros(
                (mask.shape[0], signals.shape[0]),
                dtype=signals.dtype,
            )
            start, stop = self._slices[part_name]
            data[part_name][mask.ravel()] = signals[:, start:stop].T
            if return_1D:
                data[part_name] = data[part_name].squeeze()

        return SurfaceImage(mesh=self.mask_img_.mesh, data=data)

    def generate_report(self):
        """Generate a report for the SurfaceMasker.

        Returns
        -------
        list(None) or HTMLReport
        """
        from nilearn.reporting.html_report import generate_report

        return generate_report(self)

    def _reporting(self):
        """Load displays needed for report.

        Returns
        -------
        displays : :obj:`list` of None or bytes
            A list of all displays figures encoded as bytes to be rendered.
            Or a list with a single None element.
        """
        # avoid circular import
        import matplotlib.pyplot as plt

        from nilearn.reporting.utils import figure_to_png_base64

        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if self._reporting_data is None:
            return [None]

        fig = self._create_figure_for_report()

        if not fig:
            return [None]

        plt.close()

        init_display = figure_to_png_base64(fig)

        return [init_display]

    def _create_figure_for_report(self):
        """Generate figure to include in the report.

        Returns
        -------
        None, :class:`~matplotlib.figure.Figure` or\
              :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`
            Returns ``None`` in case the masker was not fitted.
        """
        # avoid circular import
        import matplotlib.pyplot as plt

        from nilearn.plotting import plot_surf, plot_surf_contours

        if not self._reporting_data["images"] and not getattr(
            self, "mask_img_", None
        ):
            return None

        background_data = self.mask_img_
        vmin = None
        vmax = None
        if self._reporting_data["images"]:
            background_data = self._reporting_data["images"]
            background_data = mean_img(background_data)
            vmin, vmax = background_data.data._get_min_max()

        views = ["lateral", "medial"]
        hemispheres = ["left", "right"]

        fig, axes = plt.subplots(
            len(views),
            len(hemispheres),
            subplot_kw={"projection": "3d"},
            figsize=(20, 20),
            **constrained_layout_kwargs(),
        )
        axes = np.atleast_2d(axes)

        for ax_row, view in zip(axes, views):
            for ax, hemi in zip(ax_row, hemispheres):
                plot_surf(
                    surf_map=background_data,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                    cmap=self.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    darkness=None,
                )

                colors = None
                n_regions = len(np.unique(self.mask_img_.data.parts[hemi]))
                if n_regions == 1:
                    colors = "b"
                elif n_regions == 2:
                    colors = ["w", "b"]

                plot_surf_contours(
                    roi_map=self.mask_img_,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                    colors=colors,
                )

        return fig
