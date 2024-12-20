"""Masker for surface objects."""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Memory

from nilearn import signal
from nilearn._utils import _constrained_layout_kwargs, fill_doc
from nilearn._utils.cache_mixin import cache
from nilearn._utils.class_inspect import get_params
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers._utils import (
    check_same_n_vertices,
    compute_mean_surface_image,
    concatenate_surface_images,
    get_min_max_surface_image,
)
from nilearn.maskers.base_masker import _BaseSurfaceMasker
from nilearn.surface import SurfaceImage


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

    clean_args : :obj:`dict` or None, default=None
        Keyword arguments to be passed
        to :func:`nilearn.signal.clean`
        called within the masker.

    Attributes
    ----------
    output_dimension_ : :obj:`int` or None
        number of vertices included in mask

    mask_img_ : :obj:`~nilearn.surface.SurfaceImage` or None
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
        cmap="inferno",
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
        self._shelving = False
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
        }
        # data necessary to construct figure for the report
        self._reporting_data = None

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "mask_img_")
            and hasattr(self, "output_dimension_")
            and self.mask_img_ is not None
            and self.output_dimension_ is not None
        )

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                "This masker has not been fitted.\n"
                "Call fit before calling transform."
            )

    def _fit_mask_img(self, img):
        """Get mask passed during init or compute one from input image.

        Parameters
        ----------
        img : SurfaceImage object or :obj:`list` of SurfaceImage or None
        """
        if img is None:
            if self.mask_img is None:
                raise ValueError(
                    "Please provide either a mask_img "
                    "when initializing the masker "
                    "or an img when calling fit()."
                )

            if self.mask_img is not None:
                self.mask_img_ = self.mask_img
                return

        if not isinstance(img, list):
            img = [img]
        img = concatenate_surface_images(img)

        if self.mask_img is not None:
            check_same_n_vertices(self.mask_img.mesh, img.mesh)
            self.mask_img_ = self.mask_img
            return

        # TODO: don't store a full array of 1 to mean "no masking"; use some
        # sentinel value
        mask_data = {
            part: np.ones((v.n_vertices, 1), dtype=bool)
            for (part, v) in img.mesh.parts.items()
        }
        self.mask_img_ = SurfaceImage(mesh=img.mesh, data=mask_data)

    def fit(self, img=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage` or None, \
              default = None
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        SurfaceMasker object
        """
        del y
        self._fit_mask_img(img)
        assert self.mask_img_ is not None

        start, stop = 0, 0
        self._slices = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            stop = start + mask.sum()
            self._slices[part_name] = start, stop
            start = stop
        self.output_dimension_ = stop

        if self.reports:
            for part in self.mask_img_.data.parts:
                self._report_content["n_vertices"][part] = (
                    self.mask_img_.mesh.parts[part].n_vertices
                )
            self._reporting_data = {
                "mask": self.mask_img_,
                "images": img,
            }

        return self

    def transform(
        self,
        img,
        confounds=None,
        sample_mask=None,
    ):
        """Extract signals from fitted surface object.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres.

        confounds : :class:`numpy.ndarray`, :obj:`str`,\
                    :class:`pathlib.Path`, \
                    :class:`pandas.DataFrame` \
                    or :obj:`list` of confounds timeseries, default=None
            Confounds to pass to :func:`nilearn.signal.clean`.

        sample_mask : None, Any type compatible with numpy-array indexing, \
                  or :obj:`list` of \
                  shape: (number of scans - number of volumes removed, ) \
                  for explicit index, or (number of scans, ) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.

        Returns
        -------
        2D :class:`numpy.ndarray`
            Signal for each element.
            shape: (n samples, total number of vertices)
        """
        if self.smoothing_fwhm is not None:
            warnings.warn(
                "Parameter smoothing_fwhm "
                "is not yet supported for surface data",
                UserWarning,
                stacklevel=2,
            )
            self.smoothing_fwhm = None

        parameters = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
            ],
        )
        if self.clean_args is None:
            self.clean_args = {}
        parameters["clean_args"] = self.clean_args

        self._check_fitted()

        if not isinstance(img, list):
            img = [img]
        img = concatenate_surface_images(img)

        check_same_n_vertices(self.mask_img_.mesh, img.mesh)

        if self.reports:
            self._reporting_data["images"] = img

        output = np.empty((1, self.output_dimension_))
        if len(img.shape) == 2:
            output = np.empty((img.shape[1], self.output_dimension_))
        for part_name, (start, stop) in self._slices.items():
            mask = self.mask_img_.data.parts[part_name].ravel()
            output[:, start:stop] = img.data.parts[part_name][mask].T

        if self.memory is None:
            self.memory = Memory(location=None)

        # signal cleaning here
        output = cache(
            signal.clean,
            memory=self.memory,
            func_memory_level=2,
            memory_level=self.memory_level,
            shelve=self._shelving,
        )(
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

    def fit_transform(
        self,
        img,
        y=None,
        confounds=None,
        sample_mask=None,
    ):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        confounds : :class:`numpy.ndarray`, :obj:`str`,\
                    :class:`pathlib.Path`, \
                    :class:`pandas.DataFrame` \
                    or :obj:`list` of confounds timeseries, default=None
            Confounds to pass to :func:`nilearn.signal.clean`.

        sample_mask : None, or any type compatible with numpy-array indexing, \
                  or :obj:`list` of \
                  shape: (number of scans - number of volumes removed) \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.

        Returns
        -------
        :class:`numpy.ndarray`
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit(img).transform(img, confounds, sample_mask)

    def inverse_transform(self, signals):
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        signals : :class:`numpy.ndarray`
            Extracted signal.

        Returns
        -------
        :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres.
        """
        self._check_fitted()

        if signals.ndim == 1:
            signals = np.array([signals])

        if signals.shape[1] != self.output_dimension_:
            raise ValueError(
                "Input to 'inverse_transform' has wrong shape.\n"
                f"Last dimension should be {self.output_dimension_}.\n"
                f"Got {signals.shape[1]}."
            )

        data = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            data[part_name] = np.zeros(
                (mask.shape[0], signals.shape[0]),
                dtype=signals.dtype,
            )
            start, stop = self._slices[part_name]
            data[part_name][mask.ravel()] = signals[:, start:stop].T

        return SurfaceImage(mesh=self.mask_img_.mesh, data=data)

    def generate_report(self):
        """Generate a report for the SurfaceMasker.

        Returns
        -------
        list(None) or HTMLReport
        """
        if not is_matplotlib_installed():
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    "Matplotlib is not imported! "
                    "No reports will be generated."
                )
                warnings.filterwarnings("always", message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

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
            background_data = compute_mean_surface_image(background_data)
            vmin, vmax = get_min_max_surface_image(background_data)

        views = ["lateral", "medial"]
        hemispheres = ["left", "right"]

        fig, axes = plt.subplots(
            len(views),
            len(hemispheres),
            subplot_kw={"projection": "3d"},
            figsize=(20, 20),
            **_constrained_layout_kwargs(),
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
