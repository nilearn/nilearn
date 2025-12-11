"""Extract data from a SurfaceImage, using maps of potentially overlapping
brain regions.
"""

import warnings
from typing import Literal

import numpy as np
from scipy import linalg
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn import DEFAULT_SEQUENTIAL_CMAP, signal
from nilearn._utils.class_inspect import get_params
from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import (
    is_matplotlib_installed,
    is_plotly_installed,
)
from nilearn._utils.logger import find_stack_level
from nilearn._utils.masker_validation import (
    check_compatibility_mask_and_images,
)
from nilearn._utils.param_validation import (
    check_parameter_in_allowed,
    check_params,
)
from nilearn.image import index_img, mean_img
from nilearn.maskers.base_masker import (
    _BaseSurfaceMasker,
    check_displayed_maps,
    mask_logger,
    sanitize_displayed_maps,
)
from nilearn.surface.surface import (
    SurfaceImage,
    at_least_2d,
    check_surf_img,
    get_data,
)
from nilearn.surface.utils import check_polymesh_equal


@fill_doc
class SurfaceMapsMasker(ClassNamePrefixFeaturesOutMixin, _BaseSurfaceMasker):
    """Extract data from a SurfaceImage, using maps of potentially overlapping
    brain regions.

    .. nilearn_versionadded:: 0.11.1

    Parameters
    ----------
    maps_img : :obj:`~nilearn.surface.SurfaceImage`
        Set of maps that define the regions. A representative time course \
        per map is extracted using least square regression. The data for \
        each hemisphere is of shape (n_vertices_per_hemisphere, n_regions).

    mask_img : :obj:`~nilearn.surface.SurfaceImage`, optional, default=None
        Mask to apply to regions before extracting signals. Defines the \
        overall area of the brain to consider. The data for each \
        hemisphere is of shape (n_vertices_per_hemisphere, n_regions).

    allow_overlap : :obj:`bool`, default=True
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel).

    %(smoothing_fwhm)s
        This parameter is not implemented yet.

    %(standardize_false)s

    %(standardize_confounds)s

    %(detrend)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image \
        with :func:`nilearn.image.high_variance_confounds` and default \
        parameters and regressed out.

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

    maps_img_ : :obj:`~nilearn.surface.SurfaceImage`
        The same as the input `maps_img`, kept solely for consistency
        across maskers.

    mask_img_ : A 1D binary :obj:`~nilearn.surface.SurfaceImage` or None.
        The mask of the data.
        If no ``mask_img`` was passed at masker construction,
        then ``mask_img_`` is ``None``, otherwise
        is the resulting binarized version of ``mask_img``
        where each vertex is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0.

    memory_ : joblib memory cache

    n_elements_ : :obj:`int`
        The number of regions in the maps image.

    See Also
    --------
        nilearn.maskers.SurfaceMasker
        nilearn.maskers.SurfaceLabelsMasker

    """

    _template_name = "body_surface_maps_masker.jinja"

    def __init__(
        self,
        maps_img=None,
        mask_img=None,
        allow_overlap=True,
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
        self.maps_img = maps_img
        self.mask_img = mask_img
        self.allow_overlap = allow_overlap
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

        self._report_content = {
            "description": (
                "This report shows the input surface image "
                "(if provided via img) overlaid with the regions provided "
                "via maps_img."
            ),
            "n_vertices": {},
            "number_of_regions": getattr(self, "n_elements_", 0),
            "displayed_maps": [],
            "number_of_maps": 0,
            "summary": {},
            "warning_messages": [],
        }

    @fill_doc
    def fit(self, imgs=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`~nilearn.surface.SurfaceImage` object or None, \
               default=None

        %(y_dummy)s

        Returns
        -------
        SurfaceMapsMasker object
        """
        del y
        check_params(self.__dict__)

        # Reset warning message
        # in case where the masker was previously fitted
        self._report_content["warning_messages"] = []

        if imgs is not None:
            self._check_imgs(imgs)

            if isinstance(imgs, SurfaceImage) and any(
                hemi.ndim > 2 for hemi in imgs.data.parts.values()
            ):
                raise ValueError(
                    "should only be SurfaceImage should 1D or 2D."
                )
            elif hasattr(imgs, "__iter__"):
                for i, x in enumerate(imgs):
                    x.data._check_n_samples(1, f"imgs[{i}]")

        return self._fit(imgs)

    def _fit(self, imgs):
        if self.maps_img is None:
            raise ValueError(
                "Please provide a maps_img during initialization. "
                "For example, masker = SurfaceMapsMasker(maps_img=maps_img)"
            )

        if imgs is not None:
            check_surf_img(imgs)

        self._fit_cache()

        mask_logger("load_regions", self.maps_img, verbose=self.verbose)

        # check maps_img data is 2D
        self.maps_img.data._check_ndims(2, "maps_img")
        self.maps_img_ = self.maps_img

        self.n_elements_ = self.maps_img.shape[1]

        self.mask_img_ = self._load_mask(imgs)
        if self.mask_img_ is not None:
            check_polymesh_equal(self.maps_img.mesh, self.mask_img_.mesh)

        self._report_content["reports_at_fit_time"] = self.reports
        # initialize reporting content and data
        if self.reports:
            for part in self.maps_img.data.parts:
                self._report_content["n_vertices"][part] = (
                    self.maps_img.mesh.parts[part].n_vertices
                )

            self._report_content["number_of_regions"] = self.n_elements_

            self._reporting_data = {
                "maps_image": self.maps_img_,
                "mask": self.mask_img_,
                "images": None,  # we will update image in transform
            }

            if self.clean_args is None:
                self.clean_args_ = {}
            else:
                self.clean_args_ = self.clean_args

        mask_logger("fit_done", verbose=self.verbose)

        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_elements_")

    @fill_doc
    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from surface object.

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

        check_compatibility_mask_and_images(self.maps_img, imgs)
        check_polymesh_equal(self.maps_img.mesh, imgs.mesh)

        if isinstance(imgs, SurfaceImage) and any(
            hemi.ndim > 2 for hemi in imgs.data.parts.values()
        ):
            raise ValueError("should only be SurfaceImage should 1D or 2D.")
        elif hasattr(imgs, "__iter__"):
            for i, x in enumerate(imgs):
                x.data._check_n_samples(1, f"imgs[{i}]")

        imgs = at_least_2d(imgs)

        self._reporting_data["images"] = imgs

        img_data = np.concatenate(
            list(imgs.data.parts.values()), axis=0
        ).astype(np.float32)

        # get concatenated hemispheres/parts data from maps_img and mask_img
        maps_data = get_data(self.maps_img)
        mask_data = (
            get_data(self.mask_img_) if self.mask_img_ is not None else None
        )

        parameters = get_params(self.__class__, self)
        parameters["clean_args"] = self.clean_args_

        # apply mask if provided
        # and then extract signal via least square regression
        mask_logger("extracting", verbose=self.verbose)
        if mask_data is not None:
            region_signals = self._cache(linalg.lstsq, func_memory_level=2)(
                maps_data[mask_data.flatten(), :],
                img_data[mask_data.flatten(), :],
            )[0].T
        # if no mask, directly extract signal
        else:
            region_signals = self._cache(linalg.lstsq, func_memory_level=2)(
                maps_data, img_data
            )[0].T

        mask_logger("cleaning", verbose=self.verbose)

        parameters = get_params(self.__class__, self)

        parameters["clean_args"] = self.clean_args_

        # signal cleaning here
        region_signals = self._cache(signal.clean, func_memory_level=2)(
            region_signals,
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

        return region_signals

    @fill_doc
    def inverse_transform(self, region_signals):
        """Compute :term:`vertex` signals from region signals.

        Parameters
        ----------
        %(region_signals_inv_transform)s

        Returns
        -------
        %(img_inv_transform_surface)s
        """
        check_is_fitted(self)

        return_1D = region_signals.ndim < 2

        region_signals = self._check_array(region_signals)

        # get concatenated hemispheres/parts data from maps_img and mask_img
        maps_data = get_data(self.maps_img)
        mask_data = (
            get_data(self.mask_img) if self.mask_img is not None else None
        )
        if region_signals.shape[1] != self.n_elements_:
            raise ValueError(
                f"Expected {self.n_elements_} regions, "
                f"but got {region_signals.shape[1]}."
            )

        mask_logger("inverse_transform", verbose=self.verbose)

        # project region signals back to vertices
        if mask_data is not None:
            # vertices that are not in the mask will have a signal of 0
            # so we initialize the vertex signals with 0
            # and shape (n_timepoints, n_vertices)
            vertex_signals = np.zeros(
                (region_signals.shape[0], self.maps_img.mesh.n_vertices)
            )
            # dot product between (n_timepoints, n_regions) and
            # (n_regions, n_vertices)
            vertex_signals[:, mask_data.flatten()] = np.dot(
                region_signals, maps_data[mask_data.flatten(), :].T
            )
        else:
            vertex_signals = np.dot(region_signals, maps_data.T)

        # we need the data to be of shape (n_vertices, n_timepoints)
        # because the SurfaceImage object expects it
        vertex_signals = vertex_signals.T

        # split the signal into hemispheres
        vertex_signals = {
            "left": vertex_signals[
                : self.maps_img.data.parts["left"].shape[0], :
            ],
            "right": vertex_signals[
                self.maps_img.data.parts["left"].shape[0] :, :
            ],
        }

        imgs = SurfaceImage(mesh=self.maps_img.mesh, data=vertex_signals)

        if return_1D:
            for k, v in imgs.data.parts.items():
                imgs.data.parts[k] = v.squeeze()

        return imgs

    @fill_doc
    def generate_report(
        self,
        displayed_maps: list[int]
        | np.typing.NDArray[np.int_]
        | int
        | Literal["all"] = 10,
        engine: str = "matplotlib",
        title: str | None = None,
    ):
        """Generate an HTML report for the current ``SurfaceMapsMasker``
        object.

        .. note::
            This functionality requires to have ``Matplotlib`` installed.

        Parameters
        ----------
        %(displayed_maps)s

        title : :obj:`str` or None, default=None
            title for the report. If None, title will be the class name.

        engine : :obj:`str`, default="matplotlib"
            The plotting engine to use for the report. Can be either
            "matplotlib" or "plotly". If "matplotlib" is selected, the report
            will be static. If "plotly" is selected, the report
            will be interactive. If the selected engine is not installed, the
            report will use the available plotting engine. If none of the
            engines are installed, no report will be generated.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        check_displayed_maps(displayed_maps)

        self._report_content["number_of_maps"] = 0
        self._report_content["displayed_maps"] = []

        if self._has_report_data():
            maps_image = self._reporting_data["maps_image"]
            n_maps = maps_image.shape[1]

            self._report_content["number_of_maps"] = n_maps

            self, maps_to_be_displayed = sanitize_displayed_maps(
                self, displayed_maps, n_maps
            )

            self._report_content["displayed_maps"] = maps_to_be_displayed

            if self._reporting_data.get("images") is None:
                msg = (
                    "SurfaceMapsMasker has not been transformed "
                    "(via transform() method) on any image yet. "
                    "Plotting only maps for reporting."
                )
                self._report_content["warning_messages"].append(msg)

        # need to have matplotlib installed to generate reports no matter what
        # engine is selected
        if is_matplotlib_installed():
            check_parameter_in_allowed(
                engine, ["plotly", "matplotlib"], "engine"
            )

            # switch to matplotlib if plotly is selected but not installed
            if engine == "plotly" and not is_plotly_installed():
                engine = "matplotlib"
                warnings.warn(
                    "Plotly is not installed. "
                    "Switching to matplotlib for report generation.",
                    stacklevel=find_stack_level(),
                )
            self._report_content["engine"] = engine

        return super().generate_report(title)

    def _reporting(self) -> list:
        """Load displays needed for report.

        Returns
        -------
        displays : list
            A list of all displays to be rendered.
        """
        # Handle the edge case where this function is called
        # without matplolib or
        # with a masker having report capabilities disabled
        if not is_matplotlib_installed() or not self._has_report_data():
            return [None]

        from nilearn.reporting.utils import figure_to_png_base64

        img = self._reporting_data["images"]
        if img:
            img = mean_img(img)

        maps_image = self._reporting_data["maps_image"]

        embeded_images = []

        for roi in self._report_content["displayed_maps"]:
            roi = index_img(maps_image, roi)
            fig = self._create_figure_for_report(roi=roi, bg_img=img)[0]
            if self._report_content["engine"] == "plotly":
                embeded_images.append(fig)
            elif self._report_content["engine"] == "matplotlib":
                embeded_images.append(figure_to_png_base64(fig))

        return embeded_images

    def _create_figure_for_report(self, roi, bg_img) -> list:
        """Create a figure of maps image, one region at a time.

        If transform() was applied to an image, this image is used as
        background on which the maps are plotted.

        Returns
        -------
        list of :class:`~matplotlib.figure.Figure` or None
        """
        threshold = 1e-6

        if self._report_content["engine"] == "plotly":
            from nilearn.plotting import view_surf

            # squeeze the last dimension
            for part in roi.data.parts:
                roi.data.parts[part] = np.squeeze(
                    roi.data.parts[part], axis=-1
                )
            fig = view_surf(
                surf_map=roi,
                bg_map=bg_img,
                bg_on_data=True,
                threshold=threshold,
                hemi="both",
                cmap=self.cmap,
            ).get_iframe(width=500)

        elif self._report_content["engine"] == "matplotlib":
            fig = self._generate_figure(
                img=roi, bg_map=bg_img, threshold=threshold
            )

        return [fig]
