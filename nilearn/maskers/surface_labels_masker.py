"""Extract data from a SurfaceImage, averaging over atlas regions."""

import warnings

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn import signal
from nilearn._utils import _constrained_layout_kwargs, fill_doc
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.class_inspect import get_params
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers._utils import (
    check_same_n_vertices,
    compute_mean_surface_image,
    concatenate_surface_images,
    get_min_max_surface_image,
)
from nilearn.surface import SurfaceImage


@fill_doc
class SurfaceLabelsMasker(TransformerMixin, CacheMixin, BaseEstimator):
    """Extract data from a SurfaceImage, averaging over atlas regions.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    labels_img : :obj:`~nilearn.surface.SurfaceImage` object
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, default=None
        Full labels corresponding to the labels image.
        This is used to improve reporting quality if provided.

        .. warning::
            The labels must be consistent with the label values
            provided through ``labels_img``.

    background_label : :obj:`int` or :obj:`float`, default=0
        Label used in labels_img to represent background.

        .. warning::

            This value must be consistent with label values
            and image provided.

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
    n_elements_ : :obj:`int`
        The number of discrete values in the mask.
        This is equivalent to the number of unique values in the mask image,
        ignoring the background value.
    """

    def __init__(
        self,
        labels_img,
        labels=None,
        background_label=0,
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
        self.labels_img = labels_img
        self.labels = labels
        self.background_label = background_label
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
            "number_of_regions": 0,
            "summary": {},
        }

    @property
    def _labels_data(self):
        """Return data of label image concatenated over hemispheres."""
        all_labels = [x.ravel() for x in self.labels_img.data.parts.values()]
        return np.concatenate(all_labels)

    def fit(self, img=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or None, default=None
            This parameter is currently unused.

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        SurfaceLabelsMasker object
        """
        del img, y

        all_labels = set(self._labels_data.ravel())
        all_labels.discard(self.background_label)
        self._labels_ = list(all_labels)

        self.n_elements_ = len(self._labels_)

        if self.labels is None:
            self.label_names_ = [str(label) for label in self._labels_]
        else:
            self.label_names_ = [self.labels[x] for x in self._labels_]

        if not self.reports:
            self._reporting_data = None
            return self

        self._report_content["number_of_regions"] = self.n_elements_
        for part in self.labels_img.data.parts:
            self._report_content["n_vertices"][part] = (
                self.labels_img.mesh.parts[part].n_vertices
            )

        self._reporting_data = self._generate_reporting_data()

        return self

    def _generate_reporting_data(self):
        for part in self.labels_img.data.parts:
            size = []
            relative_size = []
            regions_summary = {
                "label value": [],
                "region name": [],
                "size<br>(number of vertices)": [],
                "relative size<br>(% vertices in hemisphere)": [],
            }

            for i, label in enumerate(self.label_names_):
                regions_summary["label value"].append(i)
                regions_summary["region name"].append(label)

                n_vertices = self.labels_img.data.parts[part] == i
                size.append(n_vertices.sum())
                tmp = (
                    n_vertices.sum()
                    / self.labels_img.mesh.parts[part].n_vertices
                    * 100
                )
                relative_size.append(f"{tmp :.2}")

            regions_summary["size<br>(number of vertices)"] = size
            regions_summary["relative size<br>(% vertices in hemisphere)"] = (
                relative_size
            )

            self._report_content["summary"][part] = regions_summary

        return {
            "labels_image": self.labels_img,
            "label_names": [str(x) for x in self.label_names_],
            "images": None,
        }

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_elements_")

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                f"It seems that {self.__class__.__name__} "
                "has not been fitted."
            )

    def transform(self, img, confounds=None, sample_mask=None):
        """Extract signals from surface object.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or \
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
                  shape: (number of scans - number of volumes removed) \
                  for explicit index, or (number of scans) for binary mask, \
                  default=None
            sample_mask to pass to :func:`nilearn.signal.clean`.

        Returns
        -------
        output : :obj:`numpy.ndarray`
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        self._check_fitted()

        # if img is a single image, convert it to a list
        # to be able to concatenate it
        if not isinstance(img, list):
            img = [img]
        img = concatenate_surface_images(img)
        check_same_n_vertices(self.labels_img.mesh, img.mesh)
        # concatenate data over hemispheres
        img_data = np.concatenate(list(img.data.parts.values()), axis=0)

        if self.smoothing_fwhm is not None:
            warnings.warn(
                "Parameter smoothing_fwhm "
                "is not yet supported for surface data",
                UserWarning,
                stacklevel=2,
            )
            self.smoothing_fwhm = None

        if self.reports:
            self._reporting_data["images"] = img

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

        if self.memory is None:
            self.memory = Memory(location=None)

        n_time_points = 1 if len(img_data.shape) == 1 else img_data.shape[1]
        output = np.empty((n_time_points, len(self._labels_)))
        for i, label in enumerate(self._labels_):
            output[:, i] = img_data[self._labels_data == label].mean(axis=0)

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

    def fit_transform(self, img, y=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : :obj:`~nilearn.surface.SurfaceImage` object or \
              :obj:`list` of :obj:`~nilearn.surface.SurfaceImage` or \
              :obj:`tuple` of :obj:`~nilearn.surface.SurfaceImage`
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        :obj:`numpy.ndarray`
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit().transform(img)

    def inverse_transform(self, masked_img):
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        masked_img : :obj:`numpy.ndarray`
            Extracted signal.

        Returns
        -------
        :obj:`~nilearn.surface.SurfaceImage` object
            Mesh and data for both hemispheres.
        """
        data = {}
        for part_name, labels_part in self.labels_img.data.parts.items():
            data[part_name] = np.zeros(
                (labels_part.shape[0], masked_img.shape[0]),
                dtype=masked_img.dtype,
            )
            for label_idx, label in enumerate(self._labels_):
                data[part_name][labels_part == label] = masked_img[
                    :, label_idx
                ].T
        return SurfaceImage(mesh=self.labels_img.mesh, data=data)

    def generate_report(self):
        """Generate a report."""
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
        displays : list
            A list of all displays to be rendered.
        """
        import matplotlib.pyplot as plt

        from nilearn.reporting.utils import figure_to_png_base64

        # Handle the edge case where this function is
        # called with a masker having report capabilities disabled
        if self._reporting_data is None:
            return [None]

        fig = self._create_figure_for_report()

        plt.close()

        init_display = figure_to_png_base64(fig)

        return [init_display]

    def _create_figure_for_report(self):
        """Create a figure of the contours of label image.

        If transform() was applied to an image,
        this image is used as background
        on which the contours are drawn.
        """
        import matplotlib.pyplot as plt

        from nilearn.plotting import plot_surf, plot_surf_contours

        labels_img = self._reporting_data["labels_image"]

        img = self._reporting_data["images"]
        if img:
            img = compute_mean_surface_image(img)
            vmin, vmax = get_min_max_surface_image(img)

        # TODO: possibly allow to generate a report with other views
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
                if img:
                    plot_surf(
                        surf_map=img,
                        hemi=hemi,
                        view=view,
                        figure=fig,
                        axes=ax,
                        cmap=self.cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                plot_surf_contours(
                    roi_map=labels_img,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                )

        return fig
