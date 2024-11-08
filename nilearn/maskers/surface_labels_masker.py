"""Extract data from a SurfaceImage, averaging over atlas regions."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator

from nilearn._utils import _constrained_layout_kwargs
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.experimental.surface._surface_image import SurfaceImage
from nilearn.maskers._utils import (
    check_same_n_vertices,
    compute_mean_surface_image,
    get_min_max_surface_image,
)


class SurfaceLabelsMasker(BaseEstimator):
    """Extract data from a SurfaceImage, averaging over atlas regions.

    Parameters
    ----------
    labels_img : SurfaceImage object
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

    %(verbose0)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default=`inferno`
        Only relevant for the report figures.

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
        verbose=0,
        reports=True,
        cmap="inferno",
    ):
        self.labels_img = labels_img
        self.labels = labels
        self.background_label = background_label
        self.verbose = verbose
        self.reports = reports
        self.cmap = cmap

    @property
    def _labels_data(self):
        """Return data of label image concatenated over hemispheres."""
        return np.concatenate(list(self.labels_img.data.parts.values()))

    def fit(self, img=None, y=None):
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object or None
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
            self._label_names_ = [str(label) for label in self._labels_]
        else:
            self._label_names_ = [self.labels[x] for x in self._labels_]

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

            for i, label in enumerate(self._label_names_):
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
            "label_names": [str(x) for x in self._label_names_],
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

    def transform(self, img):
        """Extract signals from surface object.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        Returns
        -------
        output : numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        self._check_fitted()

        if self.reports:
            self._reporting_data["images"] = img

        check_same_n_vertices(self.labels_img.mesh, img.mesh)
        img_data = np.concatenate(list(img.data.parts.values()), axis=-1)
        output = np.empty((*img_data.shape[:-1], len(self._labels_)))
        for i, label in enumerate(self._labels_):
            output[..., i] = img_data[..., self._labels_data == label].mean(
                axis=-1
            )
        return output

    def fit_transform(self, img, y=None):
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        Returns
        -------
        numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit().transform(img)

    def inverse_transform(self, masked_img):
        """Transform extracted signal back to surface object.

        Parameters
        ----------
        masked_img : numpy.ndarray
            Extracted signal.

        Returns
        -------
        SurfaceImage object
            Mesh and data for both hemispheres.
        """
        data = {}
        for part_name, labels_part in self.labels_img.data.parts.items():
            data[part_name] = np.zeros(
                (*masked_img.shape[:-1], labels_part.shape[0]),
                dtype=masked_img.dtype,
            )
            for label_idx, label in enumerate(self._labels_):
                data[part_name][..., labels_part == label] = masked_img[
                    ..., label_idx
                ]
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
