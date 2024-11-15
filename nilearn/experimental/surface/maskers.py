"""Masker for surface objects."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from nilearn._utils import _constrained_layout_kwargs
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers._utils import (
    check_same_n_vertices,
    compute_mean_surface_image,
    get_min_max_surface_image,
)
from nilearn.surface import SurfaceImage


class SurfaceLabelsMasker(BaseEstimator):
    """Extract data from a SurfaceImage, averaging over atlas regions.

    Parameters
    ----------
    labels_img : SurfaceImage object
        Region definitions, as one image of labels.

    label_names : :obj:`list` of :obj:`str`, default=None
        Full labels corresponding to the labels image.

    Attributes
    ----------
    labels_data_ : :obj:`numpy.ndarray`

    labels_ : :obj:`numpy.ndarray`

    label_names_ : :obj:`numpy.ndarray`

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.experimental.surface import (
    ...     SurfaceImage,
    ...     SurfaceLabelsMasker,
    ... )
    >>> from nilearn.experimental.surface._datasets import toy_mesh

    >>> (mesh := toy_mesh())
    <PolyMesh with 6 vertices>
    >>> left_data = np.asarray([[0.1, 1.1], [0.1, 1.1], [0.2, 2.2]])
    >>> data = {"left": left_data, "right": -left_data}
    >>> img = SurfaceImage(mesh=mesh, data=data)
    >>> img
    <SurfaceImage (6, 2)>
    >>> img.data.parts["left"]
    array([[0.1, 1.1],
           [0.1, 1.1],
           [0.2, 2.2]])
    >>> labels = {
    ...     "left": np.asarray([10, 10, 20]),
    ...     "right": np.asarray([30, 40, 40]),
    ... }
    >>> labels_img = SurfaceImage(mesh=mesh, data=labels)
    >>> masker = SurfaceLabelsMasker(labels_img)
    >>> (masked := masker.fit_transform(img))
    array([[-0.15,  0.1 ,  0.2 , -0.1 ],
           [-1.65,  1.1 ,  2.2 , -1.1 ]])
    >>> masker.label_names_
    array(['40', '10', '20', '30'], dtype='<U2')
    >>> masked[:, masker.labels_ == 10]
    array([[0.1],
           [1.1]])
    >>> (unmasked := masker.inverse_transform(masked))
    <SurfaceImage (6, 2)>
    >>> unmasked.data.parts["left"]
    array([[0.1, 1.1],
           [0.1, 1.1],
           [0.2, 2.2]])
    >>> unmasked.data.parts["right"]
    array([[-0.1 , -1.1 ],
           [-0.15, -1.65],
           [-0.15, -1.65]])
    """

    # TODO check attribute names after PR 3761 and harmonize with volume labels
    # masker if necessary.
    labels_img: SurfaceImage
    label_names: dict[Any, str] | None

    labels_data_: type[np.ndarray]
    labels_: type[np.ndarray]
    label_names_: type[np.ndarray]

    def __init__(
        self,
        labels_img: SurfaceImage,
        label_names: dict[Any, str] | None = None,
        reports: bool = True,
        **kwargs,
    ) -> None:
        # TODO this does not respect the scikit-learn requirement that __init__
        # only stores the arguments without doing anything else, so set_params
        # will not work with this estimator. Also make kwargs explicit.
        self.labels_img = labels_img
        self.label_names = label_names
        self.labels_data_ = np.concatenate(
            list(labels_img.data.parts.values())
        )
        all_labels = set(self.labels_data_.ravel())
        all_labels.discard(0)
        self.labels_ = np.asarray(list(all_labels))

        if label_names is None:
            self.label_names_ = np.asarray(
                [str(label) for label in self.labels_]
            )
        else:
            self.label_names_ = np.asarray(
                [label_names[x] for x in self.labels_]
            )

        self.reports = reports
        self.cmap = kwargs.get("cmap", "inferno")

        self._report_content = {
            "description": (
                "This report shows the input surface image overlaid "
                "with the outlines of the mask. "
                "We recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "n_vertices": {},
            "number_of_regions": len(self.label_names_),
            "summary": {},
        }
        for part in self.labels_img.data.parts:
            self._report_content["n_vertices"][part] = (
                self.labels_img.mesh.parts[part].n_vertices
            )

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
                relative_size.append(f"{tmp:.2}")

            regions_summary["size<br>(number of vertices)"] = size
            regions_summary["relative size<br>(% vertices in hemisphere)"] = (
                relative_size
            )

            self._report_content["summary"][part] = regions_summary

        self._reporting_data = {
            "labels_image": self.labels_img,
            "label_names": [str(x) for x in self.label_names_],
            "images": None,
        }

    def fit(
        self, img: SurfaceImage | None = None, y: Any = None
    ) -> SurfaceLabelsMasker:
        """Prepare signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        SurfaceLabelsMasker object
        """
        del img, y

        return self

    def transform(self, img: SurfaceImage) -> np.ndarray:
        """Extract signals from fitted surface object.

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
        if not self.reports:
            self._reporting_data = None
        else:
            self._reporting_data["images"] = img

        check_same_n_vertices(self.labels_img.mesh, img.mesh)
        img_data = np.concatenate(list(img.data.parts.values()), axis=0)
        output = np.empty((*img_data.shape[1:], len(self.labels_)))
        for i, label in enumerate(self.labels_):
            output[..., i] = img_data[self.labels_data_ == label].mean(axis=0)
        return output

    def fit_transform(self, img: SurfaceImage, y: Any = None) -> np.ndarray:
        """Prepare and perform signal extraction from regions.

        Parameters
        ----------
        img : SurfaceImage object
            Mesh and data for both hemispheres.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        Returns
        -------
        numpy.ndarray
            Signal for each element.
            shape: (img data shape, total number of vertices)
        """
        del y
        return self.fit(img).transform(img)

    def inverse_transform(self, masked_img: np.ndarray) -> SurfaceImage:
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
                (labels_part.shape[0], *masked_img.shape[:-1]),
                dtype=masked_img.dtype,
            )
            for label_idx, label in enumerate(self.labels_):
                data[part_name][labels_part == label] = masked_img[
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
        import matplotlib.pyplot as plt

        from nilearn import plotting

        labels_img = self._reporting_data["labels_image"]

        img = self._reporting_data["images"]
        if img:
            img = compute_mean_surface_image(img)
            vmin, vmax = get_min_max_surface_image(img)

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
                    plotting.plot_surf(
                        surf_map=img,
                        hemi=hemi,
                        view=view,
                        figure=fig,
                        axes=ax,
                        cmap=self.cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                plotting.plot_surf_contours(
                    roi_map=labels_img,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                )

        return fig
