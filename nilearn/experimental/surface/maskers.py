"""Masker for surface objects."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn import signal
from nilearn._utils import _constrained_layout_kwargs
from nilearn._utils.cache_mixin import CacheMixin, cache
from nilearn._utils.class_inspect import get_params
from nilearn.experimental.surface._surface_image import PolyMesh, SurfaceImage


def check_same_n_vertices(mesh_1: PolyMesh, mesh_2: PolyMesh) -> None:
    """Check that 2 meshes have the same keys and that n vertices match."""
    keys_1, keys_2 = set(mesh_1.parts.keys()), set(mesh_2.parts.keys())
    if keys_1 != keys_2:
        diff = keys_1.symmetric_difference(keys_2)
        raise ValueError(
            "Meshes do not have the same keys. " f"Offending keys: {diff}"
        )
    for key in keys_1:
        if mesh_1.parts[key].n_vertices != mesh_2.parts[key].n_vertices:
            raise ValueError(
                f"Number of vertices do not match for '{key}'."
                "number of vertices in mesh_1: "
                f"{mesh_1.parts[key].n_vertices}; "
                f"in mesh_2: {mesh_2.parts[key].n_vertices}"
            )


def _compute_mean_image(img: SurfaceImage):
    """Compute mean of the surface (for 'time series')."""
    if len(img.shape) <= 1:
        return img
    new_data = {}
    for part, value in img.data.parts.items():
        new_data[part] = (
            value.mean(axis=tuple(range(1, len(value.shape))))
        ).astype(float)
    return SurfaceImage(img.mesh, new_data)


def _get_min_max(img: SurfaceImage):
    """Get min and max across hemisphere for a SurfaceImage."""
    vmin = min(min(x) for x in img.data.parts.values())
    vmax = max(max(x) for x in img.data.parts.values())
    return vmin, vmax


class SurfaceMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Extract data from a SurfaceImage.

    >>> import numpy as np
    >>> from nilearn.experimental.surface import SurfaceImage, SurfaceMasker
    >>> from nilearn.experimental.surface._datasets import toy_mesh
    >>> (mesh := toy_mesh())
    <PolyMesh with 6 vertices>
    >>> data = {"left": np.ones((3, 2)), "right": np.ones((3, 2)) * 2}
    >>> img = SurfaceImage(mesh=mesh, data=data)
    >>> img
    <SurfaceImage (6, 2)>
    >>> masker = SurfaceMasker().fit(img)
    >>> masker.slices
    {'left': (0, np.int64(3)), 'right': (np.int64(3), np.int64(6))}
    >>> (masked_data := masker.transform(img))
    array([[1., 1., 1., 2., 2., 2.],
           [1., 1., 1., 2., 2., 2.]])
    >>> (unmasked := masker.inverse_transform(masked_data))
    <SurfaceImage (6, 2)>
    >>> unmasked.data
    <PolyData (6, 2)>
    >>> unmasked.data.parts["left"]
    array([[1., 1.],
           [1., 1.],
           [1., 1.]])
    >>> unmasked.data.parts["right"]
    array([[2., 2.],
           [2., 2.],
           [2., 2.]])
    """

    mask_img: SurfaceImage | None

    mask_img_: SurfaceImage | None
    output_dimension_: int | None

    def __init__(
        self,
        mask_img=None,
        standardize=False,
        standardize_confounds=True,
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory_level=1,
        memory=None,
        reports=True,
        **kwargs,
    ):
        # TODO this does not respect the scikit-learn requirement that __init__
        # only stores the arguments without doing anything else, so set_params
        # will not work with this estimator. Also make kwargs explicit.
        if memory is None:
            memory = Memory(location=None)
        self.mask_img = mask_img
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.memory = memory
        self.memory_level = memory_level
        self._shelving = False
        self.clean_kwargs = {
            k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
        }

        self.reports = reports
        self.cmap = kwargs.get("cmap", "inferno")
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

    def _fit_mask_img(self, img: SurfaceImage | None) -> None:
        if self.mask_img is not None:
            if img is not None:
                check_same_n_vertices(self.mask_img.mesh, img.mesh)
            self.mask_img_ = self.mask_img
            return
        if img is None:
            raise ValueError(
                "Please provide either a mask_img when initializing "
                "the masker or an img when calling fit()."
            )
        # TODO: don't store a full array of 1 to mean "no masking"; use some
        # sentinel value
        mask_data = {
            k: np.ones(v.n_vertices, dtype=bool)
            for (k, v) in img.mesh.parts.items()
        }
        self.mask_img_ = SurfaceImage(mesh=img.mesh, data=mask_data)

    def fit(
        self, img: SurfaceImage | None = None, y: Any = None
    ) -> SurfaceMasker:
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
        SurfaceMasker object
        """
        del y
        self._fit_mask_img(img)
        assert self.mask_img_ is not None
        start, stop = 0, 0
        self.slices = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            assert isinstance(mask, np.ndarray)
            stop = start + mask.sum()
            self.slices[part_name] = start, stop
            start = stop
        self.output_dimension_ = stop

        for part in self.mask_img_.data.parts.keys():
            self._report_content["n_vertices"][part] = (
                self.mask_img_.mesh.parts[part].n_vertices
            )
        self._reporting_data = {
            "mask": self.mask_img_,
            "images": img,
        }

        return self

    def _check_fitted(self):
        if not hasattr(self, "mask_img_"):
            raise ValueError(
                "This masker has not been fitted. Call fit "
                "before calling transform."
            )

    def transform(
        self,
        img: SurfaceImage,
        confounds: pd.DataFrame | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
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
        parameters = get_params(
            self.__class__,
            self,
            ignore=[
                "mask_img",
            ],
        )
        parameters["clean_kwargs"] = self.clean_kwargs

        self._check_fitted()
        assert self.mask_img_ is not None
        assert self.output_dimension_ is not None
        check_same_n_vertices(self.mask_img_.mesh, img.mesh)
        output = np.empty((*img.shape[1:], self.output_dimension_))
        for part_name, (start, stop) in self.slices.items():
            mask = self.mask_img_.data.parts[part_name]
            assert isinstance(mask, np.ndarray)
            output[..., start:stop] = np.moveaxis(
                img.data.parts[part_name][mask], 0, -1
            )

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
            **parameters["clean_kwargs"],
        )

        return output

    def fit_transform(
        self,
        img: SurfaceImage,
        y: Any = None,
        confounds: pd.DataFrame | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
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
        return self.fit(img).transform(img, confounds, sample_mask)

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
        self._check_fitted()
        assert self.mask_img_ is not None
        if masked_img.shape[-1] != self.output_dimension_:
            raise ValueError(
                "Input to inverse_transform has wrong shape; "
                f"last dimension should be {self.output_dimension_}"
            )
        data = {}
        for part_name, mask in self.mask_img_.data.parts.items():
            assert isinstance(mask, np.ndarray)
            data[part_name] = np.zeros(
                (mask.shape[0], *masked_img.shape[:-1]),
                dtype=masked_img.dtype,
            )
            start, stop = self.slices[part_name]
            data[part_name][mask] = np.moveaxis(
                masked_img[..., start:stop], -1, 0
            )
        return SurfaceImage(mesh=self.mask_img_.mesh, data=data)

    def generate_report(self):
        """Generate a report."""
        try:
            from nilearn.reporting.html_report import generate_report
        except ImportError:
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    "Matplotlib is not imported! "
                    "No reports will be generated."
                )
                warnings.filterwarnings("always", message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

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

        if not fig:
            return [None]

        plt.close()

        init_display = figure_to_png_base64(fig)

        return [init_display]

    def _create_figure_for_report(self):
        import matplotlib.pyplot as plt

        from nilearn.experimental import plotting

        if not self._reporting_data["images"] and not getattr(
            self, "mask_img_", None
        ):
            return None

        if self._reporting_data["images"]:
            background_data = self._reporting_data["images"]
        else:
            background_data = self.mask_img_

        background_data = _compute_mean_image(background_data)
        vmin, vmax = _get_min_max(background_data)

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
                plotting.plot_surf(
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

                plotting.plot_surf_contours(
                    self.mask_img_,
                    hemi=hemi,
                    view=view,
                    figure=fig,
                    axes=ax,
                    colors=colors,
                )

        return fig


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
        for part in self.labels_img.data.parts.keys():
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
                relative_size.append(f"{tmp :.2}")

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
        try:
            from nilearn.reporting.html_report import generate_report
        except ImportError:
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    "Matplotlib is not imported! "
                    "No reports will be generated."
                )
                warnings.filterwarnings("always", message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

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

        from nilearn.experimental import plotting

        labels_img = self._reporting_data["labels_image"]

        img = self._reporting_data["images"]
        if img:
            img = _compute_mean_image(img)
            vmin, vmax = _get_min_max(img)

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
