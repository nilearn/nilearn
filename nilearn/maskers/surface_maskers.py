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
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.experimental.surface._surface_image import SurfaceImage
from nilearn.maskers._utils import (
    _compute_mean_image,
    _get_min_max,
    check_same_n_vertices,
)


class SurfaceMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Extract data from a SurfaceImage."""

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

        for part in self.mask_img_.data.parts:
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
        output = np.empty((*img.shape[:-1], self.output_dimension_))
        for part_name, (start, stop) in self.slices.items():
            mask = self.mask_img_.data.parts[part_name]
            assert isinstance(mask, np.ndarray)
            output[..., start:stop] = img.data.parts[part_name][..., mask]

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
                (*masked_img.shape[:-1], mask.shape[0]),
                dtype=masked_img.dtype,
            )
            start, stop = self.slices[part_name]
            data[part_name][..., mask] = masked_img[..., start:stop]
        return SurfaceImage(mesh=self.mask_img_.mesh, data=data)

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
