"""Functions that are common to all possible backend implementations for
surface visualization functions in
:obj:`~nilearn.plotting.surface.surf_plotting`.

"Matplotlib" is default engine for surface visualization functions. For some
functions, there is also a "plotly" implementation.

All dependencies and functions related to "matplotlib" implementation is in
:obj:`~nilearn.plotting.surface._matplotlib_backend` module.

All dependencies and functions related to "plotly" implementation is in
:obj:`~nilearn.plotting.surface._plotly_backend` module.

Each backend engine implementation should be a self contained module. Any
imports on the engine package, or engine specific utility functions should not
appear elsewhere.
"""

from collections.abc import Sequence
from warnings import warn

import numpy as np
import pandas as pd

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn._utils.param_validation import check_params
from nilearn.image import get_data
from nilearn.plotting._utils import create_colormap_from_lut
from nilearn.plotting.surface._utils import (
    DEFAULT_HEMI,
    check_surface_plotting_inputs,
)
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
    check_mesh_is_fsaverage,
)

# subset of data format extensions supported
DATA_EXTENSIONS = (
    "gii",
    "gii.gz",
    "mgz",
)

VALID_VIEWS = (
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "dorsal",
    "ventral",
    "left",
    "right",
)

VALID_HEMISPHERES = "left", "right", "both"


class BaseSurfaceBackend:
    """A base class that behaves as an interface for Surface plotting
    backend.

    The methods of class should be implemented by each engine used as backend.
    """

    def _check_engine_params(self, params):
        """Check default values of the parameters that are not implemented for
        current engine and warn the user if the parameter has other value then
        None.

        Parameters
        ----------
        params: :obj:`dict`
            A dictionary where keys are the unimplemented parameter names for a
        specific engine and values are the assigned value for corresponding
        parameter.
        """
        for parameter, value in params.items():
            if value is not None:
                warn(
                    f"'{parameter}' is not implemented "
                    f"for the {self.name} engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning.",
                    stacklevel=find_stack_level(),
                )

    def _sanitize_hemi_view(self, hemi, view):
        """Check ``hemi`` and ``view``, if ``view`` is `None`, set value for
        ``view`` depending on the ``hemi`` value and return ``view``.
        """
        check_hemispheres([hemi])
        if view is None:
            view = "dorsal" if hemi == "both" else "lateral"
        check_views([view])
        return view

    def load_surf_mesh(self, surf_mesh):
        return load_surf_mesh(surf_mesh)

    def load_surf_data(self, surf_map):
        return load_surf_data(surf_map)

    def get_image_data(self, img):
        return get_data(img)

    def plot_surf(
        self,
        surf_mesh=None,
        surf_map=None,
        bg_map=None,
        hemi=DEFAULT_HEMI,
        view=None,
        cmap=None,
        symmetric_cmap=None,
        colorbar=True,
        avg_method=None,
        threshold=None,
        alpha=None,
        bg_on_data=False,
        darkness=0.7,
        vmin=None,
        vmax=None,
        cbar_vmin=None,
        cbar_vmax=None,
        cbar_tick_format="auto",
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
    ):
        check_params(locals())
        surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            surf_map, surf_mesh, hemi, bg_map
        )
        check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        return self._plot_surf(
            surf_mesh,
            surf_map=surf_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            avg_method=avg_method,
            threshold=threshold,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            cbar_tick_format=cbar_tick_format,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
        )

    def plot_surf_contours(
        self,
        surf_mesh=None,
        roi_map=None,
        hemi=DEFAULT_HEMI,
        levels=None,
        labels=None,
        colors=None,
        legend=False,
        cmap="tab20",
        title=None,
        output_file=None,
        axes=None,
        figure=None,
        **kwargs,
    ):
        roi_map, surf_mesh, _ = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, map_var_name="roi_map"
        )
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        return self._plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            levels=levels,
            labels=labels,
            colors=colors,
            legend=legend,
            cmap=cmap,
            title=title,
            output_file=output_file,
            axes=axes,
            figure=figure,
            **kwargs,
        )

    def plot_surf_stat_map(
        self,
        surf_mesh=None,
        stat_map=None,
        bg_map=None,
        hemi=DEFAULT_HEMI,
        view=None,
        cmap=DEFAULT_DIVERGING_CMAP,
        colorbar=True,
        avg_method=None,
        threshold=None,
        alpha=None,
        bg_on_data=False,
        darkness=0.7,
        vmin=None,
        vmax=None,
        symmetric_cbar="auto",
        cbar_tick_format="auto",
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
        **kwargs,
    ):
        check_params(locals())

        stat_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            stat_map, surf_mesh, hemi, bg_map, map_var_name="stat_map"
        )
        check_extensions(stat_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
        loaded_stat_map = load_surf_data(stat_map)

        # derive symmetric vmin, vmax and colorbar limits depending on
        # symmetric_cbar settings
        cbar_vmin, cbar_vmax, vmin, vmax = (
            self._adjust_colorbar_and_data_ranges(
                stat_map,
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=symmetric_cbar,
            )
        )

        return self._plot_surf(
            surf_mesh,
            surf_map=stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            avg_method=avg_method,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=True,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmax=vmax,
            vmin=vmin,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            **kwargs,
        )

    def plot_img_on_surf(
        self,
        stat_map,
        surf_mesh="fsaverage5",
        mask_img=None,
        hemispheres=None,
        bg_on_data=False,
        inflate=False,
        views=None,
        output_file=None,
        title=None,
        colorbar=True,
        vmin=None,
        vmax=None,
        threshold=None,
        symmetric_cbar=None,
        cmap=DEFAULT_DIVERGING_CMAP,
        cbar_tick_format=None,
        **kwargs,
    ):
        check_params(locals())

        for arg in ("figure", "axes", "engine"):
            if arg in kwargs:
                raise ValueError(
                    f"plot_img_on_surf does not accept {arg} as an argument"
                )

        if hemispheres in (None, "both", ["both"]):
            hemispheres = ["left", "right"]
        if not isinstance(hemispheres, list):
            hemispheres = [hemispheres]
        hemis = check_hemispheres(hemispheres)

        if views is None:
            views = ["lateral", "medial"]
        modes = check_views(views)

        stat_map = check_niimg_3d(stat_map, dtype="auto")
        surf_mesh = check_mesh_is_fsaverage(surf_mesh)

        mesh_prefix = "infl" if inflate else "pial"
        surf = {
            "left": surf_mesh[f"{mesh_prefix}_left"],
            "right": surf_mesh[f"{mesh_prefix}_right"],
        }
        texture = {
            "left": vol_to_surf(
                stat_map, surf_mesh["pial_left"], mask_img=mask_img
            ),
            "right": vol_to_surf(
                stat_map, surf_mesh["pial_right"], mask_img=mask_img
            ),
        }

        # get vmin and vmax for entire data (all hemis)
        _, _, vmin, vmax = self._adjust_colorbar_and_data_ranges(
            self.get_image_data(stat_map),
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )

        return self._plot_img_on_surf(
            surf,
            surf_mesh=surf_mesh,
            stat_map=stat_map,
            texture=texture,
            hemis=hemis,
            modes=modes,
            bg_on_data=bg_on_data,
            inflate=inflate,
            output_file=output_file,
            title=title,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            symmetric_cbar=symmetric_cbar,
            cmap=cmap,
            cbar_tick_format=cbar_tick_format,
            **kwargs,
        )

    def plot_surf_roi(
        self,
        surf_mesh=None,
        roi_map=None,
        bg_map=None,
        hemi=DEFAULT_HEMI,
        view=None,
        cmap="gist_ncar",
        colorbar=True,
        avg_method=None,
        threshold=1e-14,
        alpha=None,
        bg_on_data=False,
        darkness=0.7,
        vmin=None,
        vmax=None,
        cbar_tick_format="auto",
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
        **kwargs,
    ):
        check_params(locals())
        roi_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, bg_map
        )
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        # preload roi and mesh to determine vmin, vmax and give more useful
        # error messages in case of wrong inputs
        roi = load_surf_data(roi_map)

        if roi.ndim != 1:
            raise ValueError(
                "roi_map can only have one dimension but has "
                f"{roi.ndim} dimensions"
            )
        if (roi < 0).any():
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Negative values in roi_map will no longer be allowed in"
                    " Nilearn version 0.13"
                ),
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )

        mesh = load_surf_mesh(surf_mesh)
        if roi.shape[0] != mesh.n_vertices:
            raise ValueError(
                "roi_map does not have the same number of vertices "
                "as the mesh. If you have a list of indices for the "
                "ROI you can convert them into a ROI map like this:\n"
                "roi_map = np.zeros(n_vertices)\n"
                "roi_map[roi_idx] = 1"
            )

        idx_not_na = ~np.isnan(roi)
        if vmin is None:
            vmin = float(np.nanmin(roi))
        if vmax is None:
            vmax = float(1 + np.nanmax(roi))

        if not np.array_equal(roi[idx_not_na], roi[idx_not_na].astype(int)):
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Non-integer values in roi_map will no longer be allowed "
                    "in Nilearn version 0.13"
                ),
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
        if isinstance(cmap, pd.DataFrame):
            cmap = create_colormap_from_lut(cmap)

        params = {
            "avg_method": avg_method,
            "cbar_tick_format": cbar_tick_format,
        }

        self._adjust_plot_roi_params(params)

        return self._plot_surf(
            mesh,
            surf_map=roi,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
            colorbar=colorbar,
            avg_method=params["avg_method"],
            threshold=threshold,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            cbar_tick_format=params["cbar_tick_format"],
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            **kwargs,
        )


def _check_hemisphere_is_valid(hemi):
    return hemi in VALID_HEMISPHERES


def check_hemispheres(hemispheres):
    """Check whether the hemispheres passed to in plot_img_on_surf are \
    correct.

    hemispheres : :obj:`list`
        Any combination of 'left' and 'right'.

    """
    invalid_hemis = [
        not _check_hemisphere_is_valid(hemi) for hemi in hemispheres
    ]
    if any(invalid_hemis):
        raise ValueError(
            "Invalid hemispheres definition!\n"
            f"Got: {np.array(hemispheres)[invalid_hemis]!s}\n"
            f"Supported values are: {VALID_HEMISPHERES!s}"
        )
    return hemispheres


def _check_view_is_valid(view) -> bool:
    """Check whether a single view is one of two valid input types.

    Parameters
    ----------
    view : :obj:`str` in {"anterior", "posterior", "medial", "lateral",
        "dorsal", "ventral" or pair of floats (elev, azim).

    Returns
    -------
    valid : True if view is valid, False otherwise.
    """
    if isinstance(view, str) and (view in VALID_VIEWS):
        return True
    return (
        isinstance(view, Sequence)
        and len(view) == 2
        and all(isinstance(x, (int, float)) for x in view)
    )


def check_views(views) -> list:
    """Check whether the views passed to in plot_img_on_surf are correct.

    Parameters
    ----------
    views : :obj:`list`
        Any combination of strings in {"anterior", "posterior", "medial",
        "lateral", "dorsal", "ventral"} and / or pair of floats (elev, azim).

    Returns
    -------
    views : :obj:`list`
        Views given as inputs.
    """
    invalid_views = [not _check_view_is_valid(view) for view in views]

    if any(invalid_views):
        raise ValueError(
            "Invalid view definition!\n"
            f"Got: {np.array(views)[invalid_views]!s}\n"
            f"Supported values are: {VALID_VIEWS!s}"
            " or a sequence of length 2"
            " setting the elevation and azimut of the camera."
        )

    return views


def check_surf_map(surf_map, n_vertices):
    """Help for plot_surf.

    This function checks the dimensions of provided surf_map.
    """
    surf_map_data = load_surf_data(surf_map)
    if surf_map_data.ndim != 1:
        raise ValueError(
            "'surf_map' can only have one dimension "
            f"but has '{surf_map_data.ndim}' dimensions"
        )
    if surf_map_data.shape[0] != n_vertices:
        raise ValueError(
            "The surf_map does not have the same number "
            "of vertices as the mesh."
        )
    return surf_map_data
