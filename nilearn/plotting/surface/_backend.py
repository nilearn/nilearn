from warnings import warn

import numpy as np
import pandas as pd

from nilearn import DEFAULT_DIVERGING_CMAP, image
from nilearn._utils import check_niimg_3d
from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn._utils.param_validation import check_params
from nilearn.plotting._utils import (
    create_colormap_from_lut,
    get_colorbar_and_data_ranges,
)
from nilearn.plotting.surface._utils import (
    DATA_EXTENSIONS,
    check_hemispheres,
    check_surface_plotting_inputs,
    check_views,
    sanitize_hemi_for_surface_image,
)
from nilearn.surface import (
    load_surf_data,
    load_surf_mesh,
    vol_to_surf,
)
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
    check_mesh_is_fsaverage,
)


def get_surface_backend(engine=None):
    if engine == "matplotlib":
        if is_matplotlib_installed():
            from nilearn.plotting.surface._matplotlib_backend import (
                MatplotlibBackend,
            )

            return MatplotlibBackend()
        else:
            raise ImportError(
                "Using engine='matplotlib' requires that ``matplotlib`` is "
                "installed."
            )
    elif engine == "plotly":
        if is_plotly_installed():
            from nilearn.plotting.surface._plotly_backend import PlotlyBackend

            return PlotlyBackend()
        else:
            raise ImportError(
                "Using engine='plotly' requires that ``plotly`` is installed."
            )
    else:
        raise ValueError(
            f"Unknown plotting engine {engine}. "
            "Please use either 'matplotlib' or "
            "'plotly'."
        )


class SurfaceBackend:
    def plot_surf(
        self,
        surf_mesh=None,
        surf_map=None,
        bg_map=None,
        hemi=None,
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
        if view is None:
            view = "dorsal" if hemi == "both" else "lateral"

        surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            surf_map, surf_mesh, hemi, bg_map
        )

        check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        coords, faces = load_surf_mesh(surf_mesh)

        fig = self._plot_surf(
            coords,
            faces,
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

        return fig

    def plot_surf_contours(
        self,
        surf_mesh=None,
        roi_map=None,
        hemi=None,
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
        # TODO hemi returns None from here, if I pass to plot_surf,
        # returns error
        hemi = sanitize_hemi_for_surface_image(hemi, roi_map, surf_mesh)
        roi_map, surf_mesh, _ = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, map_var_name="roi_map"
        )
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        fig = self._plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
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
        return fig

    def plot_surf_stat_map(
        self,
        surf_mesh=None,
        stat_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        threshold=None,
        alpha=None,
        vmin=None,
        vmax=None,
        cmap=DEFAULT_DIVERGING_CMAP,
        colorbar=True,
        symmetric_cbar="auto",
        cbar_tick_format="auto",
        bg_on_data=False,
        darkness=0.7,
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
        avg_method=None,
        **kwargs,
    ):
        check_params(locals())

        stat_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            stat_map, surf_mesh, hemi, bg_map, map_var_name="stat_map"
        )

        check_extensions(stat_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
        loaded_stat_map = load_surf_data(stat_map)

        # Call get_colorbar_and_data_ranges to derive symmetric vmin, vmax
        # And colorbar limits depending on symmetric_cbar settings
        cbar_vmin, cbar_vmax, vmin, vmax = get_colorbar_and_data_ranges(
            loaded_stat_map,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )

        fig = self._plot_surf_stat_map(
            surf_mesh,
            surf_map=loaded_stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            threshold=threshold,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            bg_on_data=bg_on_data,
            darkness=darkness,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            avg_method=avg_method,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            **kwargs,
        )
        return fig

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
        symmetric_cbar="auto",
        cmap=DEFAULT_DIVERGING_CMAP,
        cbar_tick_format="%i",
        **kwargs,
    ):
        check_params(locals())
        if hemispheres in (None, "both"):
            hemispheres = ["left", "right"]
        if views is None:
            views = ["lateral", "medial"]

        stat_map = check_niimg_3d(stat_map, dtype="auto")
        modes = check_views(views)
        hemis = check_hemispheres(hemispheres)
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
        _, _, vmin, vmax = get_colorbar_and_data_ranges(
            image.get_data(stat_map),
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )

        fig = self._plot_img_on_surf(
            stat_map=stat_map,
            surf_mesh=surf_mesh,
            hemispheres=hemispheres,
            modes=modes,
            hemis=hemis,
            surf=surf,
            texture=texture,
            bg_on_data=bg_on_data,
            inflate=inflate,
            threshold=threshold,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            symmetric_cbar=symmetric_cbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def plot_surf_roi(
        self,
        surf_mesh=None,
        roi_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        avg_method=None,
        threshold=1e-14,
        alpha=None,
        vmin=None,
        vmax=None,
        cmap="gist_ncar",
        cbar_tick_format="auto",
        bg_on_data=False,
        darkness=0.7,
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
        colorbar=True,
        **kwargs,
    ):
        # set default view to dorsal if hemi is both and view is not set
        check_params(locals())
        if view is None:
            view = "dorsal" if hemi == "both" else "lateral"

        roi_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, bg_map
        )
        # preload roi and mesh to determine vmin, vmax and give more useful
        # error messages in case of wrong inputs
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        roi = load_surf_data(roi_map)

        idx_not_na = ~np.isnan(roi)
        if vmin is None:
            vmin = float(np.nanmin(roi))
        if vmax is None:
            vmax = float(1 + np.nanmax(roi))

        mesh = load_surf_mesh(surf_mesh)

        if roi.ndim != 1:
            raise ValueError(
                "roi_map can only have one dimension but has "
                f"{roi.ndim} dimensions"
            )
        if roi.shape[0] != mesh.n_vertices:
            raise ValueError(
                "roi_map does not have the same number of vertices "
                "as the mesh. If you have a list of indices for the "
                "ROI you can convert them into a ROI map like this:\n"
                "roi_map = np.zeros(n_vertices)\n"
                "roi_map[roi_idx] = 1"
            )
        if (roi < 0).any():
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Negative values in roi_map will no longer be allowed in"
                    " Nilearn version 0.13"
                ),
                DeprecationWarning,
            )
        if not np.array_equal(roi[idx_not_na], roi[idx_not_na].astype(int)):
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Non-integer values in roi_map will no longer be allowed "
                    "in Nilearn version 0.13"
                ),
                DeprecationWarning,
            )

        if isinstance(cmap, pd.DataFrame):
            cmap = create_colormap_from_lut(cmap)

        fig = self._plot_surf_roi(
            mesh,
            roi_map=roi,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            avg_method=avg_method,
            threshold=threshold,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_tick_format=cbar_tick_format,
            bg_on_data=bg_on_data,
            darkness=darkness,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            colorbar=colorbar,
            **kwargs,
        )

        return fig

    def _check_backend_params(self, params_not_implemented):
        for parameter, value in params_not_implemented.items():
            if value is not None:
                warn(
                    f"'{parameter}' is not implemented "
                    f"for the {self.name} engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning."
                )
