import math

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.helpers import is_kaleido_installed
from nilearn.plotting.displays import PlotlySurfaceFigure
from nilearn.plotting.html_surface import get_vertexcolor
from nilearn.plotting.js_plotting_utils import colorscale
from nilearn.plotting.surface._backend import SurfaceBackend
from nilearn.plotting.surface._utils import (
    VALID_HEMISPHERES,
    check_hemispheres,
    check_surf_map,
    check_views,
)
from nilearn.surface import load_surf_data

try:
    from plotly.graph_objects import Figure, Mesh3d
except ImportError:
    from nilearn.plotting._utils import engine_warning

    engine_warning("plotly")


CAMERAS = {
    "left": {
        "eye": {"x": -1.5, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "right": {
        "eye": {"x": 1.5, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "dorsal": {
        "eye": {"x": 0, "y": 0, "z": 1.5},
        "up": {"x": -1, "y": 0, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "ventral": {
        "eye": {"x": 0, "y": 0, "z": -1.5},
        "up": {"x": 1, "y": 0, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "anterior": {
        "eye": {"x": 0, "y": 1.5, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "posterior": {
        "eye": {"x": 0, "y": -1.5, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
}


AXIS_CONFIG = {
    "showgrid": False,
    "showline": False,
    "ticks": "",
    "title": "",
    "showticklabels": False,
    "zeroline": False,
    "showspikes": False,
    "spikesides": False,
    "showbackground": False,
}


LAYOUT = {
    "scene": {
        "dragmode": "orbit",
        **{f"{dim}axis": AXIS_CONFIG for dim in ("x", "y", "z")},
    },
    "paper_bgcolor": "#fff",
    "hovermode": False,
    "margin": {"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
}


def _configure_title(title, font_size, color="black"):
    """Help for plot_surf with plotly engine.

    This function configures the title if provided.
    """
    if title is None:
        return {}
    return {
        "text": title,
        "font": {
            "size": font_size,
            "color": color,
        },
        "y": 0.96,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }


def _get_camera_view_from_string_view(hemi, view):
    """Return plotly camera parameters from string view."""
    if hemi in ["left", "right"]:
        if view == "lateral":
            return CAMERAS[hemi]
        elif view == "medial":
            return CAMERAS[
                (
                    VALID_HEMISPHERES[0]
                    if hemi == VALID_HEMISPHERES[1]
                    else VALID_HEMISPHERES[1]
                )
            ]
    elif hemi == "both" and view in ["lateral", "medial"]:
        raise ValueError(
            "Invalid view definition: when hemi is 'both', "
            "view cannot be 'lateral' or 'medial'.\n"
            "Maybe you meant 'left' or 'right'?"
        )
    return CAMERAS[view]


def _get_camera_view_from_elevation_and_azimut(view):
    """Compute plotly camera parameters from elevation and azimut."""
    elev, azim = view
    # The radius is useful only when using a "perspective" projection,
    # otherwise, if projection is "orthographic",
    # one should tweak the "aspectratio" to emulate zoom
    r = 1.5
    # The camera position and orientation is set by three 3d vectors,
    # whose coordinates are independent of the plotted data.
    return {
        # Where the camera should look at
        # (it should always be looking at the center of the scene)
        "center": {"x": 0, "y": 0, "z": 0},
        # Where the camera should be located
        "eye": {
            "x": (
                r
                * math.cos(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "y": (
                r
                * math.sin(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "z": r * math.sin(elev / 360 * 2 * math.pi),
        },
        # How the camera should be rotated.
        # It is determined by a 3d vector indicating which direction
        # should look up in the generated plot
        "up": {
            "x": math.sin(elev / 360 * 2 * math.pi)
            * math.cos(azim / 360 * 2 * math.pi + math.pi),
            "y": math.sin(elev / 360 * 2 * math.pi)
            * math.sin(azim / 360 * 2 * math.pi + math.pi),
            "z": math.cos(elev / 360 * 2 * math.pi),
        },
        # "projection": {"type": "perspective"},
        "projection": {"type": "orthographic"},
    }


def _get_cbar_plotly(
    colorscale,
    vmin,
    vmax,
    cbar_tick_format,
    fontsize=25,
    color="black",
    height=0.5,
):
    """Help for _plot_surf_plotly.

    This function configures the colorbar and creates a small
    invisible plot that uses the appropriate cmap to trigger
    the generation of the colorbar. This dummy plot has then to
    be added to the figure.
    """
    dummy = {
        "opacity": 0,
        "colorbar": {
            "tickfont": {"size": fontsize, "color": color},
            "tickformat": cbar_tick_format,
            "len": height,
        },
        "type": "mesh3d",
        "colorscale": colorscale,
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1],
        "i": [0],
        "j": [1],
        "k": [2],
        "intensity": [0.0],
        "cmin": vmin,
        "cmax": vmax,
    }
    return dummy


def _get_view_plot_surf_plotly(hemi, view):
    """
    Get camera parameters from hemi and view for the plotly engine.

    This function checks the selected hemisphere and view, and
    returns the cameras view.
    """
    check_views([view])
    check_hemispheres([hemi])
    if isinstance(view, str):
        return _get_camera_view_from_string_view(hemi, view)
    return _get_camera_view_from_elevation_and_azimut(view)


class PlotlyBackend(SurfaceBackend):
    @property
    def name(self):
        return "plotly"

    def _plot_surf(
        self,
        coords,
        faces,
        surf_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        cmap=None,
        symmetric_cmap=False,
        colorbar=False,
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
        title_font_size=18,
        output_file=None,
        axes=None,
        figure=None,
    ):
        parameters_not_implemented = {
            "avg_method": avg_method,
            "alpha": alpha,
            "cbar_vmin": cbar_vmin,
            "cbar_vmax": cbar_vmax,
            "figure": figure,
            "axes": axes,
        }

        self._check_backend_params(parameters_not_implemented)

        if title_font_size is None:
            title_font_size = 18
        if symmetric_cmap is None:
            symmetric_cmap = False

        if cbar_tick_format == "auto":
            cbar_tick_format = ".1f"

        x, y, z = coords.T
        i, j, k = faces.T

        if cmap is None:
            cmap = DEFAULT_DIVERGING_CMAP

        bg_data = None
        if bg_map is not None:
            bg_data = load_surf_data(bg_map)
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError(
                    "The bg_map does not have the same number "
                    "of vertices as the mesh."
                )

        if surf_map is not None:
            check_surf_map(surf_map, coords.shape[0])
            colors = colorscale(
                cmap,
                surf_map,
                threshold,
                vmax=vmax,
                vmin=vmin,
                symmetric_cmap=symmetric_cmap,
            )
            vertexcolor = get_vertexcolor(
                surf_map,
                colors["cmap"],
                colors["norm"],
                absolute_threshold=colors["abs_threshold"],
                bg_map=bg_data,
                bg_on_data=bg_on_data,
                darkness=darkness,
            )
        else:
            if bg_data is None:
                bg_data = np.zeros(coords.shape[0])
            colors = colorscale("Greys", bg_data, symmetric_cmap=False)
            vertexcolor = get_vertexcolor(
                bg_data,
                colors["cmap"],
                colors["norm"],
                absolute_threshold=colors["abs_threshold"],
            )

        mesh_3d = Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertexcolor)
        fig_data = [mesh_3d]
        if colorbar:
            dummy = _get_cbar_plotly(
                colors["colors"],
                float(colors["vmin"]),
                float(colors["vmax"]),
                cbar_tick_format,
            )
            fig_data.append(dummy)

        # instantiate plotly figure
        camera_view = _get_view_plot_surf_plotly(hemi, view)
        fig = Figure(data=fig_data)
        fig.update_layout(
            scene_camera=camera_view,
            title=_configure_title(title, title_font_size),
            **LAYOUT,
        )

        # save figure
        plotly_figure = PlotlySurfaceFigure(
            figure=fig, output_file=output_file, hemi=hemi
        )

        if output_file is not None:
            if not is_kaleido_installed():
                msg = (
                    "Saving figures to file with engine='plotly' requires "
                    "that ``kaleido`` is installed."
                )
                raise ImportError(msg)
            plotly_figure.savefig()

        return plotly_figure

    def _plot_surf_contours(
        self,
        surf_mesh=None,
        roi_map=None,
        hemi="left",
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
        raise NotImplementedError(
            "'plot_surf_contours' is not implemented for plotly!"
        )

    def _plot_surf_stat_map(
        self,
        surf_mesh=None,
        surf_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        threshold=None,
        alpha=None,
        vmin=None,
        vmax=None,
        cmap=DEFAULT_DIVERGING_CMAP,
        colorbar=True,
        cbar_tick_format="auto",
        bg_on_data=False,
        darkness=0.7,
        title=None,
        title_font_size=18,
        output_file=None,
        axes=None,
        figure=None,
        avg_method=None,
        cbar_vmin=None,
        cbar_vmax=None,
        **kwargs,
    ):
        # Set to None the values that are not used by plotly
        # to avoid warnings thrown by plot_surf
        cbar_vmin = None
        cbar_vmax = None

        fig = self.plot_surf(
            surf_mesh,
            surf_map=surf_map,
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

        return fig

    def _plot_img_on_surf(
        self,
        stat_map,
        surf_mesh,
        hemispheres,
        modes,
        hemis,
        surf,
        texture,
        bg_on_data=False,
        inflate=False,
        threshold=None,
        colorbar=True,
        cbar_tick_format="%i",
        symmetric_cbar="auto",
        cmap=DEFAULT_DIVERGING_CMAP,
        vmin=None,
        vmax=None,
        title=None,
        output_file=None,
        **kwargs,
    ):
        raise NotImplementedError(
            "'plot_img_on_surf' is not implemented for plotly!"
        )

    def _plot_surf_roi(
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
        title_font_size=18,
        output_file=None,
        axes=None,
        figure=None,
        colorbar=True,
        **kwargs,
    ):
        if cbar_tick_format == "auto":
            cbar_tick_format = "."
        if title_font_size is None:
            title_font_size = 18

        fig = self.plot_surf(
            surf_mesh=surf_mesh,
            surf_map=roi_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            avg_method=avg_method,
            threshold=threshold,
            cmap=cmap,
            cbar_tick_format=cbar_tick_format,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            colorbar=colorbar,
            **kwargs,
        )

        return fig
