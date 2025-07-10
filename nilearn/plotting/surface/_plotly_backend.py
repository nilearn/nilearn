"""Functions specific to "plotly" backend for surface visualization
functions in :obj:`~nilearn.plotting.surface.surf_plotting`.

Any imports from "plotly" package, or "plotly" engine specific utility
functions in :obj:`~nilearn.plotting.surface` should be in this file.
"""

import math

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.helpers import is_kaleido_installed
from nilearn.plotting._engine_utils import colorscale
from nilearn.plotting._utils import get_colorbar_and_data_ranges
from nilearn.plotting.displays import PlotlySurfaceFigure
from nilearn.plotting.surface._utils import (
    DEFAULT_ENGINE,
    DEFAULT_HEMI,
    VALID_HEMISPHERES,
    check_engine_params,
    check_surf_map,
    get_surface_backend,
    sanitize_hemi_view,
)
from nilearn.surface import load_surf_data, load_surf_mesh

try:
    import plotly.graph_objects as go
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


def _adjust_colorbar_and_data_ranges(
    stat_map, vmin=None, vmax=None, symmetric_cbar=None
):
    """Adjust colorbar and data ranges for 'plotly' engine.

    .. note::
        colorbar ranges are not used for 'plotly' engine.

    Parameters
    ----------
    stat_map : :obj:`str` or :class:`numpy.ndarray` or None, default=None

    %(vmin)s

    %(vmax)s

    %(symmetric_cbar)s

    Returns
    -------
        cbar_vmin, cbar_vmax, vmin, vmax
    """
    _, _, vmin, vmax = get_colorbar_and_data_ranges(
        stat_map,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )

    return None, None, vmin, vmax


def _adjust_plot_roi_params(params):
    """Adjust cbar_tick_format value for 'plotly' engine.

    Sets the values in params dict.

    Parameters
    ----------
    params : dict
        dictionary to set the adjusted parameters
    """
    cbar_tick_format = params.get("cbar_tick_format", "auto")
    if cbar_tick_format == "auto":
        params["cbar_tick_format"] = "."


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


def _get_cbar(
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


def _get_view_plot_surf(hemi, view):
    """Check ``hemi`` and ``view``, and return camera view for plotly
    engine.
    """
    view = sanitize_hemi_view(hemi, view)
    if isinstance(view, str):
        return _get_camera_view_from_string_view(hemi, view)
    return _get_camera_view_from_elevation_and_azimut(view)


def _plot_surf(
    surf_mesh,
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
    """Implement 'plotly' backend code for
    `~nilearn.plotting.surface.surf_plotting.plot_surf` function.
    """
    parameters_not_implemented_in_plotly = {
        "avg_method": avg_method,
        "alpha": alpha,
        "cbar_vmin": cbar_vmin,
        "cbar_vmax": cbar_vmax,
        "axes": axes,
        "figure": figure,
    }
    check_engine_params(parameters_not_implemented_in_plotly, "plotly")

    # adjust values
    cbar_tick_format = (
        ".1f" if cbar_tick_format == "auto" else cbar_tick_format
    )
    cmap = DEFAULT_DIVERGING_CMAP if cmap is None else cmap
    symmetric_cmap = False if symmetric_cmap is None else symmetric_cmap
    title_font_size = 18 if title_font_size is None else title_font_size

    coords, faces = load_surf_mesh(surf_mesh)

    x, y, z = coords.T
    i, j, k = faces.T

    bg_data = None
    if bg_map is not None:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError(
                "The bg_map does not have the same number "
                "of vertices as the mesh."
            )

    backend = get_surface_backend(DEFAULT_ENGINE)
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
        vertexcolor = backend._get_vertexcolor(
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
        vertexcolor = backend._get_vertexcolor(
            bg_data,
            colors["cmap"],
            colors["norm"],
            absolute_threshold=colors["abs_threshold"],
        )

    mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertexcolor)
    fig_data = [mesh_3d]
    if colorbar:
        dummy = _get_cbar(
            colors["colors"],
            float(colors["vmin"]),
            float(colors["vmax"]),
            cbar_tick_format,
        )
        fig_data.append(dummy)

    # instantiate plotly figure
    camera_view = _get_view_plot_surf(hemi, view)
    fig = go.Figure(data=fig_data)
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
