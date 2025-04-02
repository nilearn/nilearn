import math

from nilearn.plotting.surface._backend import (
    VALID_HEMISPHERES,
    _check_hemispheres,
    _check_views,
)

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


def _configure_title_plotly(title, font_size, color="black"):
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
    _check_views([view])
    _check_hemispheres([hemi])
    if isinstance(view, str):
        return _get_camera_view_from_string_view(hemi, view)
    return _get_camera_view_from_elevation_and_azimut(view)
