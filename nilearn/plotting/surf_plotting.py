"""Functions for surface visualization."""

import itertools
import math
from collections.abc import Sequence
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nilearn import image, surface
from nilearn._utils import check_niimg_3d, compare_version, fill_doc
from nilearn._utils.helpers import is_kaleido_installed, is_plotly_installed
from nilearn.plotting._utils import check_surface_plotting_inputs
from nilearn.plotting.cm import cold_hot, mix_colormaps
from nilearn.plotting.displays._figures import PlotlySurfaceFigure
from nilearn.plotting.displays._slicers import _get_cbar_ticks
from nilearn.plotting.html_surface import get_vertexcolor
from nilearn.plotting.img_plotting import get_colorbar_and_data_ranges
from nilearn.plotting.js_plotting_utils import colorscale
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
    load_surf_data,
    load_surf_mesh,
    vol_to_surf,
)
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
    check_mesh_is_fsaverage,
)

VALID_VIEWS = "anterior", "posterior", "medial", "lateral", "dorsal", "ventral"
VALID_HEMISPHERES = "left", "right"

# subset of data format extensions supported
DATA_EXTENSIONS = (
    "gii",
    "gii.gz",
    "mgz",
)


MATPLOTLIB_VIEWS = {
    "left": {
        "lateral": (0, 180),
        "medial": (0, 0),
        "dorsal": (90, 0),
        "ventral": (270, 0),
        "anterior": (0, 90),
        "posterior": (0, 270),
    },
    "right": {
        "lateral": (0, 0),
        "medial": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
        "anterior": (0, 90),
        "posterior": (0, 270),
    },
}


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


def _get_camera_view_from_string_view(hemi, view):
    """Return plotly camera parameters from string view."""
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
    else:
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


def _plot_surf_plotly(
    coords,
    faces,
    surf_map=None,
    bg_map=None,
    hemi="left",
    view="lateral",
    cmap=None,
    symmetric_cmap=True,
    colorbar=False,
    threshold=None,
    bg_on_data=False,
    darkness=0.7,
    vmin=None,
    vmax=None,
    cbar_tick_format=".1f",
    title=None,
    title_font_size=18,
    output_file=None,
):
    """Help for plot_surf.

    .. versionadded:: 0.9.0

    This function handles surface plotting when the selected
    engine is plotly.

    .. note::
        This function assumes that plotly and kaleido are
        installed.

    .. warning::
        This function is new and experimental. Please report
        bugs that you may encounter.

    """
    if is_plotly_installed():
        import plotly.graph_objects as go
    else:
        msg = "Using engine='plotly' requires that ``plotly`` is installed."
        raise ImportError(msg)

    x, y, z = coords.T
    i, j, k = faces.T

    if cmap is None:
        cmap = cold_hot

    bg_data = None
    if bg_map is not None:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError(
                "The bg_map does not have the same number "
                "of vertices as the mesh."
            )

    if surf_map is not None:
        _check_surf_map(surf_map, coords.shape[0])
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

    mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertexcolor)
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
    fig = go.Figure(data=fig_data)
    fig.update_layout(
        scene_camera=camera_view,
        title=_configure_title_plotly(title, title_font_size),
        **LAYOUT,
    )

    # save figure
    plotly_figure = PlotlySurfaceFigure(figure=fig, output_file=output_file)

    if output_file is not None:
        if not is_kaleido_installed():
            msg = (
                "Saving figures to file with engine='plotly' requires "
                "that ``kaleido`` is installed."
            )
            raise ImportError(msg)
        plotly_figure.savefig()

    return plotly_figure


def _get_view_plot_surf_matplotlib(hemi, view):
    """Help function for plot_surf with matplotlib engine.

    This function checks the selected hemisphere and view, and
    returns elev and azim.
    """
    _check_views([view])
    _check_hemispheres([hemi])
    if isinstance(view, str):
        return MATPLOTLIB_VIEWS[hemi][view]
    return view


def _check_surf_map(surf_map, n_vertices):
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


def _compute_surf_map_faces_matplotlib(
    surf_map, faces, avg_method, n_vertices, face_colors_size
):
    """Help for plot_surf.

    This function computes the surf map faces using the
    provided averaging method.

    .. note::
        This method is called exclusively when using matplotlib,
        since it only supports plotting face-colour maps and not
        vertex-colour maps.

    """
    surf_map_data = _check_surf_map(surf_map, n_vertices)

    # create face values from vertex values by selected avg methods
    error_message = (
        "avg_method should be either "
        "['mean', 'median', 'max', 'min'] "
        "or a custom function"
    )
    if isinstance(avg_method, str):
        try:
            avg_method = getattr(np, avg_method)
        except AttributeError:
            raise ValueError(error_message)
        surf_map_faces = avg_method(surf_map_data[faces], axis=1)
    elif callable(avg_method):
        surf_map_faces = np.apply_along_axis(
            avg_method, 1, surf_map_data[faces]
        )

        # check that surf_map_faces has the same length as face_colors
        if surf_map_faces.shape != (face_colors_size,):
            raise ValueError(
                "Array computed with the custom function "
                "from avg_method does not have the correct shape: "
                f"{surf_map_faces[0]} != {face_colors_size}"
            )

        # check that dtype is either int or float
        if not (
            "int" in str(surf_map_faces.dtype)
            or "float" in str(surf_map_faces.dtype)
        ):
            raise ValueError(
                "Array computed with the custom function "
                "from avg_method should be an array of numbers "
                "(int or float)"
            )
    else:
        raise ValueError(error_message)
    return surf_map_faces


def _get_ticks_matplotlib(vmin, vmax, cbar_tick_format, threshold):
    """Help for plot_surf with matplotlib engine.

    This function computes the tick values for the colorbar.
    """
    # Default number of ticks is 5...
    n_ticks = 5
    # ...unless we are dealing with integers with a small range
    # in this case, we reduce the number of ticks
    if cbar_tick_format == "%i" and vmax - vmin < n_ticks - 1:
        return np.arange(vmin, vmax + 1)
    else:
        return _get_cbar_ticks(vmin, vmax, threshold, n_ticks)


def _get_cmap_matplotlib(cmap, vmin, vmax, cbar_tick_format, threshold=None):
    """Help for plot_surf with matplotlib engine.

    This function returns the colormap.
    """
    our_cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
    if threshold is not None:
        if cbar_tick_format == "%i" and int(threshold) != threshold:
            warn(
                "You provided a non integer threshold "
                "but configured the colorbar to use integer formatting."
            )
        # set colors to gray for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
    our_cmap = LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, our_cmap.N
    )
    return our_cmap, norm


def _compute_facecolors_matplotlib(bg_map, faces, n_vertices, darkness, alpha):
    """Help for plot_surf with matplotlib engine.

    This function computes the facecolors.
    """
    if bg_map is None:
        bg_data = np.ones(n_vertices) * 0.5
    else:
        bg_data = np.copy(load_surf_data(bg_map))
        if bg_data.shape[0] != n_vertices:
            raise ValueError(
                "The bg_map does not have the same number "
                "of vertices as the mesh."
            )

    bg_faces = np.mean(bg_data[faces], axis=1)
    # scale background map if need be
    bg_vmin, bg_vmax = np.min(bg_faces), np.max(bg_faces)
    if bg_vmin < 0 or bg_vmax > 1:
        bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
        bg_faces = bg_norm(bg_faces)

    if darkness is not None:
        bg_faces *= darkness
        warn(
            (
                "The `darkness` parameter will be deprecated in release 0.13. "
                "We recommend setting `darkness` to None"
            ),
            DeprecationWarning,
        )

    face_colors = plt.cm.gray_r(bg_faces)

    # set alpha if in auto mode
    if alpha == "auto":
        alpha = 0.5 if bg_map is None else 1
    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]

    return face_colors


def _threshold_and_rescale(data, threshold, vmin, vmax):
    """Help for plot_surf.

    This function thresholds and rescales the provided data.
    """
    data_copy, vmin, vmax = _rescale(data, vmin, vmax)
    return data_copy, _threshold(data, threshold, vmin, vmax), vmin, vmax


def _threshold(data, threshold, vmin, vmax):
    """Thresholds the data."""
    # If no thresholding and nans, filter them out
    if threshold is None:
        mask = np.logical_not(np.isnan(data))
    else:
        mask = np.abs(data) >= threshold
        if vmin > -threshold:
            mask = np.logical_and(mask, data >= vmin)
        if vmax < threshold:
            mask = np.logical_and(mask, data <= vmax)
    return mask


def _rescale(data, vmin=None, vmax=None):
    """Rescales the data."""
    data_copy = np.copy(data)
    # if no vmin/vmax are passed figure them out from data
    vmin, vmax = _get_bounds(data_copy, vmin, vmax)
    data_copy -= vmin
    data_copy /= vmax - vmin
    return data_copy, vmin, vmax


def _get_bounds(data, vmin=None, vmax=None):
    """Help returning the data bounds."""
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    return vmin, vmax


def _plot_surf_matplotlib(
    coords,
    faces,
    surf_map=None,
    bg_map=None,
    hemi="left",
    view="lateral",
    cmap=None,
    colorbar=False,
    avg_method="mean",
    threshold=None,
    alpha="auto",
    bg_on_data=False,
    darkness=0.7,
    vmin=None,
    vmax=None,
    cbar_vmin=None,
    cbar_vmax=None,
    cbar_tick_format="%.2g",
    title=None,
    output_file=None,
    axes=None,
    figure=None,
):
    """Help for plot_surf.

    This function handles surface plotting when the selected
    engine is matplotlib.
    """
    _default_figsize = [4, 5]
    limits = [coords.min(), coords.max()]

    # Get elevation and azimut from view
    elev, azim = _get_view_plot_surf_matplotlib(hemi, view)

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.get_cmap(plt.rcParamsDefault["image.cmap"])
    # if cmap is given as string, translate to matplotlib cmap
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    figsize = _default_figsize
    # Leave space for colorbar
    if colorbar:
        figsize[0] += 0.7
    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        axes = figure.add_axes((0, 0, 1, 1), projection="3d")
    elif figure is None:
        figure = axes.get_figure()
    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        triangles=faces,
        linewidth=0.1,
        antialiased=False,
        color="white",
    )

    # reduce viewing distance to remove space around mesh
    axes.set_box_aspect(None, zoom=1.3)

    bg_face_colors = _compute_facecolors_matplotlib(
        bg_map, faces, coords.shape[0], darkness, alpha
    )
    if surf_map is not None:
        surf_map_faces = _compute_surf_map_faces_matplotlib(
            surf_map,
            faces,
            avg_method,
            coords.shape[0],
            bg_face_colors.shape[0],
        )
        surf_map_faces, kept_indices, vmin, vmax = _threshold_and_rescale(
            surf_map_faces, threshold, vmin, vmax
        )

        surf_map_face_colors = cmap(surf_map_faces)
        # set transparency of voxels under threshold to 0
        surf_map_face_colors[~kept_indices, 3] = 0
        if bg_on_data:
            # if need be, set transparency of voxels above threshold to 0.7
            # so that background map becomes visible
            surf_map_face_colors[kept_indices, 3] = 0.7

        face_colors = mix_colormaps(surf_map_face_colors, bg_face_colors)

        if colorbar:
            cbar_vmin = cbar_vmin if cbar_vmin is not None else vmin
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
            ticks = _get_ticks_matplotlib(
                cbar_vmin, cbar_vmax, cbar_tick_format, threshold
            )
            our_cmap, norm = _get_cmap_matplotlib(
                cmap, vmin, vmax, cbar_tick_format, threshold
            )
            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, _ = make_axes(
                axes,
                location="right",
                fraction=0.15,
                shrink=0.5,
                pad=0.0,
                aspect=10.0,
            )
            figure.colorbar(
                proxy_mappable,
                cax=cax,
                ticks=ticks,
                boundaries=bounds,
                spacing="proportional",
                format=cbar_tick_format,
                orientation="vertical",
            )

        # fix floating point bug causing highest to sometimes surpass 1
        # (for example 1.0000000000000002)
        face_colors[face_colors > 1] = 1

        p3dcollec.set_facecolors(face_colors)
        p3dcollec.set_edgecolors(face_colors)

    if title is not None:
        axes.set_title(title)
    if output_file is None:
        return figure
    figure.savefig(output_file)
    plt.close()


@fill_doc
def plot_surf(
    surf_mesh=None,
    surf_map=None,
    bg_map=None,
    hemi="left",
    view="lateral",
    engine="matplotlib",
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
    """Plot surfaces with optional background and data.

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray`\
                or a :obj:`~nilearn.surface.InMemoryMesh`, \
                or a :obj:`~nilearn.surface.PolyMesh`, or None
        Surface :term:`mesh` geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the :term:`mesh` :term:`vertices<vertex>`,
        the second containing the indices (into coords)
        of the :term:`mesh` :term:`faces`,
        or a :obj:`~nilearn.surface.InMemoryMesh` object with
        "coordinates" and "faces" attributes,
        or a :obj:`~nilearn.surface.PolyMesh` object,
        or None.
        If None is passed, then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and the mesh from that :obj:`~nilearn.surface.SurfaceImage` instance
        will be used.

    surf_map : :obj:`str` or :class:`numpy.ndarray`\
               or :obj:`~nilearn.surface.SurfaceImage` or None, \
               default=None
        Data to be displayed on the surface :term:`mesh`.
        Can be a file
        (valid formats are .gii, .mgz, .nii, .nii.gz,
        or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`,
        or a :obj:`~nilearn.surface.SurfaceImage` instance.
        If None is passed for ``surf_mesh``
        then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

    bg_map : :obj:`str` or :class:`numpy.ndarray` \
             or :obj:`~nilearn.surface.SurfaceImage` or None,\
             default=None
        Background image to be plotted on the :term:`mesh`
        underneath the surf_data in grayscale,
        most likely a sulcal depth map for realistic shading.
        If the map contains values outside [0, 1],
        it will be rescaled such that all values are in [0, 1].
        Otherwise, it will not be modified.

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine, you need to
            have ``plotly`` installed.

        .. note::
            To be able to save figures to disk with the
            ``plotly`` engine, you need to have
            ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental.
            Please report bugs that you may encounter.

    %(cmap)s
        If None, matplotlib default will be chosen.

    symmetric_cmap : :obj:`bool`, default=False
        Whether to use a symmetric colormap or not.

        .. note::
            This option is currently only implemented for
            the ``plotly`` engine.

        .. versionadded:: 0.9.0

    %(colorbar)s
        Default=False.

    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        When using matplotlib as engine,
        `avg_method` will default to ``"mean"`` if ``None`` is passed.

    threshold : a number or None, default=None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.

    alpha : :obj:`float` or None, default=None
        Alpha level of the :term:`mesh` (not surf_data).
        When using matplotlib as engine,
        `alpha` will default to ``"auto"`` if ``None`` is passed.
        If 'auto' is chosen, alpha will default to 0.5 when no bg_map
        is passed and to 1 if a bg_map is passed.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

    %(vmin)s

    %(vmax)s

    cbar_vmin : :obj:`float` or None, default=None
        Lower bound for the colorbar.
        If None, the value will be set from the data.

    cbar_vmax : :obj:`float` or None, default=None
        Upper bound for the colorbar.
        If None, the value will be set from the data.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(cbar_tick_format)s
        Default="auto" which will select:

        - '%%.2g' (scientific notation) with ``matplotlib`` engine.
        - '.1f' (rounded floats) with ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(title)s

    title_font_size : :obj:`int`, default=18
        Size of the title font (only implemented for the plotly engine).

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure` or\
    :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`
        The surface figure. If ``engine='matplotlib'`` then a
        :class:`~matplotlib.figure.Figure` is returned.
        If ``engine='plotly'``, then a
        :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`
        is returned

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_roi : For plotting statistical maps on brain
        surfaces.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    parameters_not_implemented_in_plotly = {
        "avg_method": avg_method,
        "figure": figure,
        "axes": axes,
        "cbar_vmin": cbar_vmin,
        "cbar_vmax": cbar_vmax,
        "alpha": alpha,
    }

    surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )

    check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    if engine == "plotly":
        for parameter, value in parameters_not_implemented_in_plotly.items():
            if value is not None:
                warn(
                    f"'{parameter}' is not implemented "
                    "for the plotly engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning."
                )

    coords, faces = load_surf_mesh(surf_mesh)

    if engine == "matplotlib":
        # setting defaults
        if avg_method is None:
            avg_method = "mean"
        if alpha is None:
            alpha = "auto"

        if cbar_tick_format == "auto":
            cbar_tick_format = "%.2g"
        fig = _plot_surf_matplotlib(
            coords,
            faces,
            surf_map=surf_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
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
            output_file=output_file,
            axes=axes,
            figure=figure,
        )

    elif engine == "plotly":
        if cbar_tick_format == "auto":
            cbar_tick_format = ".1f"
        fig = _plot_surf_plotly(
            coords,
            faces,
            surf_map=surf_map,
            bg_map=bg_map,
            view=view,
            hemi=hemi,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            threshold=threshold,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            cbar_tick_format=cbar_tick_format,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
        )

    else:
        raise ValueError(
            f"Unknown plotting engine {engine}. "
            "Please use either 'matplotlib' or "
            "'plotly'."
        )

    return fig


def _get_faces_on_edge(faces, parc_idx):
    """Identify which faces lie on the outeredge of the parcellation \
    defined by the indices in parc_idx.

    Parameters
    ----------
    faces : :class:`numpy.ndarray` of shape (n, 3), indices of the mesh faces

    parc_idx : :class:`numpy.ndarray`, indices of the vertices
        of the region to be plotted

    """
    # count how many vertices belong to the given parcellation in each face
    verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

    # test if parcellation forms regions
    if np.all(verts_per_face < 2):
        raise ValueError("Vertices in parcellation do not form region.")

    vertices_on_edge = np.intersect1d(
        np.unique(faces[verts_per_face == 2]), parc_idx
    )
    faces_outside_edge = np.isin(faces, vertices_on_edge).sum(axis=1)

    return np.logical_and(faces_outside_edge > 0, verts_per_face < 3)


@fill_doc
def plot_surf_contours(
    surf_mesh=None,
    roi_map=None,
    hemi=None,
    axes=None,
    figure=None,
    levels=None,
    labels=None,
    colors=None,
    legend=False,
    cmap="tab20",
    title=None,
    output_file=None,
    **kwargs,
):
    """Plot contours of ROIs on a surface, \
    optionally over a statistical map.

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray`\
                or a :obj:`~nilearn.surface.InMemoryMesh`, \
                or a :obj:`~nilearn.surface.PolyMesh`, or None
        Surface :term:`mesh` geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the :term:`mesh` :term:`vertices<vertex>`,
        the second containing the indices (into coords)
        of the :term:`mesh` :term:`faces`,
        or a :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates"
        and "faces" attributes,
        or a :obj:`~nilearn.surface.PolyMesh` object,
        or None.
        If None is passed, then ``roi_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and the mesh from that :obj:`~nilearn.surface.SurfaceImage` instance
        will be used.

    roi_map : :obj:`str` or :class:`numpy.ndarray` or \
              :obj:`~nilearn.surface.SurfaceImage` or None, \
              default=None
        ROI map to be displayed on the surface mesh,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`.
        The value at each :term:`vertex` one inside the ROI
        and zero inside ROI,
        or an integer giving the label number for atlases.
        If None is passed for ``surf_mesh``
        then ``roi_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

    hemi : {"left", "right", None}, default=None
        Hemisphere to display in case a :obj:`~nilearn.surface.SurfaceImage`
        is passed as ``roi_map``
        and / or if PolyMesh is passed as ``surf_mesh``.
        In these cases, if ``hemi`` is set to None, it will default to "left".

        .. versionadded:: 0.11.0

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, uses axes from figure if available, else creates new axes.

    %(figure)s

    levels : :obj:`list` of :obj:`int`, or None, default=None
        A list of indices of the regions that are to be outlined.
        Every index needs to correspond to one index in roi_map.
        If None, all regions in roi_map are used.

    labels : :obj:`list` of :obj:`str` or None, or None, default=None
        A list of labels for the individual regions of interest.
        Provide None as list entry to skip showing the label of that region.
        If None no labels are used.

    colors : :obj:`list` of matplotlib color names or RGBA values, or None \
        default=None
        Colors to be used.

    legend : :obj:`bool`,  optional, default=False
        Whether to plot a legend of region's labels.

    %(cmap)s
        Default='tab20'.

    %(title)s

    %(output_file)s

    kwargs : extra keyword arguments, optional
        Extra keyword arguments passed to
        :func:`~nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    if hemi is None and (
        isinstance(roi_map, SurfaceImage) or isinstance(surf_mesh, PolyMesh)
    ):
        hemi = "left"
    elif (
        hemi is not None
        and not isinstance(roi_map, SurfaceImage)
        and not isinstance(surf_mesh, PolyMesh)
    ):
        warn(
            category=UserWarning,
            message=(
                f"{hemi=} was passed "
                f"with {type(roi_map)=} and {type(surf_mesh)=}.\n"
                "This value will be ignored as it is only used when "
                "'roi_map' is a SurfaceImage instance "
                "and  / or 'surf_mesh' is a PolyMesh instance."
            ),
            stacklevel=2,
        )
    roi_map, surf_mesh, _ = check_surface_plotting_inputs(
        roi_map, surf_mesh, hemi, map_var_name="roi_map"
    )

    if isinstance(figure, PlotlySurfaceFigure):
        raise ValueError(
            "figure argument is a PlotlySurfaceFigure"
            "but it should be None or a matplotlib figure"
        )
    if isinstance(axes, PlotlySurfaceFigure):
        raise ValueError(
            "axes argument is a PlotlySurfaceFigure"
            "but it should be None or a matplotlib axes"
        )
    if figure is None and axes is None:
        figure = plot_surf(surf_mesh, **kwargs)
        axes = figure.axes[0]
    if figure is None:
        figure = axes.get_figure()
    if axes is None:
        axes = figure.axes[0]
    if axes.name != "3d":
        raise ValueError("Axes must be 3D.")
    # test if axes contains Poly3DCollection, if not initialize surface
    if not axes.collections or not isinstance(
        axes.collections[0], Poly3DCollection
    ):
        _ = plot_surf(surf_mesh, axes=axes, **kwargs)

    _, faces = load_surf_mesh(surf_mesh)

    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
    roi = load_surf_data(roi_map)

    if levels is None:
        levels = np.unique(roi_map)
    if colors is None:
        n_levels = len(levels)
        vmax = n_levels
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=vmax)
        colors = [cmap(norm(color_i)) for color_i in range(vmax)]
    else:
        try:
            colors = [to_rgba(color, alpha=1.0) for color in colors]
        except ValueError:
            raise ValueError(
                "All elements of colors need to be either a"
                " matplotlib color string or RGBA values."
            )

    if labels is None:
        labels = [None] * len(levels)
    if not (len(levels) == len(labels) == len(colors)):
        raise ValueError(
            "Levels, labels, and colors "
            "argument need to be either the same length or None."
        )

    patch_list = []
    for level, color, label in zip(levels, colors, labels):
        roi_indices = np.where(roi == level)[0]
        faces_outside = _get_faces_on_edge(faces, roi_indices)
        # Fix: Matplotlib version 3.3.2 to 3.3.3
        # Attribute _facecolors3d changed to _facecolor3d in
        # matplotlib version 3.3.3
        if compare_version(mpl.__version__, "<", "3.3.3"):
            axes.collections[0]._facecolors3d[faces_outside] = color
            if axes.collections[0]._edgecolors3d.size == 0:
                axes.collections[0].set_edgecolor(
                    axes.collections[0]._facecolors3d
                )
            axes.collections[0]._edgecolors3d[faces_outside] = color
        else:
            axes.collections[0]._facecolor3d[faces_outside] = color
            if axes.collections[0]._edgecolor3d.size == 0:
                axes.collections[0].set_edgecolor(
                    axes.collections[0]._facecolor3d
                )
            axes.collections[0]._edgecolor3d[faces_outside] = color
        if label and legend:
            patch_list.append(Patch(color=color, label=label))
    # plot legend only if indicated and labels provided
    if legend and np.any([lbl is not None for lbl in labels]):
        figure.legend(handles=patch_list)
        # if legends, then move title to the left
    if title is None and hasattr(figure._suptitle, "_text"):
        title = figure._suptitle._text
    if title:
        axes.set_title(title)
    if output_file is None:
        return figure
    figure.savefig(output_file)
    plt.close(figure)


@fill_doc
def plot_surf_stat_map(
    surf_mesh=None,
    stat_map=None,
    bg_map=None,
    hemi="left",
    view="lateral",
    engine="matplotlib",
    threshold=None,
    alpha=None,
    vmin=None,
    vmax=None,
    cmap="cold_hot",
    colorbar=True,
    symmetric_cbar="auto",
    cbar_tick_format="auto",
    bg_on_data=False,
    darkness=0.7,
    title=None,
    title_font_size=18,
    output_file=None,
    axes=None,
    figure=None,
    avg_method=None,
    **kwargs,
):
    """Plot a stats map on a surface :term:`mesh` with optional background.

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray`\
                or a :obj:`~nilearn.surface.InMemoryMesh`, \
                or a :obj:`~nilearn.surface.PolyMesh`, or None
        Surface :term:`mesh` geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z
        coordinates of the :term:`mesh` :term:`vertices<vertex>`,
        the second containing the indices (into coords)
        of the :term:`mesh` :term:`faces`,
        or a :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates"
        and "faces" attributes, or a :obj:`~nilearn.surface.PolyMesh` object,
        or None.
        If None is passed, then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and the mesh from
        that :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    stat_map : :obj:`str` or :class:`numpy.ndarray`
        Statistical map to be displayed on the surface :term:`mesh`,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`.
        If None is passed for ``surf_mesh``
        then ``stat_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

    bg_map : :obj:`str` or :class:`numpy.ndarray` or \
             :obj:`~nilearn.surface.SurfaceImage` or None,\
             default=None
        Background image to be plotted on the :term:`mesh` underneath
        the stat_map in grayscale, most likely a sulcal depth map
        for realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf_stat_map``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine you need to
            have ``plotly`` installed.

        .. note::
            To be able to save figures to disk with the ``plotly``
            engine you need to have ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental.
            Please report bugs that you may encounter.


    threshold : a number or None, default=None
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image,
        values below the threshold (in absolute value) are plotted
        as transparent.

    %(cmap)s

    %(cbar_tick_format)s
        Default="auto" which will select:

            - '%%.2g' (scientific notation) with ``matplotlib`` engine.
            - '.1f' (rounded floats) with ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(colorbar)s

        .. note::
            This function uses a symmetric colorbar for the statistical map.

        Default=True.

    alpha : :obj:`float` or 'auto' or None, default=None
        Alpha level of the :term:`mesh` (not the stat_map).
        Will default to ``"auto"`` if ``None`` is passed.
        If 'auto' is chosen, alpha will default to .5 when no bg_map is
        passed and to 1 if a bg_map is passed.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(vmin)s

    %(vmax)s

    %(symmetric_cbar)s

    %(bg_on_data)s

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(title)s

    title_font_size : :obj:`int`, default=18
        Size of the title font (only implemented for the plotly engine).

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        When using matplotlib as engine,
        `avg_method` will default to ``"mean"`` if ``None`` is passed.

        .. versionadded:: 0.10.3dev

    kwargs : :obj:`dict`, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
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
    # Set to None the values that are not used by plotly
    # to avoid warnings thrown by plot_surf
    if engine == "plotly":
        cbar_vmin = None
        cbar_vmax = None

    display = plot_surf(
        surf_mesh,
        surf_map=loaded_stat_map,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        engine=engine,
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
    return display


def _check_hemisphere_is_valid(hemi):
    return hemi in VALID_HEMISPHERES


def _check_hemispheres(hemispheres):
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


def _check_views(views) -> list:
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


def _colorbar_from_array(
    array, vmin, vmax, threshold, symmetric_cbar=True, cmap="cold_hot"
):
    """Generate a custom colorbar for an array.

    Internal function used by plot_img_on_surf

    array : :class:`np.ndarray`
        Any 3D array.

    vmin : :obj:`float`
        lower bound for plotting of stat_map values.

    vmax : :obj:`float`
        upper bound for plotting of stat_map values.

    threshold : :obj:`float`
        If None is given, the colorbar is not thresholded.
        If a number is given, it is used to threshold the colorbar.
        Absolute values lower than threshold are shown in gray.

    kwargs : :obj:`dict`
        Extra arguments passed to get_colorbar_and_data_ranges.

    cmap : :obj:`str`, default='cold_hot'
        The name of a matplotlib or nilearn colormap.

    """
    _, _, vmin, vmax = get_colorbar_and_data_ranges(
        array,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if threshold is None:
        threshold = 0.0

    # set colors to gray for absolute values < threshold
    istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
    our_cmap = LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )
    sm = plt.cm.ScalarMappable(cmap=our_cmap, norm=norm)

    # fake up the array of the scalar mappable.
    sm._A = []

    return sm


@fill_doc
def plot_img_on_surf(
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
    cmap="cold_hot",
    cbar_tick_format="%i",
    **kwargs,
):
    """Plot multiple views of plot_surf_stat_map \
    in a single figure.

    It projects stat_map into meshes and plots views of
    left and right hemispheres. The *views* argument defines the views
    that are shown. This function returns the fig, axes elements from
    matplotlib unless kwargs sets and output_file, in which case nothing
    is returned.

    Parameters
    ----------
    stat_map : :obj:`str` or :class:`pathlib.Path` or 3D Niimg-like object
        See :ref:`extracting_data`.

    surf_mesh : :obj:`str`, :obj:`dict`, or None, default='fsaverage5'
        If str, either one of the two:
        'fsaverage5': the low-resolution fsaverage5 :term:`mesh` (10242 nodes)
        'fsaverage': the high-resolution fsaverage :term:`mesh` (163842 nodes)
        If dict, a dictionary with keys: ['infl_left', 'infl_right',
        'pial_left', 'pial_right', 'sulc_left', 'sulc_right'], where
        values are surface :term:`mesh` geometries as accepted
        by plot_surf_stat_map.

    mask_img : Niimg-like object or None, default=None
        The mask is passed to vol_to_surf.
        Samples falling out of this mask or out of the image are ignored
        during projection of the volume to the surface.
        If ``None``, don't apply any mask.

    %(bg_on_data)s

    hemispheres : :obj:`list` of :obj:`str`, default=None
        Hemispheres to display.
        Will default to ``['left', 'right']`` if ``None`` is passed.

    inflate : :obj:`bool`, default=False
        If True, display images in inflated brain.
        If False, display images in pial surface.

    views : :obj:`list` of :obj:`str`, default=None
        A list containing all views to display.
        The montage will contain as many rows as views specified by
        display mode. Order is preserved, and left and right hemispheres
        are shown on the left and right sides of the figure.
        Will default to ``['lateral', 'medial']`` if ``None`` is passed.
    %(output_file)s
    %(title)s
    %(colorbar)s

        .. note::
            This function uses a symmetric colorbar for the statistical map.

        Default=True.
    %(vmin)s
    %(vmax)s
    %(threshold)s
    %(symmetric_cbar)s
    %(cmap)s
        Default='cold_hot'.
    %(cbar_tick_format)s
    kwargs : :obj:`dict`, optional
        keyword arguments passed to plot_surf_stat_map.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as the default background map for this plotting function.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    nilearn.plotting.plot_surf_stat_map : For info on kwargs options
        accepted by plot_img_on_surf.

    """
    if hemispheres is None:
        hemispheres = ["left", "right"]
    if views is None:
        views = ["lateral", "medial"]
    for arg in ("figure", "axes", "engine"):
        if arg in kwargs:
            raise ValueError(
                f"plot_img_on_surf does not accept {arg} as an argument"
            )

    stat_map = check_niimg_3d(stat_map, dtype="auto")
    modes = _check_views(views)
    hemis = _check_hemispheres(hemispheres)
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

    cbar_h = 0.25
    title_h = 0.25 * (title is not None)
    w, h = plt.figaspect((len(modes) + cbar_h + title_h) / len(hemispheres))
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    height_ratios = [title_h] + [1.0] * len(modes) + [cbar_h]
    grid = gridspec.GridSpec(
        len(modes) + 2,
        len(hemis),
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.0,
    )
    axes = []

    # get vmin and vmax for entire data (all hemis)
    _, _, vmin, vmax = get_colorbar_and_data_ranges(
        image.get_data(stat_map),
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )

    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = None
        # By default, add curv sign background map if mesh is inflated,
        # sulc depth background map otherwise
        if inflate:
            curv_map = surface.load_surf_data(surf_mesh[f"curv_{hemi}"])
            curv_sign_map = (np.sign(curv_map) + 1) / 4 + 0.25
            bg_map = curv_sign_map
        else:
            sulc_map = surf_mesh[f"sulc_{hemi}"]
            bg_map = sulc_map

        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)

        plot_surf_stat_map(
            surf[hemi],
            texture[hemi],
            view=mode,
            hemi=hemi,
            bg_map=bg_map,
            bg_on_data=bg_on_data,
            axes=ax,
            colorbar=False,  # Colorbar created externally.
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            **kwargs,
        )

        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.set_box_aspect(None, zoom=1.3)

    if colorbar:
        sm = _colorbar_from_array(
            image.get_data(stat_map),
            vmin,
            vmax,
            threshold,
            symmetric_cbar=symmetric_cbar,
            cmap=plt.get_cmap(cmap),
        )

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        # Get custom ticks to set in colorbar
        ticks = _get_ticks_matplotlib(vmin, vmax, cbar_tick_format, threshold)
        fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="horizontal",
            ticks=ticks,
            format=cbar_tick_format,
        )

    if title is not None:
        fig.suptitle(title, y=1.0 - title_h / sum(height_ratios), va="bottom")

    if output_file is None:
        return fig, axes
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


@fill_doc
def plot_surf_roi(
    surf_mesh=None,
    roi_map=None,
    bg_map=None,
    hemi="left",
    view="lateral",
    engine="matplotlib",
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
    **kwargs,
):
    """Plot ROI on a surface :term:`mesh` with optional background.

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray`\
                or a :obj:`~nilearn.surface.InMemoryMesh`, \
                or a :obj:`~nilearn.surface.PolyMesh`, or None
        Surface :term:`mesh` geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the :term:`mesh` :term:`vertices<vertex>`,
        the second containing the indices (into coords)
        of the :term:`mesh` :term:`faces`,
        or a :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates"
        and "faces" attributes, or a :obj:`~nilearn.surface.PolyMesh` object,
        or None.
        If None is passed, then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and the mesh
        from that :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    roi_map : :obj:`str` or :class:`numpy.ndarray` or \
              :obj:`list` of :class:`numpy.ndarray` or \
              :obj:`~nilearn.surface.SurfaceImage` or None, \
              default=None
        ROI map to be displayed on the surface :term:`mesh`,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`
        or a :obj:`~nilearn.surface.SurfaceImage` instance.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.
        If None is passed for ``surf_mesh``
        then ``roi_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

    bg_map : :obj:`str` or :class:`numpy.ndarray` or \
             :obj:`~nilearn.surface.SurfaceImage` or None,\
             default=None
        Background image to be plotted on the :term:`mesh` underneath
        the stat_map in grayscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf_roi``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine you need to have
            ``plotly`` installed.

        .. note::
            To be able to save figures to disk with ``plotly`` engine
            you need to have ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental.
            Please report bugs that you may encounter.

    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        When using matplotlib as engine,
        `avg_method` will default to ``"median"`` if ``None`` is passed.

    threshold : a number or None, default=1e-14
        Threshold regions that are labeled 0.
        If you want to use 0 as a label, set threshold to None.

    %(cmap)s
        Default='gist_ncar'.

    %(cbar_tick_format)s
        Default="auto" which defaults to integers format:

            - "%%i" for ``matplotlib`` engine.
            - "." for ``plotly`` engine.

        .. versionadded:: 0.7.1

    alpha : :obj:`float` or 'auto' or None, default=None
        Alpha level of the :term:`mesh` (not surf_data).
        When using matplotlib as engine,
        `alpha` will default to ``"auto"`` if ``None`` is passed.
        If 'auto' is chosen, alpha will default to 0.5 when no bg_map
        is passed and to 1 if a bg_map is passed.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(title)s

    title_font_size : :obj:`int`, default=18
        Size of the title font (only implemented for the plotly engine).

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : Axes instance or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `plt.subplots(subplot_kw={'projection': '3d'})`).
        If None, a new axes is created.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    kwargs : :obj:`dict`, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    roi_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        roi_map, surf_mesh, hemi, bg_map
    )

    if engine == "matplotlib" and avg_method is None:
        avg_method = "median"

    # preload roi and mesh to determine vmin, vmax and give more useful error
    # messages in case of wrong inputs
    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    roi = load_surf_data(roi_map)

    idx_not_na = ~np.isnan(roi)
    if vmin is None:
        vmin = np.nanmin(roi)
    if vmax is None:
        vmax = 1 + np.nanmax(roi)

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
                "Non-integer values in roi_map will no longer be allowed in"
                " Nilearn version 0.13"
            ),
            DeprecationWarning,
        )

    if cbar_tick_format == "auto":
        cbar_tick_format = "." if engine == "plotly" else "%i"

    display = plot_surf(
        mesh,
        surf_map=roi,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        engine=engine,
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
        **kwargs,
    )

    return display
