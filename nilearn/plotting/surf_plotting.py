"""
Functions for surface visualization.
"""
import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec
from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn import image, surface
from nilearn.plotting.cm import cold_hot
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges
from nilearn.surface import (load_surf_data,
                             load_surf_mesh,
                             vol_to_surf)
from nilearn.surface.surface import _check_mesh
from nilearn._utils import check_niimg_3d, fill_doc
from nilearn.plotting.js_plotting_utils import colorscale
from nilearn.plotting.html_surface import (
    _get_vertexcolor,
    _mix_colormaps
)

from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VALID_VIEWS = "anterior", "posterior", "medial", "lateral", "dorsal", "ventral"
VALID_HEMISPHERES = "left", "right"


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


MATPLOTLIB_VIEWS = {"right": {"lateral": (0, 0),
                              "medial": (0, 180),
                              "dorsal": (90, 0),
                              "ventral": (270, 0),
                              "anterior": (0, 90),
                              "posterior": (0, 270)
                              },
                    "left": {"medial": (0, 0),
                             "lateral": (0, 180),
                             "dorsal": (90, 0),
                             "ventral": (270, 0),
                             "anterior": (0, 90),
                             "posterior": (0, 270)
                             }
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
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "ventral": {
        "eye": {"x": 0, "y": 0, "z": -1.5},
        "up": {"x": 0, "y": 1, "z": 0},
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


LAYOUT = {
    "scene": {f"{dim}axis": AXIS_CONFIG for dim in ("x", "y", "z")},
    "paper_bgcolor": "#fff",
    "hovermode": False,
    "margin": {"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
}


def _set_view_plot_surf_plotly(hemi, view):
    """Helper function for plot_surf with plotly engine.

    This function checks the selected hemisphere and view, and
    returns the cameras view.
    """
    if hemi not in VALID_HEMISPHERES:
        raise ValueError(f"hemi must be one of {VALID_HEMISPHERES}")
    if view == 'lateral':
        view = hemi
    elif view == 'medial':
        view = (VALID_HEMISPHERES[0]
                if hemi == VALID_HEMISPHERES[1]
                else VALID_HEMISPHERES[1])
    if view not in CAMERAS:
        raise ValueError(f"view must be one of {VALID_VIEWS}")
    return view


def _configure_title_plotly(title, font_size, color="black"):
    """Helper function for plot_surf with plotly engine.

    This function configures the title if provided.
    """
    if title is None:
        return dict()
    return {"text": title,
            "font": {"size": font_size,
                     "color": color,
                     },
            "y": 0.96,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"}


def _get_cbar_plotly(colorscale, vmin, vmax, cbar_tick_format,
                     fontsize=25, color="black", height=0.5):
    """Helper function for _plot_surf_plotly.

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


def _plot_surf_plotly(coords, faces, surf_map=None, bg_map=None,
                      hemi='left', view='lateral', cmap=None,
                      symmetric_cmap=True, colorbar=False,
                      threshold=None, bg_on_data=False,
                      darkness=.7, vmin=None, vmax=None,
                      cbar_vmin=None, cbar_vmax=None,
                      cbar_tick_format=".1f", title=None,
                      title_font_size=18, output_file=None):
    """Helper function for plot_surf.

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
    try:
        import plotly.graph_objects as go
        from nilearn.plotting.displays import PlotlySurfaceFigure
    except ImportError:
        msg = "Using engine='plotly' requires that ``plotly`` is installed."
        raise ImportError(msg)  # noqa

    x, y, z = coords.T
    i, j, k = faces.T

    if cmap is None:
        cmap = cold_hot

    bg_data = None
    if bg_map is not None:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')

    if surf_map is not None:
        _check_surf_map(surf_map, coords.shape[0])
        colors = colorscale(
            cmap, surf_map, threshold, vmax=vmax, vmin=vmin,
            symmetric_cmap=symmetric_cmap
        )
        vertexcolor = _get_vertexcolor(
            surf_map, colors["cmap"], colors["norm"],
            absolute_threshold=colors["abs_threshold"],
            bg_map=bg_data, bg_on_data=bg_on_data,
            darkness=darkness
        )
    else:
        if bg_data is None:
            bg_data = np.zeros(coords.shape[0])
        colors = colorscale('Greys', bg_data, symmetric_cmap=False)
        vertexcolor = _get_vertexcolor(
            bg_data, colors["cmap"], colors["norm"],
            absolute_threshold=colors["abs_threshold"]
        )

    mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertexcolor)
    fig_data = [mesh_3d]
    if colorbar:
        dummy = _get_cbar_plotly(
            colors["colors"], float(colors["vmin"]), float(colors["vmax"]),
            cbar_tick_format
        )
        fig_data.append(dummy)

    # instantiate plotly figure
    cameras_view = _set_view_plot_surf_plotly(hemi, view)
    fig = go.Figure(data=fig_data)
    fig.update_layout(scene_camera=CAMERAS[cameras_view],
                      title=_configure_title_plotly(title, title_font_size),
                      **LAYOUT)

    # save figure
    if output_file is not None:
        try:
            import kaleido  # noqa: F401
        except ImportError:
            msg = ("Saving figures to file with engine='plotly' requires "
                   "that ``kaleido`` is installed.")
            raise ImportError(msg)  # noqa
    plotly_figure = PlotlySurfaceFigure(figure=fig, output_file=output_file)
    if output_file is not None:
        plotly_figure.savefig()

    return plotly_figure


def _set_view_plot_surf_matplotlib(hemi, view):
    """Helper function for plot_surf with matplotlib engine.

    This function checks the selected hemisphere and view, and
    returns elev and azim.
    """
    if hemi not in VALID_HEMISPHERES:
        raise ValueError(f"hemi must be one of {VALID_HEMISPHERES}")
    if view not in MATPLOTLIB_VIEWS[hemi]:
        raise ValueError(f"view must be one of {VALID_VIEWS}")
    return MATPLOTLIB_VIEWS[hemi][view]


def _check_surf_map(surf_map, n_vertices):
    """Helper function for plot_surf.

    This function checks the dimensions of provided surf_map.
    """
    surf_map_data = load_surf_data(surf_map)
    if surf_map_data.ndim != 1:
        raise ValueError("'surf_map' can only have one dimension "
                         f"but has '{surf_map_data.ndim}' dimensions")
    if surf_map_data.shape[0] != n_vertices:
        raise ValueError('The surf_map does not have the same number '
                         'of vertices as the mesh.')
    return surf_map_data


def _compute_surf_map_faces_matplotlib(surf_map, faces, avg_method,
                                       n_vertices, face_colors_size):
    """Helper function for plot_surf.

    This function computes the surf map faces using the
    provided averaging method.

    .. note::
        This method is called exclusively when using matplotlib,
        since it only supports plotting face-colour maps and not
        vertex-colour maps.

    """
    surf_map_data = _check_surf_map(surf_map, n_vertices)

    # create face values from vertex values by selected avg methods
    error_message = ("avg_method should be either "
                     "['mean', 'median', 'max', 'min'] "
                     "or a custom function")
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
                f"{surf_map_faces[0]} != {face_colors_size}")

        # check that dtype is either int or float
        if not (
            "int" in str(surf_map_faces.dtype)
            or "float" in str(surf_map_faces.dtype)
        ):
            raise ValueError(
                'Array computed with the custom function '
                'from avg_method should be an array of numbers '
                '(int or float)'
            )
    else:
        raise ValueError(error_message)
    return surf_map_faces


def _get_ticks_matplotlib(vmin, vmax, cbar_tick_format):
    """Helper function for plot_surf with matplotlib engine.

    This function computes the tick values for the colorbar.
    """
    # Default number of ticks is 5...
    n_ticks = 5
    # ...unless we are dealing with integers with a small range
    # in this case, we reduce the number of ticks
    if cbar_tick_format == "%i" and vmax - vmin < n_ticks - 1:
        ticks = np.arange(vmin, vmax + 1)
    else:
        # remove duplicate ticks when vmin == vmax, or almost
        ticks = np.unique(np.linspace(vmin, vmax, n_ticks))
    return ticks


def _get_cmap_matplotlib(cmap, vmin, vmax, threshold=None):
    """Helper function for plot_surf with matplotlib engine.

    This function returns the colormap.
    """
    our_cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
    if threshold is not None:
        # set colors to grey for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
    our_cmap = LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, our_cmap.N)
    return our_cmap, norm


def _compute_facecolors_matplotlib(bg_map, faces, n_vertices,
                                   darkness, alpha):
    """Helper function for plot_surf with matplotlib engine.

    This function computes the facecolors.
    """
    if bg_map is None:
        bg_data = np.ones(n_vertices) * 0.5
    else:
        bg_data = np.copy(load_surf_data(bg_map))
        if bg_data.shape[0] != n_vertices:
            raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')

    bg_faces = np.mean(bg_data[faces], axis=1)
    # scale background map if need be
    bg_vmin, bg_vmax = np.min(bg_faces), np.max(bg_faces)
    if (bg_vmin < 0 or bg_vmax > 1):
        bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
        bg_faces = bg_norm(bg_faces)

    if darkness is not None:
        bg_faces *= darkness

    face_colors = plt.cm.gray_r(bg_faces)

    # set alpha if in auto mode
    if alpha == 'auto':
        alpha = .5 if bg_map is None else 1
    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]

    return face_colors


def _threshold_and_rescale(data, threshold, vmin, vmax):
    """Helper function for plot_surf.

    This function thresholds and rescales the provided data.
    """
    data_copy, vmin, vmax = _rescale(data, vmin, vmax)
    return data_copy, _threshold(data, threshold), vmin, vmax


def _threshold(data, threshold):
    """Thresholds the data."""
    # If no thresholding and nans, filter them out
    return (np.logical_not(np.isnan(data)) if threshold is None
            else np.abs(data) >= threshold)


def _rescale(data, vmin=None, vmax=None):
    """Rescales the data."""
    data_copy = np.copy(data)
    # if no vmin/vmax are passed figure them out from data
    vmin, vmax = _get_bounds(data_copy, vmin, vmax)
    data_copy -= vmin
    data_copy /= (vmax - vmin)
    return data_copy, vmin, vmax


def _get_bounds(data, vmin=None, vmax=None):
    """Helper function returning the data bounds."""
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    return vmin, vmax


def _plot_surf_matplotlib(coords, faces, surf_map=None, bg_map=None,
                          hemi='left', view='lateral', cmap=None,
                          colorbar=False, avg_method='mean', threshold=None,
                          alpha='auto', bg_on_data=False,
                          darkness=.7, vmin=None, vmax=None, cbar_vmin=None,
                          cbar_vmax=None, cbar_tick_format='%.2g',
                          title=None, title_font_size=18, output_file=None,
                          axes=None, figure=None, **kwargs):
    """Helper function for plot_surf.

    This function handles surface plotting when the selected
    engine is matplotlib.
    """
    _default_figsize = [4, 4]
    limits = [coords.min(), coords.max()]

    # set view
    elev, azim = _set_view_plot_surf_matplotlib(hemi, view)

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.get_cmap(plt.rcParamsDefault['image.cmap'])
    # if cmap is given as string, translate to matplotlib cmap
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    figsize = _default_figsize
    # Leave space for colorbar
    if colorbar:
        figsize[0] += .7
    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        axes = figure.add_axes((0, 0, 1, 1), projection="3d")
    else:
        if figure is None:
            figure = axes.get_figure()
    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.set_box_aspect(None, zoom=1.3)

    bg_face_colors = _compute_facecolors_matplotlib(
        bg_map, faces, coords.shape[0], darkness, alpha
    )
    if surf_map is not None:
        surf_map_faces = _compute_surf_map_faces_matplotlib(
            surf_map, faces, avg_method, coords.shape[0],
            bg_face_colors.shape[0]
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

        face_colors = _mix_colormaps(
            surf_map_face_colors,
            bg_face_colors
        )

        if colorbar:
            cbar_vmin = cbar_vmin if cbar_vmin is not None else vmin
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
            ticks = _get_ticks_matplotlib(cbar_vmin, cbar_vmax,
                                          cbar_tick_format)
            our_cmap, norm = _get_cmap_matplotlib(cmap, vmin, vmax, threshold)
            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)
            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, kw = make_axes(axes, location='right', fraction=.15,
                                shrink=.5, pad=.0, aspect=10.)
            figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks,
                boundaries=bounds, spacing='proportional',
                format=cbar_tick_format, orientation='vertical')

        p3dcollec.set_facecolors(face_colors)

    if title is not None:
        axes.set_title(title)
    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close()
    else:
        return figure


@fill_doc
def plot_surf(surf_mesh, surf_map=None, bg_map=None,
              hemi='left', view='lateral', engine='matplotlib',
              cmap=None, symmetric_cmap=False, colorbar=False,
              avg_method='mean', threshold=None, alpha='auto',
              bg_on_data=False, darkness=.7,
              vmin=None, vmax=None,
              cbar_vmin=None, cbar_vmax=None, cbar_tick_format="auto",
              title=None, title_font_size=18, output_file=None, axes=None,
              figure=None, **kwargs):
    """Plotting of surfaces with optional background and data

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray or Mesh
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces, or a Mesh object with
        "coordinates" and "faces" attributes.

    surf_map : str or numpy.ndarray, optional
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each vertex of the surf_mesh.

    bg_map : str or numpy.ndarray, optional
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.

    %(hemi)s
    %(view)s
    engine : {'matplotlib', 'plotly'}, optional

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

        Default='matplotlib'.
    %(cmap)s
        If None, matplotlib default will be chosen.
    symmetric_cmap : :obj:`bool`, optional
        Whether to use a symmetric colormap or not.

        .. note::
            This option is currently only implemented for
            the ``plotly`` engine.

        .. versionadded:: 0.9.0

        Default=False.
    %(colorbar)s
        Default=False.
    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        Default='mean'.

    threshold : a number or None, default is None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.

    alpha : float or 'auto', optional
        Alpha level of the mesh (not surf_data).
        If 'auto' is chosen, alpha will default to .5 when no bg_map
        is passed and to 1 if a bg_map is passed.
        Default='auto'.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s
        Default=False.

    %(darkness)s
        Default=1.

    %(vmin)s
    %(vmax)s
    cbar_vmin, cbar_vmax : float, float, optional
        Lower / upper bounds for the colorbar.
        If None, the values will be set from the data.
        Default values are None.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(cbar_tick_format)s
        Default="auto" which will select:

            - '%%.2g' (scientific notation) with ``matplotlib`` engine.
            - '.1f' (rounded floats) with ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(title)s
    title_font_size : :obj:`int`, optional
        Size of the title font.

        .. versionadded:: 0.9.0

        Default=18.
    %(output_file)s
    axes : instance of matplotlib axes, None, optional
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
    coords, faces = load_surf_mesh(surf_mesh)
    if engine == 'matplotlib':
        if cbar_tick_format == "auto":
            cbar_tick_format = '%.2g'
        fig = _plot_surf_matplotlib(
            coords, faces, surf_map=surf_map, bg_map=bg_map, hemi=hemi,
            view=view, cmap=cmap, colorbar=colorbar, avg_method=avg_method,
            threshold=threshold, alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness, vmin=vmin, vmax=vmax, cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax, cbar_tick_format=cbar_tick_format,
            title=title, title_font_size=title_font_size,
            output_file=output_file, axes=axes, figure=figure, **kwargs)
    elif engine == 'plotly':
        if cbar_tick_format == "auto":
            cbar_tick_format = ".1f"
        fig = _plot_surf_plotly(
            coords, faces, surf_map=surf_map, bg_map=bg_map, view=view,
            hemi=hemi, cmap=cmap, symmetric_cmap=symmetric_cmap,
            colorbar=colorbar, threshold=threshold,
            bg_on_data=bg_on_data, darkness=darkness,
            vmin=vmin, vmax=vmax, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
            cbar_tick_format=cbar_tick_format, title=title,
            title_font_size=title_font_size, output_file=output_file)
    else:
        raise ValueError(f"Unknown plotting engine {engine}. "
                         "Please use either 'matplotlib' or "
                         "'plotly'.")
    return fig


def _get_faces_on_edge(faces, parc_idx):
    '''
    Internal function for identifying which faces lie on the outer
    edge of the parcellation defined by the indices in parc_idx.

    Parameters
    ----------
    faces : numpy.ndarray of shape (n, 3), indices of the mesh faces

    parc_idx : numpy.ndarray, indices of the vertices
        of the region to be plotted

    '''
    # count how many vertices belong to the given parcellation in each face
    verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

    # test if parcellation forms regions
    if np.all(verts_per_face < 2):
        raise ValueError('Vertices in parcellation do not form region.')

    vertices_on_edge = np.intersect1d(np.unique(faces[verts_per_face == 2]),
                                      parc_idx)
    faces_outside_edge = np.isin(faces, vertices_on_edge).sum(axis=1)

    return np.logical_and(faces_outside_edge > 0, verts_per_face < 3)


@fill_doc
def plot_surf_contours(surf_mesh, roi_map, axes=None, figure=None, levels=None,
                       labels=None, colors=None, legend=False, cmap='tab20',
                       title=None, output_file=None, **kwargs):
    """Plotting contours of ROIs on a surface, optionally over a statistical map.

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    roi_map : str or numpy.ndarray or list of numpy.ndarray
        ROI map to be displayed on the surface mesh, can be a file
        (valid formats are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific
        files such as .annot or .label), or
        a Numpy array with a value for each vertex of the surf_mesh.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.

    axes : instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, uses axes from figure if available, else creates new axes.
    %(figure)s
    levels : list of integers, or None, optional
        A list of indices of the regions that are to be outlined.
        Every index needs to correspond to one index in roi_map.
        If None, all regions in roi_map are used.

    labels : list of strings or None, or None, optional
        A list of labels for the individual regions of interest.
        Provide None as list entry to skip showing the label of that region.
        If None no labels are used.

    colors : list of matplotlib color names or RGBA values, or None, optional
        Colors to be used.

    legend : boolean,  optional
        Whether to plot a legend of region's labels. Default=False.
    %(cmap)s
        Default='tab20'.
    %(title)s
    %(output_file)s

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    """
    if figure is None and axes is None:
        figure = plot_surf(surf_mesh, **kwargs)
        axes = figure.axes[0]
    if figure is None:
        figure = axes.get_figure()
    if axes is None:
        axes = figure.axes[0]
    if axes.name != '3d':
        raise ValueError('Axes must be 3D.')
    # test if axes contains Poly3DCollection, if not initialize surface
    if not axes.collections or not isinstance(axes.collections[0],
                                              Poly3DCollection):
        _ = plot_surf(surf_mesh, axes=axes, **kwargs)

    coords, faces = load_surf_mesh(surf_mesh)
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
            colors = [to_rgba(color, alpha=1.) for color in colors]
        except ValueError:
            raise ValueError('All elements of colors need to be either a'
                             ' matplotlib color string or RGBA values.')

    if labels is None:
        labels = [None] * len(levels)
    if not (len(levels) == len(labels) == len(colors)):
        raise ValueError('Levels, labels, and colors '
                         'argument need to be either the same length or None.')

    patch_list = []
    for level, color, label in zip(levels, colors, labels):
        roi_indices = np.where(roi == level)[0]
        faces_outside = _get_faces_on_edge(faces, roi_indices)
        # Fix: Matplotlib version 3.3.2 to 3.3.3
        # Attribute _facecolors3d changed to _facecolor3d in
        # matplotlib version 3.3.3
        try:
            axes.collections[0]._facecolors3d[faces_outside] = color
        except AttributeError:
            axes.collections[0]._facecolor3d[faces_outside] = color
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
    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
    else:
        return figure


@fill_doc
def plot_surf_stat_map(surf_mesh, stat_map, bg_map=None,
                       hemi='left', view='lateral', engine='matplotlib',
                       threshold=None, alpha='auto', vmax=None,
                       cmap='cold_hot', colorbar=True,
                       symmetric_cbar="auto", cbar_tick_format="auto",
                       bg_on_data=False, darkness=.7,
                       title=None, title_font_size=18, output_file=None,
                       axes=None, figure=None, **kwargs):
    """Plotting a stats map on a surface mesh with optional background

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray or Mesh
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z
        coordinates of the mesh vertices, the second containing the
        indices (into coords) of the mesh faces, or a Mesh object
        with "coordinates" and "faces" attributes.

    stat_map : str or numpy.ndarray
        Statistical map to be displayed on the surface mesh, can
        be a file (valid formats are .gii, .mgz, .nii, .nii.gz, or
        Freesurfer specific files such as .thickness, .area, .curv,
        .sulc, .annot, .label) or
        a Numpy array with a value for each vertex of the surf_mesh.

    bg_map : str or numpy.ndarray, optional
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.

    %(hemi)s
    %(view)s
    engine : {'matplotlib', 'plotly'}, optional

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

        Default='matplotlib'.

    threshold : a number or None, optional
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

    alpha : float or 'auto', optional
        Alpha level of the mesh (not the stat_map).
        If 'auto' is chosen, alpha will default to .5 when no bg_map is
        passed and to 1 if a bg_map is passed.
        Default='auto'.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(vmax)s
    %(symmetric_cbar)s
        Default='auto'.
    %(bg_on_data)s
        Default=False.

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(title)s
    title_font_size : :obj:`int`, optional
        Size of the title font.

        .. versionadded:: 0.9.0

        Default=18.
    %(output_file)s
    axes : instance of matplotlib axes, None, optional
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

    kwargs : dict, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    """
    loaded_stat_map = load_surf_data(stat_map)

    # Call _get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        loaded_stat_map, vmax, symmetric_cbar, kwargs)

    display = plot_surf(
        surf_mesh, surf_map=loaded_stat_map,
        bg_map=bg_map, hemi=hemi,
        view=view, engine=engine, avg_method='mean', threshold=threshold,
        cmap=cmap, symmetric_cmap=True, colorbar=colorbar,
        cbar_tick_format=cbar_tick_format, alpha=alpha,
        bg_on_data=bg_on_data, darkness=darkness,
        vmax=vmax, vmin=vmin,
        title=title, title_font_size=title_font_size, output_file=output_file,
        axes=axes, figure=figure, cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax, **kwargs
    )
    return display


def _check_hemispheres(hemispheres):
    """Checks whether the hemispheres passed to in plot_img_on_surf are
    correct.

    hemispheres : list
        Any combination of 'left' and 'right'.

    """
    invalid_hemi = any([hemi not in VALID_HEMISPHERES for hemi in hemispheres])
    if invalid_hemi:
        supported = "Supported hemispheres:\n" + str(VALID_HEMISPHERES)
        raise ValueError("Invalid hemispheres definition!\n" + supported)
    return hemispheres


def _check_views(views) -> list:
    """Checks whether the views passed to in plot_img_on_surf are
    correct.

    views : list
        Any combination of "anterior", "posterior", "medial", "lateral",
        "dorsal", "ventral".

    """
    invalid_view = any([view not in VALID_VIEWS for view in views])
    if invalid_view:
        supported = "Supported views:\n" + str(VALID_VIEWS)
        raise ValueError("Invalid view definition!\n" + supported)
    return views


def _colorbar_from_array(array, vmax, threshold, kwargs,
                         cmap='cold_hot'):
    """Generate a custom colorbar for an array.

    Internal function used by plot_img_on_surf

    array : np.ndarray
        Any 3D array.

    vmax : float
        upper bound for plotting of stat_map values.

    threshold : float
        If None is given, the colorbar is not thresholded.
        If a number is given, it is used to threshold the colorbar.
        Absolute values lower than threshold are shown in gray.

    kwargs : dict
        Extra arguments passed to _get_colorbar_and_data_ranges.

    cmap : str, optional
        The name of a matplotlib or nilearn colormap.
        Default='cold_hot'.

    """
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        array, vmax, True, kwargs
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if threshold is None:
        threshold = 0.

    # set colors to grey for absolute values < threshold
    istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)
    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable.
    sm._A = []

    return sm


@fill_doc
def plot_img_on_surf(stat_map, surf_mesh='fsaverage5', mask_img=None,
                     hemispheres=['left', 'right'],
                     bg_on_data=False, inflate=False,
                     views=['lateral', 'medial'],
                     output_file=None, title=None, colorbar=True,
                     vmax=None, threshold=None,
                     cmap='cold_hot', **kwargs):
    """Convenience function to plot multiple views of plot_surf_stat_map
    in a single figure. It projects stat_map into meshes and plots views of
    left and right hemispheres. The *views* argument defines the views
    that are shown. This function returns the fig, axes elements from
    matplotlib unless kwargs sets and output_file, in which case nothing
    is returned.

    Parameters
    ----------
    stat_map : str or 3D Niimg-like object
        See :ref:`extracting_data`.

    surf_mesh : str, dict, or None, optional
        If str, either one of the two:
        'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)
        'fsaverage': the high-resolution fsaverage mesh (163842 nodes)
        If dict, a dictionary with keys: ['infl_left', 'infl_right',
        'pial_left', 'pial_right', 'sulc_left', 'sulc_right'], where
        values are surface mesh geometries as accepted by plot_surf_stat_map.
        Default='fsaverage5'.

    mask_img : Niimg-like object or None, optional
        The mask is passed to vol_to_surf.
        Samples falling out of this mask or out of the image are ignored
        during projection of the volume to the surface.
        If ``None``, don't apply any mask.

    %(bg_on_data)s
        Default=False.

    %(hemispheres)s

    inflate : bool, optional
        If True, display images in inflated brain.
        If False, display images in pial surface.
        Default=False.

    views : list of strings, optional
        A list containing all views to display.
        The montage will contain as many rows as views specified by
        display mode. Order is preserved, and left and right hemispheres
        are shown on the left and right sides of the figure.
        Default=['lateral', 'medial'].
    %(output_file)s
    %(title)s
    %(colorbar)s

        .. note::
            This function uses a symmetric colorbar for the statistical map.

        Default=True.
    %(vmax)s
    %(threshold)s
    %(cmap)s
        Default='cold_hot'.
    kwargs : dict, optional
        keyword arguments passed to plot_surf_stat_map.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as the default background map for this plotting function.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    nilearn.plotting.plot_surf_stat_map : For info on kwargs options
        accepted by plot_img_on_surf.

    """
    for arg in ('figure', 'axes'):
        if arg in kwargs:
            raise ValueError(('plot_img_on_surf does not'
                              ' accept %s as an argument' % arg))

    stat_map = check_niimg_3d(stat_map, dtype='auto')
    modes = _check_views(views)
    hemis = _check_hemispheres(hemispheres)
    surf_mesh = _check_mesh(surf_mesh)

    mesh_prefix = "infl" if inflate else "pial"
    surf = {
        'left': surf_mesh[mesh_prefix + '_left'],
        'right': surf_mesh[mesh_prefix + '_right'],
    }

    texture = {
        'left': vol_to_surf(stat_map, surf_mesh['pial_left'],
                            mask_img=mask_img),
        'right': vol_to_surf(stat_map, surf_mesh['pial_right'],
                             mask_img=mask_img)
    }

    cbar_h = .25
    title_h = .25 * (title is not None)
    w, h = plt.figaspect((len(modes) + cbar_h + title_h) / len(hemispheres))
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    height_ratios = [title_h] + [1.] * len(modes) + [cbar_h]
    grid = gridspec.GridSpec(
        len(modes) + 2, len(hemis),
        left=0., right=1., bottom=0., top=1.,
        height_ratios=height_ratios, hspace=0.0, wspace=0.0)
    axes = []

    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = None
        # By default, add curv sign background map if mesh is inflated,
        # sulc depth background map otherwise
        if inflate:
            curv_map = surface.load_surf_data(
                surf_mesh["curv_{}".format(hemi)]
            )
            curv_sign_map = (np.sign(curv_map) + 1) / 4 + 0.25
            bg_map = curv_sign_map
        else:
            sulc_map = surf_mesh['sulc_%s' % hemi]
            bg_map = sulc_map

        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)
        plot_surf_stat_map(surf[hemi], texture[hemi],
                           view=mode, hemi=hemi,
                           bg_map=bg_map,
                           bg_on_data=bg_on_data,
                           axes=ax,
                           colorbar=False,  # Colorbar created externally.
                           vmax=vmax,
                           threshold=threshold,
                           cmap=cmap,
                           **kwargs)
        # ax.set_facecolor("#e0e0e0")
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.set_box_aspect(None, zoom=1.3)

    if colorbar:
        sm = _colorbar_from_array(image.get_data(stat_map),
                                  vmax, threshold, kwargs,
                                  cmap=plt.get_cmap(cmap))

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes


@fill_doc
def plot_surf_roi(surf_mesh, roi_map, bg_map=None,
                  hemi='left', view='lateral', engine='matplotlib',
                  threshold=1e-14, alpha='auto', vmin=None, vmax=None,
                  cmap='gist_ncar', cbar_tick_format="auto",
                  bg_on_data=False, darkness=.7,
                  title=None, title_font_size=18,
                  output_file=None, axes=None,
                  figure=None, **kwargs):
    """ Plotting ROI on a surface mesh with optional background

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray or Mesh
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z
        coordinates of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces, or a Mesh object with
        "coordinates" and "faces" attributes.

    roi_map : str or numpy.ndarray or list of numpy.ndarray
        ROI map to be displayed on the surface mesh, can be a file
        (valid formats are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific
        files such as .annot or .label), or
        a Numpy array with a value for each vertex of the surf_mesh.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.

    bg_map : str or numpy.ndarray, optional
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.

    %(hemi)s
    %(view)s
    engine : {'matplotlib', 'plotly'}, optional

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

        Default='matplotlib'.

    threshold : a number or None, optional
        Threshold regions that are labelled 0.
        If you want to use 0 as a label, set threshold to None.
        Default=1e-14.
    %(cmap)s
        Default='gist_ncar'.
    %(cbar_tick_format)s
        Default="auto" which defaults to integers format:

            - "%%i" for ``matplotlib`` engine.
            - "." for ``plotly`` engine.

        .. versionadded:: 0.7.1

    alpha : float or 'auto', optional
        Alpha level of the mesh (not the stat_map). If default,
        alpha will default to .5 when no bg_map is passed
        and to 1 if a bg_map is passed.
        Default='auto'.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s
        Default=False.

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(title)s
    title_font_size : :obj:`int`, optional
        Size of the title font.

        .. versionadded:: 0.9.0

        Default=18.
    %(output_file)s
    axes : Axes instance or None, optional
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

    kwargs : dict, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    """
    # preload roi and mesh to determine vmin, vmax and give more useful error
    # messages in case of wrong inputs

    roi = load_surf_data(roi_map)
    if vmin is None:
        vmin = np.nanmin(roi)
    if vmax is None:
        vmax = 1 + np.nanmax(roi)

    mesh = load_surf_mesh(surf_mesh)

    if roi.ndim != 1:
        raise ValueError('roi_map can only have one dimension but has '
                         '%i dimensions' % roi.ndim)
    if roi.shape[0] != mesh[0].shape[0]:
        raise ValueError('roi_map does not have the same number of vertices '
                         'as the mesh. If you have a list of indices for the '
                         'ROI you can convert them into a ROI map like this:\n'
                         'roi_map = np.zeros(n_vertices)\n'
                         'roi_map[roi_idx] = 1')

    if cbar_tick_format == "auto":
        cbar_tick_format = "." if engine == "plotly" else "%i"
    display = plot_surf(mesh, surf_map=roi, bg_map=bg_map,
                        hemi=hemi, view=view, engine=engine,
                        avg_method='median', threshold=threshold,
                        cmap=cmap, symmetric_cmap=False,
                        cbar_tick_format=cbar_tick_format,
                        alpha=alpha, bg_on_data=bg_on_data,
                        darkness=darkness, vmin=vmin, vmax=vmax,
                        title=title, title_font_size=title_font_size,
                        output_file=output_file, axes=axes,
                        figure=figure, **kwargs)

    return display
