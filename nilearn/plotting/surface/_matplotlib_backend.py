"""Functions specific to "matplotlib" backend for surface visualization
functions in :obj:`~nilearn.plotting.surface.surf_plotting`.

Any imports from "matplotlib" package, or "matplotlib" engine specific utility
functions in :obj:`~nilearn.plotting.surface` should be in this file.
"""

import itertools
from warnings import warn

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.helpers import compare_version
from nilearn._utils.logger import find_stack_level
from nilearn.image import get_data
from nilearn.plotting import cm
from nilearn.plotting._engine_utils import to_color_strings
from nilearn.plotting._utils import (
    get_cbar_ticks,
    get_colorbar_and_data_ranges,
    save_figure_if_needed,
)
from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.surface._utils import (
    DEFAULT_HEMI,
    check_engine_params,
    check_surf_map,
    check_surface_plotting_inputs,
    get_bg_data,
    get_faces_on_edge,
    sanitize_hemi_view,
)
from nilearn.surface import load_surf_data, load_surf_mesh

try:
    import matplotlib.pyplot as plt
    from matplotlib import __version__ as mpl_version
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import make_axes
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    from nilearn.plotting._utils import engine_warning

    engine_warning("matplotlib")

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
    "both": {
        "right": (0, 0),
        "left": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
        "anterior": (0, 90),
        "posterior": (0, 270),
    },
}


def _adjust_colorbar_and_data_ranges(
    stat_map, vmin=None, vmax=None, symmetric_cbar=None
):
    """Adjust colorbar and data ranges for 'matplotlib' engine.

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
    return get_colorbar_and_data_ranges(
        stat_map,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )


def _adjust_plot_roi_params(params):
    """Adjust avg_method and cbar_tick_format values for 'matplotlib' engine.

    Sets the values in params dict.

    Parameters
    ----------
    params : dict
        dictionary to set the adjusted parameters
    """
    avg_method = params.get("avg_method", None)
    if avg_method is None:
        params["avg_method"] = "median"

    cbar_tick_format = params.get("cbar_tick_format", "auto")
    if cbar_tick_format == "auto":
        params["cbar_tick_format"] = "%i"


def _normalize_bg_data(data):
    """Normalize specified ``data`` and return.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        An array containing surface data

    Returns
    -------
    data : :obj:`numpy.ndarray`
        An array containing normalized surface data
    """
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    if vmin < 0 or vmax > 1:
        norm = Normalize(vmin=vmin, vmax=vmax)
        data = norm(data)
    return data


# TODO (nilearn >= 0.13.0) remove
def _apply_darkness(data, darkness):
    if darkness is not None:
        data *= darkness
        warn(
            (
                "The `darkness` parameter will be deprecated in release 0.13. "
                "We recommend setting `darkness` to None"
            ),
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
    return data


def _get_vertexcolor(
    surf_map,
    cmap,
    norm,
    absolute_threshold=None,
    bg_map=None,
    bg_on_data=None,
    darkness=None,
):
    """Get the color of the vertices."""
    bg_data = get_bg_data(bg_map, len(surf_map))

    # scale background map if need be
    bg_data = _normalize_bg_data(bg_data)

    # TODO (nilearn >= 0.13.0) remove
    bg_data = _apply_darkness(bg_data, darkness)
    bg_colors = plt.get_cmap("Greys")(bg_data)

    # select vertices which are filtered out by the threshold
    if absolute_threshold is None:
        under_threshold = np.zeros_like(surf_map, dtype=bool)
    else:
        under_threshold = np.abs(surf_map) < absolute_threshold

    surf_colors = cmap(norm(surf_map).data)
    # set transparency of voxels under threshold to 0
    surf_colors[under_threshold, 3] = 0
    if bg_on_data:
        # if need be, set transparency of voxels above threshold to 0.7
        # so that background map becomes visible
        surf_colors[~under_threshold, 3] = 0.7

    vertex_colors = cm.mix_colormaps(surf_colors, bg_colors)

    return to_color_strings(vertex_colors)


def _colorbar_from_array(
    array,
    vmin,
    vmax,
    threshold,
    symmetric_cbar=True,
    cmap=DEFAULT_DIVERGING_CMAP,
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


def _compute_facecolors(bg_map, faces, n_vertices, darkness, alpha):
    """Help for plot_surf with matplotlib engine.

    This function computes the facecolors.
    """
    bg_data = get_bg_data(bg_map, n_vertices)
    bg_faces = np.mean(bg_data[faces], axis=1)
    # scale background map if need be
    bg_faces = _normalize_bg_data(bg_faces)

    # TODO (nilearn >= 0.13.0) remove
    bg_faces = _apply_darkness(bg_faces, darkness)
    face_colors = plt.cm.gray_r(bg_faces)

    # set alpha if in auto mode
    if alpha == "auto":
        alpha = 0.5 if bg_map is None else 1
    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]

    return face_colors


def _compute_surf_map_faces(
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
    surf_map_data = check_surf_map(surf_map, n_vertices)

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


def _get_bounds(data, vmin=None, vmax=None):
    """Help returning the data bounds."""
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax

    if vmin == vmax == 0:
        # try to avoid divide by 0 warnings / errors downstream
        vmax = 1
        vmin = -1

    return vmin, vmax


def _get_cmap(cmap, vmin, vmax, cbar_tick_format, threshold=None):
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
                "but configured the colorbar to use integer formatting.",
                stacklevel=find_stack_level(),
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


def _get_ticks(vmin, vmax, cbar_tick_format, threshold):
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
        return get_cbar_ticks(vmin, vmax, threshold, n_ticks)


def _rescale(data, vmin=None, vmax=None):
    """Rescales the data."""
    data_copy = np.copy(data)
    # if no vmin/vmax are passed figure them out from data
    vmin, vmax = _get_bounds(data_copy, vmin, vmax)
    data_copy -= vmin
    data_copy /= vmax - vmin
    return data_copy, vmin, vmax


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


def _threshold_and_rescale(data, threshold, vmin, vmax):
    """Help for plot_surf.

    This function thresholds and rescales the provided data.
    """
    data_copy, vmin, vmax = _rescale(data, vmin, vmax)
    return data_copy, _threshold(data, threshold, vmin, vmax), vmin, vmax


def _check_figure_axes_inputs(figure, axes):
    """Check if the specified figure and axes are matplotlib objects."""
    if figure is not None and not isinstance(figure, plt.Figure):
        raise ValueError(
            "figure argument should be None or a 'matplotlib.pyplot.Figure'."
        )
    if axes is not None and not isinstance(axes, plt.Axes):
        raise ValueError(
            "axes argument should be None or a 'matplotlib.pyplot.Axes'."
        )


def _get_view_plot_surf(hemi, view):
    """Check ``hemi`` and ``view``, and return `elev` and `azim` for
    matplotlib engine.
    """
    view = sanitize_hemi_view(hemi, view)
    if isinstance(view, str):
        if hemi == "both" and view in ["lateral", "medial"]:
            raise ValueError(
                "Invalid view definition: when hemi is 'both', "
                "view cannot be 'lateral' or 'medial'.\n"
                "Maybe you meant 'left' or 'right'?"
            )
        return MATPLOTLIB_VIEWS[hemi][view]
    return view


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
    """Implement 'matplotlib' backend code for
    `~nilearn.plotting.surface.surf_plotting.plot_surf` function.
    """
    parameters_not_implemented_in_matplotlib = {
        "symmetric_cmap": symmetric_cmap,
        "title_font_size": title_font_size,
    }
    check_engine_params(parameters_not_implemented_in_matplotlib, "matplotlib")

    # adjust values
    avg_method = "mean" if avg_method is None else avg_method
    alpha = "auto" if alpha is None else alpha
    cbar_tick_format = (
        "%.2g" if cbar_tick_format == "auto" else cbar_tick_format
    )
    # Leave space for colorbar
    figsize = [4.7, 5] if colorbar else [4, 5]

    coords, faces = load_surf_mesh(surf_mesh)

    limits = [coords.min(), coords.max()]

    # Get elevation and azimut from view
    elev, azim = _get_view_plot_surf(hemi, view)

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.get_cmap(plt.rcParamsDefault["image.cmap"])
    # if cmap is given as string, translate to matplotlib cmap
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        axes = figure.add_axes((0, 0, 1, 1), projection="3d")
    elif figure is None:
        figure = axes.get_figure()
    axes.set_xlim(*limits)
    axes.set_ylim(*limits)

    try:
        axes.view_init(elev=elev, azim=azim)
    except AttributeError:
        raise AttributeError(
            "'Axes' object has no attribute 'view_init'.\n"
            "Remember that the projection must be '3d'.\n"
            "For example:\n"
            "\t plt.subplots(subplot_kw={'projection': '3d'})"
        )
    except Exception as e:  # pragma: no cover
        raise e

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

    bg_face_colors = _compute_facecolors(
        bg_map, faces, coords.shape[0], darkness, alpha
    )
    if surf_map is not None:
        surf_map_faces = _compute_surf_map_faces(
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

            # in rare cases where plotting an image of zeroes
            # this avoids a matplolib error
            if cbar_vmax == cbar_vmin:
                cbar_vmax += 1
                cbar_vmin += -1

            ticks = _get_ticks(
                cbar_vmin, cbar_vmax, cbar_tick_format, threshold
            )
            our_cmap, norm = _get_cmap(
                cmap, vmin, vmax, cbar_tick_format, threshold
            )
            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            figure._colorbar_ax, _ = make_axes(
                axes,
                location="right",
                fraction=0.15,
                shrink=0.5,
                pad=0.0,
                aspect=10.0,
            )
            figure._cbar = figure.colorbar(
                proxy_mappable,
                cax=figure._colorbar_ax,
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

    return save_figure_if_needed(figure, output_file)


def _plot_surf_contours(
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
    """Implement 'matplotlib' backend code for
    `~nilearn.plotting.surface.surf_plotting.plot_surf_contours` function.
    """
    _check_figure_axes_inputs(figure, axes)

    if figure is None and axes is None:
        figure = _plot_surf(surf_mesh, hemi=hemi, **kwargs)
        axes = figure.axes[0]
    elif figure is None:
        figure = axes.get_figure()
    elif axes is None:
        axes = figure.axes[0]

    if axes.name != "3d":
        raise ValueError("Axes must be 3D.")

    # test if axes contains Poly3DCollection, if not initialize surface
    if not axes.collections or not isinstance(
        axes.collections[0], Poly3DCollection
    ):
        _ = _plot_surf(surf_mesh, hemi=hemi, axes=axes, **kwargs)

    if levels is None:
        levels = np.unique(roi_map)

    if labels is None:
        labels = [None] * len(levels)

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

    if not (len(levels) == len(labels) == len(colors)):
        raise ValueError(
            "Levels, labels, and colors "
            "argument need to be either the same length or None."
        )

    _, faces = load_surf_mesh(surf_mesh)
    roi = load_surf_data(roi_map)

    patch_list = []
    for level, color, label in zip(levels, colors, labels):
        roi_indices = np.where(roi == level)[0]
        faces_outside = get_faces_on_edge(faces, roi_indices)
        # Fix: Matplotlib version 3.3.2 to 3.3.3
        # Attribute _facecolors3d changed to _facecolor3d in
        # matplotlib version 3.3.3
        if compare_version(mpl_version, "<", "3.3.3"):
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

    return save_figure_if_needed(figure, output_file)


def _plot_img_on_surf(
    surf,
    surf_mesh,
    stat_map,
    texture,
    hemis,
    modes,
    bg_on_data=False,
    inflate=False,
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
    """Implement 'matplotlib' backend code for
    `~nilearn.plotting.surface.surf_plotting.plot_img_on_surf` function.
    """
    if symmetric_cbar is None:
        symmetric_cbar = "auto"
    if cbar_tick_format is None:
        cbar_tick_format = "%i"

    cbar_h = 0.25
    title_h = 0.25 * (title is not None)
    w, h = plt.figaspect((len(modes) + cbar_h + title_h) / len(hemis))
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    height_ratios = [title_h] + [1.0] * len(modes) + [cbar_h]
    grid = GridSpec(
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

    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = None
        # By default, add curv sign background map if mesh is inflated,
        # sulc depth background map otherwise
        if inflate:
            curv_map = load_surf_data(surf_mesh[f"curv_{hemi}"])
            curv_sign_map = (np.sign(curv_map) + 1) / 4 + 0.25
            bg_map = curv_sign_map
        else:
            sulc_map = surf_mesh[f"sulc_{hemi}"]
            bg_map = sulc_map

        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)

        # Starting from this line until _plot_surf included is actually
        # plot_surf_stat_map, but to avoid cyclic import
        # the code is duplicated here
        stat_map_iter = texture[hemi]
        surf_mesh_iter = surf[hemi]

        stat_map_iter, surf_mesh_iter, bg_map = check_surface_plotting_inputs(
            stat_map_iter,
            surf_mesh_iter,
            hemi,
            bg_map,
            map_var_name="img_on_surf",
        )
        loaded_stat_map = load_surf_data(stat_map_iter)

        # derive symmetric vmin, vmax and colorbar limits depending on
        # symmetric_cbar settings
        cbar_vmin, cbar_vmax, vmin, vmax = _adjust_colorbar_and_data_ranges(
            loaded_stat_map,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )
        _plot_surf(
            surf_mesh=surf_mesh_iter,
            surf_map=loaded_stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=mode,
            cmap=cmap,
            colorbar=False,  # Colorbar created externally.
            threshold=threshold,
            bg_on_data=bg_on_data,
            vmin=vmin,
            vmax=vmax,
            axes=ax,
            **kwargs,
        )

        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too
        # small.
        ax.set_box_aspect(None, zoom=1.3)

    if colorbar:
        sm = _colorbar_from_array(
            get_data(stat_map),
            vmin,
            vmax,
            threshold,
            symmetric_cbar=symmetric_cbar,
            cmap=plt.get_cmap(cmap),
        )

        cbar_grid = GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        # Get custom ticks to set in colorbar
        ticks = _get_ticks(vmin, vmax, cbar_tick_format, threshold)
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
