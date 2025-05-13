"""Functions specific to "matplotlib" backend for surface visualization
functions in :obj:`~nilearn.plotting.surface.surf_plotting`.

Any imports from "matplotlib" package, or "matplotlib" engine specific utility
functions in :obj:`~nilearn.plotting.surface` should be in this file.
"""

from warnings import warn

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn.plotting._utils import (
    get_cbar_ticks,
    get_colorbar_and_data_ranges,
    save_figure_if_needed,
)
from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.surface._backend import (
    BaseSurfaceBackend,
    check_hemispheres,
    check_surf_map,
    check_views,
)
from nilearn.surface import load_surf_data

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import make_axes
    from matplotlib.colors import LinearSegmentedColormap, Normalize
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


def _get_view_plot_surf(hemi, view):
    """Help function for plot_surf with matplotlib engine.

    This function checks the selected hemisphere and view, and
    returns elev and azim.
    """
    check_views([view])
    check_hemispheres([hemi])
    if isinstance(view, str):
        if hemi == "both" and view in ["lateral", "medial"]:
            raise ValueError(
                "Invalid view definition: when hemi is 'both', "
                "view cannot be 'lateral' or 'medial'.\n"
                "Maybe you meant 'left' or 'right'?"
            )
        return MATPLOTLIB_VIEWS[hemi][view]
    return view


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


class MatplotlibSurfaceBackend(BaseSurfaceBackend):
    @property
    def name(self):
        return "matplotlib"

    def _plot_surf(
        self,
        coords,
        faces,
        surf_map=None,
        bg_map=None,
        hemi="left",
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
        parameters_not_implemented_in_matplotlib = {
            "symmetric_cmap": symmetric_cmap,
            "title_font_size": title_font_size,
        }

        self._check_engine_params(parameters_not_implemented_in_matplotlib)

        # adjust values
        avg_method = "mean" if avg_method is None else avg_method
        alpha = "auto" if alpha is None else alpha
        cbar_tick_format = (
            "%.2g" if cbar_tick_format == "auto" else cbar_tick_format
        )
        # Leave space for colorbar
        figsize = [4.7, 5] if colorbar else [4, 5]

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
