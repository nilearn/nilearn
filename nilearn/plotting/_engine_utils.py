"""Module for utility functions importing from `matplotlib` and used by
multiple modules in nilearn.plotting package.
"""

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_threshold
from nilearn.plotting._utils import (
    DEFAULT_TICK_FORMAT,
    check_threshold_not_negative,
    get_cbar_bounds,
    get_cbar_ticks,
    get_colorbar_and_data_ranges,
)


def threshold_cmap(
    cmap, norm, threshold, threshold_color=(0.5, 0.5, 0.5, 1.0)
):
    """Normalize threshold value, and use it to threshold the specified
    colormap.

    Parameters
    ----------
    %(cmap)s

    norm : :class:`mpl.colors.Normalize`
        Norm to be used to normalize threshold

    threshold : :obj:`float`  or obj:`int`
        A positive value to be used as threshold

    threshold_color: :obj:`tuple`, default=(0.5, 0.5, 0.5, 1.0)
        Color to be used for thresholded values. Default value is an average
        gray color.

    Raises
    ------
    ValueError
        If the specified ``threshold`` is negative.
    """
    cmap = plt.get_cmap(cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if threshold is not None:
        check_threshold_not_negative(threshold)
        # set colors to gray for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (cmap.N - 1))

        # update values under threshold to be threshold_color
        for i in range(istart, istop):
            cmaplist[i] = threshold_color

    return LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)


def colorscale(
    cmap, values, threshold=None, symmetric_cmap=True, vmax=None, vmin=None
):
    """Calculate colorbar ranges, adjust and normalize cmap depending on
    specified vmin, vmax, and threshold values. Return the results as dict to
    be used in plotly.
    """
    _, _, vmin, vmax = get_colorbar_and_data_ranges(
        values, vmin, vmax, symmetric_cmap
    )

    if threshold is not None:
        threshold = check_threshold(threshold, values, fast_abs_percentile)

    norm = Normalize(vmin=vmin, vmax=vmax)
    thrs_cmap = threshold_cmap(cmap, norm, threshold)

    x = np.linspace(0, 1, 100)
    rgb = thrs_cmap(x, bytes=True)[:, :3]
    rgb = np.array(rgb, dtype=int)
    colors = [
        [np.round(i, 3), f"rgb({col[0]}, {col[1]}, {col[2]})"]
        for i, col in zip(x, rgb, strict=False)
    ]
    return {
        "colors": colors,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": thrs_cmap,
        "norm": norm,
        "abs_threshold": threshold,
    }


def to_color_strings(colors):
    """Return a list of colors as hex strings."""
    cmap = ListedColormap(colors)
    colors = cmap(np.arange(cmap.N))[:, :3]
    colors = np.asarray(colors * 255, dtype="uint8")
    colors = [
        f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}"
        for row in colors
    ]
    return colors


def create_colormap_from_lut(cmap, default_cmap="gist_ncar"):
    """
    Create a Matplotlib colormap from a DataFrame containing color mappings.

    Parameters
    ----------
    cmap : pd.DataFrame
        DataFrame with columns 'index', 'name', and 'color' (hex values)

    Returns
    -------
    colormap (LinearSegmentedColormap): A Matplotlib colormap
    """
    if "color" not in cmap.columns:
        warn(
            "No 'color' column found in the look-up table. "
            "Will use the default colormap instead.",
            stacklevel=find_stack_level(),
        )
        return default_cmap

    # Ensure colors are properly extracted from DataFrame
    colors = cmap.sort_values(by="index")["color"].tolist()

    # Create a colormap from the list of colors
    return LinearSegmentedColormap.from_list(
        "custom_colormap", colors, N=len(colors)
    )


def create_colorbar_for_fig(
    fig,
    axes,
    cmap,
    norm,
    threshold,
    cbar_vmin=None,
    cbar_vmax=None,
    n_ticks=5,
    tick_format=DEFAULT_TICK_FORMAT,
    spacing="proportional",
    orientation="vertical",
    threshold_color=(0.5, 0.5, 0.5, 1.0),
):
    """Create a colorbar for the specified figure and return."""
    cbar_vmin = cbar_vmin if cbar_vmin is not None else norm.vmin
    cbar_vmax = cbar_vmax if cbar_vmax is not None else norm.vmax

    # in rare cases where plotting an image of zeroes
    # this avoids a matplolib error
    if cbar_vmax == cbar_vmin:
        cbar_vmax += 1
        cbar_vmin += -1

    ticks = get_cbar_ticks(
        cbar_vmin,
        cbar_vmax,
        threshold=threshold,
        n_ticks=n_ticks,
        tick_format=tick_format,
    )
    thrs_cmap = threshold_cmap(cmap, norm, threshold, threshold_color)
    bounds = get_cbar_bounds(cbar_vmin, cbar_vmax, thrs_cmap.N, tick_format)

    mappable = ScalarMappable(norm=norm, cmap=thrs_cmap)
    # if norm is specified, no need to set data
    mappable.set_array([])

    return fig.colorbar(
        mappable,
        cax=axes,
        ticks=ticks,
        boundaries=bounds,
        spacing=spacing,
        orientation=orientation,
        format=tick_format,
    )
