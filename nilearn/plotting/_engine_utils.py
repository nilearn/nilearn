"""Module for utility functions importing from `matplotlib` and used by
multiple modules in nilearn.plotting package.
"""

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_threshold
from nilearn.plotting._utils import get_colorbar_and_data_ranges


def adjust_cmap(cmap, vmin, vmax, threshold):
    """Normalize and adjust the specified colormap according to specified vmin,
    vmax, threshold values.

    Parameters
    ----------
    %(cmap)s
    vmin : :obj:`float`  or obj:`int`
        Should not be None
    vmax : :obj:`float`  or obj:`int`
        Should not be None
    threshold : :obj:`float`  or obj:`int`
        Should be non-negative
    """
    our_cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]

    if threshold is not None:
        # set colors to gray for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)

    our_cmap = LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, our_cmap.N
    )
    return our_cmap, norm


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
    our_cmap, norm = adjust_cmap(cmap, vmin, vmax, threshold)

    x = np.linspace(0, 1, 100)
    rgb = our_cmap(x, bytes=True)[:, :3]
    rgb = np.array(rgb, dtype=int)
    colors = [
        [np.round(i, 3), f"rgb({col[0]}, {col[1]}, {col[2]})"]
        for i, col in zip(x, rgb)
    ]
    return {
        "colors": colors,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": our_cmap,
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
