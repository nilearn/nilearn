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


def colorscale(
    cmap, values, threshold=None, symmetric_cmap=True, vmax=None, vmin=None
):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    cmap = plt.get_cmap(cmap)
    abs_values = np.abs(values)

    if (
        symmetric_cmap
        and vmin is not None
        and vmax is not None
        and vmin != -vmax
    ):
        warn(
            f"Specified {vmin=} and {vmax=} values do not create a symmetric"
            " colorbar. The values will be modified to be symmetric.",
            stacklevel=find_stack_level(),
        )
    if vmax is None:
        vmax = abs_values.max()
    if vmin is None:
        vmin = values.min()
    # cast to float to avoid TypeError if vmax/vmin is a numpy boolean
    vmax = float(vmax)
    vmin = float(vmin)

    if symmetric_cmap:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    abs_threshold = None
    if threshold is not None:
        abs_threshold = check_threshold(threshold, values, fast_abs_percentile)
        istart = int(norm(-abs_threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(abs_threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)  # just an average gray color
    our_cmap = LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )
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
        "abs_threshold": abs_threshold,
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
