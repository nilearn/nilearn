from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numbers
import numpy as np


def save_figure_if_needed(fig, output_file):
    """Save figure if an output file value is given.

    Create output path if required.

    Parameters
    ----------
    fig: figure, axes, or display instance

    output_file: str, Path or None

    Returns
    -------
    None if ``output_file`` is None, ``fig`` otherwise.
    """
    # avoid circular import
    from nilearn.plotting.displays import BaseSlicer

    if output_file is None:
        return fig

    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    if not isinstance(fig, (plt.Figure, BaseSlicer)):
        fig = fig.figure

    fig.savefig(output_file)
    if isinstance(fig, plt.Figure):
        plt.close(fig)
    else:
        fig.close()

    return None


def check_threshold_not_negative(threshold):
    """Make sure threshold is non negative number."""
    if isinstance(threshold, (int, float)) and threshold < 0:
        raise ValueError("Threshold should be a non-negative number!")


def get_colorbar_and_data_ranges(
    stat_map_data,
    vmin=None,
    vmax=None,
    symmetric_cbar=True,
    force_min_stat_map_value=None,
):
    """Set colormap and colorbar limits.

    Used by plot_stat_map, plot_glass_brain and plot_img_on_surf.

    The limits for the colorbar depend on the symmetric_cbar argument. Please
    refer to docstring of plot_stat_map.
    """
    # handle invalid vmin/vmax inputs
    if (not isinstance(vmin, numbers.Number)) or (not np.isfinite(vmin)):
        vmin = None
    if (not isinstance(vmax, numbers.Number)) or (not np.isfinite(vmax)):
        vmax = None

    # avoid dealing with masked_array:
    if hasattr(stat_map_data, "_mask"):
        stat_map_data = np.asarray(
            stat_map_data[np.logical_not(stat_map_data._mask)]
        )

    if force_min_stat_map_value is None:
        stat_map_min = np.nanmin(stat_map_data)
    else:
        stat_map_min = force_min_stat_map_value
    stat_map_max = np.nanmax(stat_map_data)

    if symmetric_cbar == "auto":
        if vmin is None or vmax is None:
            min_value = (
                stat_map_min if vmin is None else max(vmin, stat_map_min)
            )
            max_value = (
                stat_map_max if vmax is None else min(stat_map_max, vmax)
            )
            symmetric_cbar = min_value < 0 < max_value
        else:
            symmetric_cbar = np.isclose(vmin, -vmax)

    # check compatibility between vmin, vmax and symmetric_cbar
    if symmetric_cbar:
        if vmin is None and vmax is None:
            vmax = max(-stat_map_min, stat_map_max)
            vmin = -vmax
        elif vmin is None:
            vmin = -vmax
        elif vmax is None:
            vmax = -vmin
        elif not np.isclose(vmin, -vmax):
            raise ValueError(
                "vmin must be equal to -vmax unless symmetric_cbar is False."
            )
        cbar_vmin = vmin
        cbar_vmax = vmax
    # set colorbar limits
    else:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            cbar_vmin = 0 if vmin is None else vmin
            cbar_vmax = vmax
        elif negative_range:
            cbar_vmax = 0 if vmax is None else vmax
            cbar_vmin = vmin
        else:
            # limit colorbar to plotted values
            cbar_vmin = vmin
            cbar_vmax = vmax

    # set vmin/vmax based on data if they are not already set
    if vmin is None:
        vmin = stat_map_min
    if vmax is None:
        vmax = stat_map_max

    return cbar_vmin, cbar_vmax, float(vmin), float(vmax)


def to_color_strings(colors):
    """Return a list of colors as hex strings."""
    cmap = mpl.colors.ListedColormap(colors)
    colors = cmap(np.arange(cmap.N))[:, :3]
    colors = np.asarray(colors * 255, dtype="uint8")
    colors = [
        f"#{int(row[0]):02x}{int(row[1]):02x}{int(row[2]):02x}"
        for row in colors
    ]
    return colors
