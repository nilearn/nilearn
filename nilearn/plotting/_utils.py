from numbers import Number
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nilearn._utils.logger import find_stack_level

DEFAULT_ENGINE = "matplotlib"


def engine_warning(engine):
    message = (
        f"'{engine}' is not installed. To be able to use '{engine}' as "
        "plotting engine for 'nilearn.plotting' package:\n"
        " pip install 'nilearn[plotting]'"
    )
    warn(message, stacklevel=find_stack_level())


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


def get_cbar_ticks(vmin, vmax, offset, n_ticks=5):
    """Help for BaseSlicer."""
    # edge case where the data has a single value yields
    # a cryptic matplotlib error message when trying to plot the color bar
    if vmin == vmax:
        return np.linspace(vmin, vmax, 1)

    # edge case where the data has all negative values but vmax is exactly 0
    if vmax == 0:
        vmax += np.finfo(np.float32).eps

    # If a threshold is specified, we want two of the tick
    # to correspond to -threshold and +threshold on the colorbar.
    # If the threshold is very small compared to vmax,
    # we use a simple linspace as the result would be very difficult to see.
    ticks = np.linspace(vmin, vmax, n_ticks)
    if offset is not None and offset / vmax > 0.12:
        diff = [abs(abs(tick) - offset) for tick in ticks]
        # Edge case where the thresholds are exactly
        # at the same distance to 4 ticks
        if diff.count(min(diff)) == 4:
            idx_closest = np.sort(np.argpartition(diff, 4)[:4])
            idx_closest = np.isin(ticks, np.sort(ticks[idx_closest])[1:3])
        else:
            # Find the closest 2 ticks
            idx_closest = np.sort(np.argpartition(diff, 2)[:2])
            if 0 in ticks[idx_closest]:
                idx_closest = np.sort(np.argpartition(diff, 3)[:3])
                idx_closest = idx_closest[[0, 2]]
        ticks[idx_closest] = [-offset, offset]
    if len(ticks) > 0 and ticks[0] < vmin:
        ticks[0] = vmin

    return ticks


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
    if (not isinstance(vmin, Number)) or (not np.isfinite(vmin)):
        vmin = None
    if (not isinstance(vmax, Number)) or (not np.isfinite(vmax)):
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


def check_threshold_not_negative(threshold):
    """Make sure threshold is non negative number.

    If threshold == "auto", it may be set to very small value.
    So we allow for that.
    """
    if isinstance(threshold, (int, float)) and threshold < -1e-5:
        raise ValueError("Threshold should be a non-negative number!")


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
