import contextlib
from itertools import pairwise
from numbers import Number
from typing import Literal
from warnings import warn

import numpy as np

from nilearn._utils.logger import find_stack_level
from nilearn._utils.versions import (
    OPTIONAL_MATPLOTLIB_MIN_VERSION,
    compare_version,
)

DEFAULT_ENGINE: Literal["matplotlib"] = "matplotlib"
DEFAULT_TICK_FORMAT = "%.2g"


def set_mpl_backend() -> None:
    """Check if matplotlib is installed.

    If installed, check if the installed version complies with the minimum
    supported matplotlib version. If it does not, raise error; otherwise set
    the matplotlib backend.

    If current backend is not usable, switch to default "Agg" backend.
    """
    with contextlib.suppress(ImportError):
        import matplotlib

    # When matplotlib was successfully imported we need to check
    # that the version is greater that the minimum required one
    mpl_version = getattr(matplotlib, "__version__", "0.0.0")
    if not compare_version(mpl_version, ">=", OPTIONAL_MATPLOTLIB_MIN_VERSION):
        raise ImportError(
            f"A matplotlib version of at least "
            f"{OPTIONAL_MATPLOTLIB_MIN_VERSION} "
            f"is required to use nilearn. {mpl_version} was found. "
            f"Please upgrade matplotlib."
        )
    current_backend = matplotlib.get_backend().lower()

    try:
        # Making sure the current backend is usable by matplotlib
        matplotlib.use(current_backend)
    except Exception:
        # If not, switching to default agg backend
        matplotlib.use("Agg")
    new_backend = matplotlib.get_backend().lower()

    if new_backend != current_backend:
        # Matplotlib backend has been changed, let's warn the user
        warn(
            f"Backend changed to {new_backend}...",
            stacklevel=find_stack_level(),
        )


def engine_warning(engine: str) -> None:
    option = "plotting"
    if engine == "plotly":
        option = "matplolib, plotly"
    message = (
        f"'{engine}' is not installed. To be able to use '{engine}' as "
        "plotting engine for 'nilearn.plotting' package:\n"
        f" pip install 'nilearn[{option}]'"
    )
    warn(message, stacklevel=find_stack_level())


def get_cbar_bounds(vmin, vmax, num_val, tick_format=DEFAULT_TICK_FORMAT):
    """Return colorbar boundaries which include vmin and vmax values when
    formatted with ``tick_format``.
    """
    # Formatting the vmin and vmax values with tick_format is
    # necessary. Because get_cbar_ticks returns formatted values. When values
    # are formatted, they are actually rounded depending on tick_format. If the
    # rounded value is bigger than vmax, or smaller than vmin, these values are
    # omitted in the display.
    bounds = np.linspace(
        float(tick_format % vmin), float(tick_format % vmax), num_val
    )

    # if all bound values are 0, return None
    if np.all(bounds == 0):
        bounds = None

    return bounds


def _resolve_zero_conflict(a, b, threshold, vmin, vmax):
    """Resolve which tick to remove when 0 is close to a threshold or bound."""
    if 0 not in (a, b):
        return None
    is_threshold_pair = (
        threshold is not None
        and (a in (-threshold, threshold) or b in (-threshold, threshold))
    )
    if is_threshold_pair and threshold <= 1e-5:
        return a if a != 0 else b
    if vmin != 0 and vmax != 0:
        return 0
    return None


def _find_removable_tick(a, b, threshold, vmin, vmax, keep_list):
    """Pick which tick of a close pair (a, b) should be removed."""
    if a not in keep_list:
        return a
    if b not in keep_list:
        return b
    return _resolve_zero_conflict(a, b, threshold, vmin, vmax)


def _remove_close_values(ticks, step_size, threshold, vmin, vmax):
    """Remove some tick values if they are very close to each other."""
    # create the list for the values to be kept in tick list
    keep_list = [vmin, 0, vmax]
    if threshold is not None:
        keep_list.extend([-threshold, threshold])

    for a, b in pairwise(ticks):
        if b - a >= step_size / 3:
            continue
        value_to_remove = _find_removable_tick(
            a, b, threshold, vmin, vmax, keep_list
        )
        if value_to_remove is not None:
            ticks = np.delete(ticks, np.where(ticks == value_to_remove))
    return ticks


def _adjust_integer_ticks(n_ticks, threshold, f_vmin, f_vmax, tick_format):
    """Validate and adjust tick count when integer formatting is requested."""
    if tick_format != "%i":
        return n_ticks
    if threshold is not None and int(threshold) != threshold:
        warn(
            "You provided a non integer threshold "
            "but configured the colorbar to use integer formatting.",
            stacklevel=find_stack_level(),
        )
    if f_vmax - f_vmin < n_ticks - 1:
        return int(f_vmax - f_vmin + 1)
    return n_ticks


def _filter_ticks_with_threshold(
    ticks, threshold, vmin, vmax, f_vmin, f_vmax, tick_format
):
    """Format and filter colorbar ticks around the threshold value."""
    if threshold is None or threshold <= 1e-6:
        return ticks
    threshold = float(tick_format % threshold)
    if 0 <= vmin <= vmax or vmin <= vmax <= 0:
        if vmax <= 0:
            threshold = -threshold
        ticks = np.append(ticks, threshold)
    else:
        ticks = np.append(ticks, [-threshold, threshold])

    valid_mask = (
        (ticks > abs(threshold))
        | (ticks < -abs(threshold))
        | np.isin(ticks, [0, f_vmin, f_vmax, threshold, -threshold])
    )
    return ticks[np.where(valid_mask)]


def get_cbar_ticks(
    vmin, vmax, threshold=None, n_ticks=5, tick_format=DEFAULT_TICK_FORMAT
):
    """Return an array of evenly spaced ``n_ticks`` tick values to be used for
    the colorbar.

    The final tick list will contain 0, vmin, vmax and threshold values. If
    there are values very close to each other, it will favor threshold value
    over 0.

    Parameters
    ----------
    vmin : :obj:`float`
        minimum value for the colorbar, should not be None
    vmax : :obj:`float`
        maximum value for the colorbar, should not be None
    threshold : :obj:`float`, :obj:`int` or None
        threshold value
    n_ticks : :obj:`int`
        number of tick values to return
    tick_format : :obj:`str`, default="%.2g"
        formatting to be used for colorbar ticks

    Returns
    -------
    :class:`~numpy.ndarray`
        An array with ``n_ticks`` elements if ``vmin`` != ``vmax``,
        else array with one element.
    """
    f_vmin = float(tick_format % vmin)
    f_vmax = float(tick_format % vmax)
    if f_vmin == f_vmax and (
        threshold is None or threshold == 0 or f_vmax == 0
    ):
        return np.linspace(f_vmin, f_vmax, 1)

    n_ticks = _adjust_integer_ticks(
        n_ticks, threshold, f_vmin, f_vmax, tick_format
    )
    ticks = np.linspace(vmin, vmax, n_ticks)
    # format tick values as matplotlib will display them
    # this is to avoid double appearance of same tick value
    # for example when threshold is 9.96 and vmax is 10, matplotlib rounds
    # 9.96 to 10. If both 9.96 and 10 are in the tick list, matplotlib will
    # display double 10 in the colorbar.
    ticks = np.vectorize(lambda x: float(tick_format % x))(ticks)
    # get the size of maximum interval between the ticks
    step_size = max(b - a for a, b in pairwise(ticks))

    ticks = _filter_ticks_with_threshold(
        ticks, threshold, vmin, vmax, f_vmin, f_vmax, tick_format
    )

    # we do this here to include the case where threshold is None
    if vmin < 0 < vmax:
        ticks = np.append(ticks, 0)

    ticks = np.sort(np.unique(ticks))
    abs_threshold = abs(threshold) if threshold is not None else None
    ticks = _remove_close_values(
        ticks, step_size, abs_threshold, f_vmin, f_vmax
    )

    return ticks


def _auto_detect_symmetric_cbar(vmin, vmax, data_min, data_max):
    """Determine whether colorbar is symmetric when symmetric_cbar='auto'."""
    if vmin is None or vmax is None:
        min_value = data_min if vmin is None else max(vmin, data_min)
        max_value = data_max if vmax is None else min(data_max, vmax)
        return min_value < 0 < max_value
    return np.isclose(vmin, -vmax)


def _compute_symmetric_colorbar_limits(vmin, vmax, data_min, data_max):
    """Compute symmetric colorbar limits and warn on mismatches."""
    if vmin is None and vmax is None:
        vmax = max(-data_min, data_max)
        vmin = -vmax
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin
    elif vmin != -vmax:
        warn(
            f"Specified {vmin=} and {vmax=} values do not create a "
            "symmetric colorbar. The values will be modified to be "
            "symmetric.",
            stacklevel=find_stack_level(),
        )
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax, vmin, vmax


def _compute_asymmetric_colorbar_limits(vmin, vmax, data_min, data_max):
    """Compute asymmetric colorbar limits based on data ranges."""
    if data_min >= 0:
        cbar_vmin = 0 if vmin is None else vmin
        cbar_vmax = vmax
    elif data_max <= 0:
        cbar_vmax = 0 if vmax is None else vmax
        cbar_vmin = vmin
    else:
        cbar_vmin = vmin
        cbar_vmax = vmax
    return cbar_vmin, cbar_vmax


def get_colorbar_and_data_ranges(
    data,
    vmin=None,
    vmax=None,
    symmetric_cbar=True,
    force_min_stat_map_value=None,
):
    """Set colormap and colorbar limits.

    The limits for the colorbar depend on the symmetric_cbar argument.

    Parameters
    ----------
    data : :class:`np.ndarray`
        The data

    vmin : :obj:`float`, default=None
        min value for data to consider

    vmax : :obj:`float`, default=None
        max value for data to consider

    symmetric_cbar : :obj:`bool`, default=True
        Whether to use a symmetric colorbar

    force_min_stat_map_value : :obj:`int`, default=None
        The value to force as minimum value for the colorbar
    """
    # handle invalid vmin/vmax inputs
    if (not isinstance(vmin, Number)) or (not np.isfinite(vmin)):
        vmin = None
    if (not isinstance(vmax, Number)) or (not np.isfinite(vmax)):
        vmax = None

    # avoid dealing with masked_array:
    if hasattr(data, "_mask"):
        data = np.asarray(data[np.logical_not(data._mask)])

    if force_min_stat_map_value is None:
        data_min = np.nanmin(data)
    else:
        data_min = force_min_stat_map_value
    data_max = np.nanmax(data)

    if symmetric_cbar == "auto":
        symmetric_cbar = _auto_detect_symmetric_cbar(
            vmin, vmax, data_min, data_max
        )

    # check compatibility between vmin, vmax and symmetric_cbar
    if symmetric_cbar:
        cbar_vmin, cbar_vmax, vmin, vmax = _compute_symmetric_colorbar_limits(
            vmin, vmax, data_min, data_max
        )
    else:
        cbar_vmin, cbar_vmax = _compute_asymmetric_colorbar_limits(
            vmin, vmax, data_min, data_max
        )

    # set vmin/vmax based on data if they are not already set
    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max

    return cbar_vmin, cbar_vmax, float(vmin), float(vmax)


def check_threshold_not_negative(threshold) -> None:
    """Make sure threshold is non negative number.

    If threshold == "auto", it may be set to very small value.
    So we allow for that.
    """
    if isinstance(threshold, (int, float)) and threshold < -1e-5:
        raise ValueError("Threshold should be a non-negative number!")
