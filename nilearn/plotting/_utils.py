from numbers import Number
from warnings import warn

import numpy as np

from nilearn._utils.logger import find_stack_level

DEFAULT_ENGINE = "matplotlib"


def engine_warning(engine):
    message = (
        f"'{engine}' is not installed. To be able to use '{engine}' as "
        "plotting engine for 'nilearn.plotting' package:\n"
        " pip install 'nilearn[plotting]'"
    )
    warn(message, stacklevel=find_stack_level())


def _add_to_ticks(ticks, threshold):
    ticks = ticks[ticks != 0]
    min_diff = min(abs(abs(ticks) - threshold))
    # check if threshold should be added to the tick list or replaced by a
    # value in the list
    return bool(
        min_diff > abs(ticks[1] - ticks[0]) / 3
        or min_diff >= threshold * 3 / 2
    )


def get_cbar_ticks(vmin, vmax, threshold=None, n_ticks=5):
    """Return an array of evenly spaced ``n_ticks`` tick values to be used for
    the colorbar.

    The tick list will contain vmin, vmax and threshold values. If
    necessary the number of ticks might increase by 2.

    Parameters
    ----------
    vmin: :obj:`float`
        minimum value for the colorbar
    vmax: :obj:`float`
        maximum value for the colorbar
    threshold: :obj:`float`, :obj:`int` or None
        if threshold is not None, ``-threshold`` and ``threshold`` values are
    replaced with the closest tick values. If the space between closest value
    is large, instead of replacing threshold value(s) will be added to the
    list.
    n_ticks: :obj:`int`
        number of tick values to return

    Returns
    -------
    :class:`~numpy.ndarray`
        an array with ``n_ticks`` elements if ``vmin`` != ``vmax``, else array
        with one element.
    """
    if vmin == vmax and (threshold is None or threshold == 0 or vmax == 0):
        return np.linspace(vmin, vmax, 1)

    ticks = np.linspace(vmin, vmax, n_ticks)

    if threshold is not None and threshold != 0:
        diff = abs(abs(ticks) - threshold)
        add = _add_to_ticks(ticks, threshold)

        # if the values are either positive or negative
        if 0 <= vmin <= vmax or vmin <= vmax <= 0:
            threshold = threshold if vmin >= 0 else -threshold
            if add:
                ticks = np.append(ticks, threshold)
            else:
                idx_closest = np.argmin(diff)
                # if the closest value to replace is one of vmin or vmax,
                # instead of replacing add the threshold value to the list
                if ticks[idx_closest] == vmin or ticks[idx_closest] == vmax:
                    ticks = np.append(ticks, threshold)
                # if threshold value is already in the list, do nothing
                elif threshold not in ticks and -threshold not in ticks:
                    ticks[idx_closest] = threshold
        # if vmin is negative and vmax is positive and threshold is in between
        # or outside vmin-vmax values
        elif add:
            ticks = np.append(ticks, [-threshold, threshold])
        else:
            # Edge case where the thresholds are exactly
            # at the same distance to 4 ticks
            if np.count_nonzero(min(diff)) == 4:
                idx_closest = np.sort(np.argpartition(diff, 4)[:4])
                idx_closest = np.isin(ticks, np.sort(ticks[idx_closest])[1:3])
            else:
                # Find the closest 2 ticks
                idx_closest = np.sort(np.argpartition(diff, 2)[:2])
                if 0 in ticks[idx_closest]:
                    idx_closest = np.sort(np.argpartition(diff, 3)[:3])
                    idx_closest = idx_closest[[0, 2]]
            if -threshold not in ticks and -threshold != vmin:
                if ticks[idx_closest[0]] != 0:
                    ticks[idx_closest[0]] = -threshold
                else:
                    ticks = np.append(-threshold)
            if threshold not in ticks and threshold != vmax:
                if ticks[idx_closest[1]] != 0:
                    ticks[idx_closest[1]] = threshold
                else:
                    ticks = np.append(threshold)

        ticks = np.append(ticks, [vmin, vmax])
        ticks = np.sort(np.unique(ticks))
    return ticks


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
        if vmin is None or vmax is None:
            min_value = data_min if vmin is None else max(vmin, data_min)
            max_value = data_max if vmax is None else min(data_max, vmax)
            symmetric_cbar = min_value < 0 < max_value
        else:
            symmetric_cbar = np.isclose(vmin, -vmax)

    # check compatibility between vmin, vmax and symmetric_cbar
    if symmetric_cbar:
        if vmin is None and vmax is None:
            vmax = max(-data_min, data_max)
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
        negative_range = data_max <= 0
        positive_range = data_min >= 0
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
        vmin = data_min
    if vmax is None:
        vmax = data_max

    return cbar_vmin, cbar_vmax, float(vmin), float(vmax)


def check_threshold_not_negative(threshold):
    """Make sure threshold is non negative number.

    If threshold == "auto", it may be set to very small value.
    So we allow for that.
    """
    if isinstance(threshold, (int, float)) and threshold < -1e-5:
        raise ValueError("Threshold should be a non-negative number!")
