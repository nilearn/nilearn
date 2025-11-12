from itertools import pairwise
from numbers import Number
from warnings import warn

import numpy as np

from nilearn._utils.logger import find_stack_level

DEFAULT_ENGINE = "matplotlib"
DEFAULT_TICK_FORMAT = "%.2g"


def engine_warning(engine):
    message = (
        f"'{engine}' is not installed. To be able to use '{engine}' as "
        "plotting engine for 'nilearn.plotting' package:\n"
        " pip install 'nilearn[plotting]'"
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


def _remove_close_values(ticks, step_size, threshold, vmin, vmax):
    """Remove some tick values if they are very close to each other."""
    # create the list for the values to be kept in tick list
    keep_list = [vmin, 0, vmax]
    if threshold is not None:
        keep_list.extend([-threshold, threshold])

    for a, b in pairwise(ticks):
        # if two consecutive values have distance less the 1/3th of step_size
        # check for the possibility to remove one
        if b - a < step_size / 3:
            value_to_remove = a if a not in keep_list else None
            if value_to_remove is None:
                value_to_remove = b if b not in keep_list else None

            # if one of the ticks is 0 and the other is threshold
            if (
                value_to_remove is None
                and threshold is not None
                and (
                    a in [-threshold, threshold]
                    or b in [-threshold, threshold]
                )
                and (a == 0 or b == 0)
            ):
                # if threshold is very close to 0, remove it and keep 0
                if threshold <= 1e-5:
                    value_to_remove = a if a != 0 else b
                    # otherwise remove 0 if it is not vmin or vmax
                elif vmin != 0 and vmax != 0:
                    value_to_remove = 0
            if value_to_remove is not None:
                ticks = np.delete(ticks, np.where(ticks == value_to_remove))
    return ticks


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
    vmin: :obj:`float`
        minimum value for the colorbar, should not be None
    vmax: :obj:`float`
        maximum value for the colorbar, should not be None
    threshold: :obj:`float`, :obj:`int` or None
        threshold value
    n_ticks: :obj:`int`
        number of tick values to return
    tick_format: :obj:`str`, default="%.2g"
        formatting to be used for colorbar ticks

    Returns
    -------
    :class:`~numpy.ndarray`
        an array with ``n_ticks`` elements if ``vmin`` != ``vmax``, else array
        with one element.
    """
    f_vmin = float(tick_format % vmin)
    f_vmax = float(tick_format % vmax)
    if f_vmin == f_vmax and (
        threshold is None or threshold == 0 or f_vmax == 0
    ):
        return np.linspace(f_vmin, f_vmax, 1)

    if tick_format == "%i":
        if threshold is not None and int(threshold) != threshold:
            warn(
                "You provided a non integer threshold "
                "but configured the colorbar to use integer formatting.",
                stacklevel=find_stack_level(),
            )
        if f_vmax - f_vmin < n_ticks - 1:
            n_ticks = int(f_vmax - f_vmin + 1)

    ticks = np.linspace(vmin, vmax, n_ticks)
    # format tick values as matplotlib will display them
    # this is to avoid double appearance of same tick value
    # for example when threshold is 9.96 and vmax is 10, matplotlib rounds
    # 9.96 to 10. If both 9.96 and 10 are in the tick list, matplotlib will
    # display double 10 in the colorbar.
    ticks = np.vectorize(lambda x: float(tick_format % x))(ticks)
    # get the size of maximum interval between the ticks
    step_size = max(b - a for a, b in pairwise(ticks))

    if threshold is not None and threshold > 1e-6:
        # set threshold to formatted threshold
        threshold = float(tick_format % threshold)

        # if the values are either positive or negative
        if 0 <= vmin <= vmax or vmin <= vmax <= 0:
            if vmax <= 0:
                threshold = -threshold
            ticks = np.append(ticks, threshold)
        else:
            ticks = np.append(ticks, [-threshold, threshold])

        # remove unnecessary ticks that would be between 0 and +-threshold
        ticks = ticks[
            np.where(
                # normally threshold should be positive
                # however in the above condition, if data is either positive or
                # negative
                # we set threshold=-threshold
                # in that case below we need to check with abs(threshold)
                (ticks > abs(threshold))
                | (ticks < -abs(threshold))
                | (np.isin(ticks, [0, f_vmin, f_vmax, threshold, -threshold]))
            )
        ]

    # we do this here to include the case where threshold is None
    if vmin < 0 < vmax:
        ticks = np.append(ticks, 0)

    ticks = np.sort(np.unique(ticks))
    abs_threshold = abs(threshold) if threshold is not None else None
    ticks = _remove_close_values(
        ticks, step_size, abs_threshold, f_vmin, f_vmax
    )

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
        elif vmin != -vmax:
            warn(
                f"Specified {vmin=} and {vmax=} values do not create a "
                "symmetric colorbar. The values will be modified to be "
                "symmetric.",
                stacklevel=find_stack_level(),
            )
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax

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
