"""Functions specific to "niivue" backend for surface visualization."""

import numpy as np

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import check_threshold


def colorscale_niivue(values, vmax, threshold=None):
    """Normalize a cmap, put it in plotly format, get threshold and range."""
    if vmax is None:
        abs_values = np.abs(values)
        vmax = abs_values.max()
    vmax = float(vmax)

    if threshold is not None:
        threshold = check_threshold(threshold, values, fast_abs_percentile)
        return vmax, float(threshold)

    return vmax, threshold
