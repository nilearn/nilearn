"""
Fast simple math routines
"""
# Author: Gael Varoquaux
# License: BSD

import numpy as np

try:
    # partition is available only in numpy >= 1.8.0
    from numpy import partition
except ImportError:
    partition = None


def fast_abs_percentile(data, percentile=80):
    """ A fast version of the percentile of the absolute value.

    Parameters
    ==========
    data: ndarray, possibly masked array
        The input data
    percentile: number between 0 and 100
        The percentile that we are asking for

    Returns
    =======
    value: number
        The score at percentile

    Notes
    =====

    This is a faster, and less accurate version of
    scipy.stats.scoreatpercentile(np.abs(data), percentile)
    """
    if hasattr(data, 'mask'):
        # Catter for masked arrays
        data = np.asarray(data[np.logical_not(data.mask)])
    data = np.abs(data)
    data = data.ravel()
    index = int(data.size * .01 * percentile)
    if partition is not None:
        # Partial sort: faster than sort
        return partition(data, index)[index + 1]
    data.sort()
    return data[index + 1]
