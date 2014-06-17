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


def fast_abs_percentile(map, percentile=80):
    """ A fast version of the percentile of the absolute value.
    """
    if hasattr(map, 'mask'):
        # Catter for masked arrays
        map = np.asarray(map[np.logical_not(map.mask)])
    map = np.abs(map)
    map = map.ravel()
    index = int(map.size * .01 * percentile)
    if partition is not None:
        # Partial sort: faster than sort
        return partition(map, index)[index + 1]
    return map.sort()[index]



