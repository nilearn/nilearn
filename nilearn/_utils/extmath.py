"""
Extended math utilities
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
    ----------
    data: ndarray, possibly masked array
        The input data
    percentile: number between 0 and 100
        The percentile that we are asking for

    Returns
    -------
    value: number
        The score at percentile

    Notes
    -----

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
        data = partition(data, index)
    else:
        data.sort()
    return data[index]


def is_spd(M, decimal=15, verbose=1):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ----------
    M: numpy.ndarray
        symmetric positive definite matrix.

    verbose: int, optional
        verbosity level (0 means no message)

    Returns
    -------
    answer: boolean
        True if matrix is symmetric positive definite, False otherwise.
    """
    if not np.allclose(M, M.T, atol=0, rtol=10 ** -decimal):
        if verbose > 0:
            print("matrix not symmetric to %d decimals" % decimal)
        return False
    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0
    if not ispd and verbose > 0:
        print("matrix has a negative eigenvalue: %.3f" % eigvalsh.min())
    return ispd
