"""Extended math utilities."""

# Author: Gael Varoquaux

import numpy as np

from nilearn._utils import logger


def fast_abs_percentile(data, percentile=80):
    """Implement a fast version of the percentile of the absolute value.

    Parameters
    ----------
    data : ndarray, possibly masked array
        The input data.

    percentile : number between 0 and 100
        The percentile that we are asking for.

    Returns
    -------
    value : number
        The score at percentile.

    Notes
    -----
    This is a faster, and less accurate version of
    scipy.stats.scoreatpercentile(np.abs(data), percentile)

    """
    if hasattr(data, "mask"):
        # Catter for masked arrays
        data = np.asarray(data[np.logical_not(data.mask)])
    data = np.abs(data)
    data = data.ravel()
    index = int(data.size * 0.01 * percentile)
    # Partial sort: faster than sort
    data = np.partition(data, index)
    return data[index]


def is_spd(M, decimal=15, verbose=1):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ----------
    M : numpy.ndarray
        Symmetric positive definite matrix.

    decimal : int, default=15
        Decimal.

    verbose : int, default=1
        Verbosity level (0 means no message).

    Returns
    -------
    answer : boolean
        True if matrix is symmetric positive definite, False otherwise.

    """
    if not np.allclose(M, M.T, atol=0, rtol=10**-decimal):
        logger.log(f"matrix not symmetric to {decimal:d} decimals", verbose)
        return False

    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0

    if not ispd:
        logger.log(
            f"matrix has a negative eigenvalue: {eigvalsh.min():.3f}", verbose
        )

    return ispd
