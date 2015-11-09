"""
Extended math utilities
"""
# Author: Gael Varoquaux
# License: BSD

import numbers
import numpy as np

from .compat import _basestring

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


def is_spd(M, decimal=15, verbose=1):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ==========
    M: numpy.ndarray
        symmetric positive definite matrix.

    verbose: int, optional
        verbosity level (0 means no message)

    Returns
    =======
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


def check_threshold(threshold, data, percentile_calculate, name):
    """ Checks if the given threshold is in correct format

    Parameters
    ----------
    threshold: a real value or a percentage in string.
        if threshold is a percentage expressed in a string
        it must finish with a percent sign like "99.7%".
    data: ndarray
        an array of the input masked data
    percentile_calculate: a percentile function
        define the name of a specific percentile function
        to calculate the score on the data.

    Returns
    -------
    threshold: a number
        returns the score of the percentile on the data or
        returns threshold as it is if given threshold is not
        a percentile.
    """
    if isinstance(threshold, _basestring):
        message = ('If "{0}" is given as string it '
                   'should be a number followed by the percent '
                   'sign, e.g. "25.3%"').format(name)
        if not threshold.endswith('%'):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message, )
            raise

        threshold = percentile_calculate(data, percentile)

    elif not isinstance(threshold, numbers.Real):
        raise TypeError('%s should be either a number '
                        'or a string finishing with a percent sign' % (name, ))
    return threshold
