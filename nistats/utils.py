""" Misc utilities for the library

Authors: Bertrand Thirion, Matthew Brett, 2015
"""
import sys
import scipy.linalg as spl
import numpy as np
from scipy.stats import norm
from warnings import warn
import pandas as pd

py3 = sys.version_info[0] >= 3


def _check_list_length_match(list_1, list_2, var_name_1, var_name_2):
    """Check length match of two given lists to raise error if necessary"""
    if len(list_1) != len(list_2):
        raise ValueError(
            'len(%s) %d does not match len(%s) %d'
            % (str(var_name_1), len(list_1), str(var_name_2), len(list_2)))


def _check_and_load_tables(tables_, var_name):
    """Check tables can be loaded in DataFrame to raise error if necessary"""
    tables = []
    for table_idx, table in enumerate(tables_):
        if isinstance(table, _basestring):
            try:
                loaded = pd.read_csv(table, index_col=0)
            except:
                raise ValueError('table path %s could not be loaded' % table)
            tables.append(loaded)
        elif isinstance(table, pd.DataFrame):
            tables.append(table)
        else:
            raise TypeError('%s can only be a pandas DataFrames or a'
                            'string. A %s was provided at idx %d' %
                            (var_name, type(table), table_idx))
    return tables


def _check_run_tables(run_imgs, tables_, tables_name):
    """Check fMRI runs and corresponding tables to raise error if necessary"""
    if isinstance(tables_, (_basestring, pd.DataFrame)):
        tables_ = [tables_]
    _check_list_length_match(run_imgs, tables_, 'run_imgs', tables_name)
    tables_ = _check_and_load_tables(tables_, tables_name)
    return tables_


def z_score(pvalue):
    """ Return the z-score corresponding to a given p-value.
    """
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return norm.isf(pvalue)


def multiple_fast_inv(a):
    """Compute the inverse of a set of arrays.

    Parameters
    ----------
    a: array_like of shape (n_samples, n_dim, n_dim)
        Set of square matrices to be inverted. A is changed in place.

    Returns
    -------
    a: ndarray
       yielding the inverse of the inputs

    Raises
    ------
    LinAlgError :
        If `a` is singular.
    ValueError :
        If `a` is not square, or not 2-dimensional.

    Notes
    -----
    This function is borrowed from scipy.linalg.inv,
    but with some customizations for speed-up.
    """
    if a.shape[1] != a.shape[2]:
        raise ValueError('a must have shape (n_samples, n_dim, n_dim)')
    from scipy.linalg import calc_lwork
    from scipy.linalg.lapack import get_lapack_funcs
    a1, n = a[0], a.shape[0]
    getrf, getri = get_lapack_funcs(('getrf', 'getri'), (a1,))
    for i in range(n):
        if (getrf.module_name[:7] == 'clapack'
            and getri.module_name[:7] != 'clapack'):
            # ATLAS 3.2.1 has getrf but not getri.
            lu, piv, info = getrf(np.transpose(a[i]), rowmajor=0,
                                  overwrite_a=True)
            a[i] = np.transpose(lu)
        else:
            a[i], piv, info = getrf(a[i], overwrite_a=True)
        if info == 0:
            if getri.module_name[:7] == 'flapack':
                lwork = calc_lwork.getri(getri.prefix, a1.shape[0])
                lwork = lwork[1]
                # XXX: the following line fixes curious SEGFAULT when
                # benchmarking 500x500 matrix inverse. This seems to
                # be a bug in LAPACK ?getri routine because if lwork is
                # minimal (when using lwork[0] instead of lwork[1]) then
                # all tests pass. Further investigation is required if
                # more such SEGFAULTs occur.
                lwork = int(1.01 * lwork)
                a[i], _ = getri(a[i], piv, lwork=lwork, overwrite_lu=1)
            else:  # clapack
                a[i], _ = getri(a[i], piv, overwrite_lu=1)
        else:
            raise ValueError('Matrix LU decomposition failed')
    return a


def multiple_mahalanobis(effect, covariance):
    """Returns the squared Mahalanobis distance for a given set of samples

    Parameters
    ----------
    effect: array of shape (n_features, n_samples),
        Each column represents a vector to be evaluated

    covariance: array of shape (n_features, n_features, n_samples),
        Corresponding covariance models stacked along the last axis

    Returns
    -------
    sqd: array of shape (n_samples,)
         the squared distances (one per sample)
    """ 
    # check size
    if effect.ndim == 1:
        effect = effect[:, np.newaxis]
    if covariance.ndim == 2:
        covariance = covariance[:, :, np.newaxis]
    if effect.shape[0] != covariance.shape[0]:
        raise ValueError('Inconsistant shape for effect and covariance')
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError('Inconsistant shape for covariance')

    # transpose and make contuguous for the sake of speed
    Xt, Kt = np.ascontiguousarray(effect.T), np.ascontiguousarray(covariance.T)

    # compute the inverse of the covariances
    Kt = multiple_fast_inv(Kt)

    # derive the squared Mahalanobis distances
    sqd = np.sum(np.sum(Xt[:, :, np.newaxis] * Xt[:, np.newaxis] * Kt, 1), 1)
    return sqd


def full_rank(X, cmax=1e15):
    """ Computes the condition number of X and if it is larger than cmax,
    returns a matrix with a condition number smaller than cmax.

    Parameters
    ----------
    X : array of shape (nrows, ncols)
        input array

    cmax : float, optional (default:1.e15),
        tolerance for condition number

    Returns
    -------
    X : array of shape (nrows, ncols)
        output array

    cond : float,
        actual condition number
    """
    U, s, V = spl.svd(X, full_matrices=False)
    smax, smin = s.max(), s.min()
    cond = smax / smin
    if cond < cmax:
        return X, cond

    warn('Matrix is singular at working precision, regularizing...')
    lda = (smax - cmax * smin) / (cmax - 1)
    X = np.dot(U, np.dot(np.diag(s + lda), V))
    return X, cmax


def pos_recipr(X):
    """ Return element-wise reciprocal of array, setting `X`>=0 to 0

    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
       array of same shape as `X`, dtype np.float, with values set to
       1/X where X > 0, 0 otherwise
    """
    X = np.asarray(X)
    return np.where(X <= 0, 0, 1. / X)

_basestring = str if py3 else basestring
