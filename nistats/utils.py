""" Misc utilities for the library

Authors: Bertrand Thirion, Matthew Brett, 2015
"""
import sys
import scipy.linalg as spl
import numpy as np
from scipy.stats import norm

py3 = sys.version_info[0] >= 3


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
        raise ValueError('a must have shape(n_samples, n_dim, n_dim)')
    from scipy.linalg import calc_lwork
    from scipy.linalg.lapack import  get_lapack_funcs
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


def matrix_rank(M, tol=None):
    ''' Return rank of matrix using SVD method

    Rank of the array is the number of SVD singular values of the
    array that are greater than `tol`.

    This version of matrix rank is very similar to the numpy.linalg version
    except for the use of:

    * scipy.linalg.svd istead of numpy.linalg.svd.
    * the MATLAB algorithm for default tolerance calculation

    ``matrix_rank`` appeared in numpy.linalg in December 2009, first available
    in numpy 1.5.0.

    Parameters
    ----------
    M : array-like
        array of <=2 dimensions
    tol : {None, float}
         threshold below which SVD values are considered zero. If `tol`
         is None, and `S` is an array with singular values for `M`, and
         `eps` is the epsilon value for datatype of `S`, then `tol` set
         to ``S.max() * eps * max(M.shape)``.

    Examples
    --------
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.zeros((4,4))) # All zeros - zero rank
    0
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    >>> matrix_rank([1]) # accepts array-like
    1

    Notes
    -----
    We check for numerical rank deficiency by using ``tol=max(M.shape) * eps *
    S[0]`` (where ``S[0]`` is the maximum singular value and thus the 2-norm of
    the matrix). This is one tolerance threshold for rank deficiency, and the
    default algorithm used by MATLAB [#2]_.  When floating point roundoff is the
    main concern, then "numerical rank deficiency" is a reasonable choice. In
    some cases you may prefer other definitions. The most useful measure of the
    tolerance depends on the operations you intend to use on your matrix. For
    example, if your data come from uncertain measurements with uncertainties
    greater than floating point epsilon, choosing a tolerance near that
    uncertainty may be preferable.  The tolerance may be absolute if the
    uncertainties are absolute rather than relative.

    References
    ----------
    .. [#1] G. H. Golub and C. F. Van Loan, _Matrix Computations_.
    Baltimore: Johns Hopkins University Press, 1996.
    .. [#2] http://www.mathworks.com/help/techdoc/ref/rank.html
    '''
    M = np.asarray(M)
    if M.ndim > 2:
        raise TypeError('array should have 2 or fewer dimensions')
    if M.ndim < 2:
        return int(not np.all(M == 0))
    S = spl.svd(M, compute_uv=False)
    if tol is None:
        tol = S.max() * np.finfo(S.dtype).eps * max(M.shape)
    return np.sum(S > tol)


def full_rank(X, r=None):
    """ Return full-rank matrix whose column span is the same as X

    Uses an SVD decomposition.

    If the rank of `X` is known it can be specified by `r` -- no check is made
    to ensure that this really is the rank of X.

    Parameters
    ----------
    X : array-like
        2D array which may not be of full rank.
    r : None or int
        Known rank of `X`.  r=None results in standard matrix rank calculation.
        We do not check `r` is really the rank of X; it is to speed up
        calculations when the rank is already known.

    Returns
    -------
    fX : array
        Full-rank matrix with column span matching that of `X`
    """
    if r is None:
        r = matrix_rank(X)
    V, D, U = spl.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:, order[i]])
    return np.asarray(value).T.astype(np.float64)


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
