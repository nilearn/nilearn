""" Misc utilities for the library

Authors: Bertrand Thirion, Matthew Brett, 2015
"""
import sys
import scipy.linalg as spl
import numpy as np

py3 = sys.version_info[0] >= 3

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
        return int(not np.all(M==0))
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
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


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
    return np.where(X<=0, 0, 1. / X)


def open4csv(fname, mode):
    """ Open filename `fname` for CSV IO in read or write `mode`

    Parameters
    ----------
    fname : str
        filename to open
    mode : {'r', 'w'}
        Mode to open file.  Don't specify binary or text modes; we need to
        chose these according to python version.

    Returns
    -------
    fobj : file object
        open file object; needs to be closed by the caller
    """
    if mode not in ('r', 'w'):
        raise ValueError('Only "r" and "w" allowed for mode')
    if not py3: # Files for csv reading and writing should be binary mode
        return open(fname, mode + 'b')
    return open(fname, mode, newline='')
