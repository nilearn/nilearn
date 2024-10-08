"""Misc utilities for the library.

Authors: Bertrand Thirion, Matthew Brett, Ana Luisa Pinho, 2020
"""

from warnings import warn

import numpy as np
import scipy.linalg as spl
from scipy.stats import norm


def z_score(pvalue, one_minus_pvalue=None):
    """Return the z-score(s) corresponding to certain p-value(s) and, \
    optionally, one_minus_pvalue(s) provided as inputs.

    Parameters
    ----------
    pvalue : float or 1-d array shape=(n_pvalues,)
        P-values computed using the survival function.

    one_minus_pvalue : float or 1-d array shape=(n_one_minus_pvalues,), \
        optional
        It shall take the value returned
        by /nilearn/glm/contrasts.py::one_minus_pvalue
        which computes the p_value using the cumulative distribution function,
        with n_one_minus_pvalues = n_pvalues.

    Returns
    -------
    z_scores : 1-d array shape=(n_z_scores,), with n_z_scores = n_pvalues

    """
    pvalue = np.clip(pvalue, 1.0e-300, 1.0 - 1.0e-16)
    z_scores_sf = norm.isf(pvalue)

    if one_minus_pvalue is not None:
        one_minus_pvalue = np.clip(one_minus_pvalue, 1.0e-300, 1.0 - 1.0e-16)
        z_scores_cdf = norm.ppf(one_minus_pvalue)
        z_scores = np.empty(pvalue.size)
        use_cdf = z_scores_sf < 0
        use_sf = np.logical_not(use_cdf)
        z_scores[np.atleast_1d(use_cdf)] = z_scores_cdf[use_cdf]
        z_scores[np.atleast_1d(use_sf)] = z_scores_sf[use_sf]
    else:
        z_scores = z_scores_sf
    return z_scores


def multiple_fast_inverse(a):
    """Compute the inverse of a set of arrays.

    Parameters
    ----------
    a : array_like of shape (n_samples, n_dim, n_dim)
        Set of square matrices to be inverted. A is changed in place.

    Returns
    -------
    a : ndarray
       Yielding the inverse of the inputs.

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
        raise ValueError("a must have shape (n_samples, n_dim, n_dim)")
    from scipy.linalg.lapack import get_lapack_funcs

    a1, n = a[0], a.shape[0]
    getrf, getri, getri_lwork = get_lapack_funcs(
        ("getrf", "getri", "getri_lwork"), (a1,)
    )
    for i in range(n):
        if (
            getrf.module_name[:7] == "clapack"
            and getri.module_name[:7] != "clapack"
        ):
            # ATLAS 3.2.1 has getrf but not getri.
            lu, piv, info = getrf(
                np.transpose(a[i]), rowmajor=0, overwrite_a=True
            )
            a[i] = np.transpose(lu)
        else:
            a[i], piv, info = getrf(a[i], overwrite_a=True)
        if info == 0:
            if getri.module_name[:7] == "flapack":
                lwork, _ = getri_lwork(a1.shape[0])
                # XXX: the following line fixes curious SEGFAULT when
                # benchmarking 500x500 matrix inverse. This seems to
                # be a bug in LAPACK ?getri routine because if lwork is
                # minimal (when using lwork[0] instead of lwork[1]) then
                # all tests pass. Further investigation is required if
                # more such SEGFAULTs occur.
                lwork = int(1.01 * lwork.real)
                a[i], _ = getri(a[i], piv, lwork=lwork, overwrite_lu=1)
            else:  # clapack
                a[i], _ = getri(a[i], piv, overwrite_lu=1)
        else:
            raise ValueError("Matrix LU decomposition failed")
    return a


def multiple_mahalanobis(effect, covariance):
    """Return the squared Mahalanobis distance for a given set of samples.

    Parameters
    ----------
    effect : array of shape (n_features, n_samples)
        Each column represents a vector to be evaluated.

    covariance : array of shape (n_features, n_features, n_samples)
        Corresponding covariance models stacked along the last axis.

    Returns
    -------
    sqd : array of shape (n_samples,)
         The squared distances (one per sample).

    """
    # check size
    if effect.ndim == 1:
        effect = effect[:, np.newaxis]
    if covariance.ndim == 2:
        covariance = covariance[:, :, np.newaxis]
    if effect.shape[0] != covariance.shape[0]:
        raise ValueError("Inconsistent shape for effect and covariance")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Inconsistent shape for covariance")

    # transpose and make contuguous for the sake of speed
    Xt, Kt = np.ascontiguousarray(effect.T), np.ascontiguousarray(covariance.T)

    # compute the inverse of the covariances
    Kt = multiple_fast_inverse(Kt)

    # derive the squared Mahalanobis distances
    sqd = np.sum(np.sum(Xt[:, :, np.newaxis] * Xt[:, np.newaxis] * Kt, 1), 1)
    return sqd


def full_rank(X, cmax=1e15):
    """Compute the condition number of X and if it is larger than cmax, \
    returns a matrix with a condition number smaller than cmax.

    Parameters
    ----------
    X : array of shape (nrows, ncols)
        Input array.

    cmax : float, default=1e15
        Tolerance for condition number.

    Returns
    -------
    X : array of shape (nrows, ncols)
        Output array.

    cond : float,
        Actual condition number.

    """
    U, s, V = spl.svd(X, full_matrices=False)
    smax, smin = s.max(), s.min()
    cond = smax / smin
    if cond < cmax:
        return X, cond

    warn("Matrix is singular at working precision, regularizing...")
    lda = (smax - cmax * smin) / (cmax - 1)
    X = np.dot(U, np.dot(np.diag(s + lda), V))
    return X, cmax


def positive_reciprocal(X):
    """Return element-wise reciprocal of array, setting `X`>=0 to 0.

    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
       Array of same shape as `X`, dtype float, with values set to
       1/X where X > 0, 0 otherwise.

    """
    X = np.asarray(X)
    return np.where(X <= 0, 0, 1.0 / X)


def pad_contrast(con_val, theta, stat_type):
    """Pad contrast with zeros if necessary.

    If the contrast is shorter than the number of parameters,
    it is padded with zeros.

    If the contrast is longer than the number of parameters,
    a ValueError is raised.

    Parameters
    ----------
    con_val : numpy.ndarray of shape (p) or (n, p)
        Where p = number of regressors
        with a value explicitly passed by the user.
        p must be <= P,
        where P is the total number of regressors in the design matrix.

    theta : numpy.ndarray with shape (P,m)
        theta of RegressionResults instances
        where P is the total number of regressors in the design matrix.

    stat_type : {'t', 'F'}, optional
        Type of the :term:`contrast`.
    """
    n_cols = con_val.shape[0] if con_val.ndim == 1 else con_val.shape[1]
    if n_cols > theta.shape[0]:
        if stat_type == "t":
            raise ValueError(
                f"t contrasts should be of length P={theta.shape[0]}, "
                f"but it has length {n_cols}."
            )
        if stat_type == "F":
            raise ValueError(
                f"F contrasts should have {theta.shape[0]} columns, "
                f"but it has {n_cols}."
            )

    pad = False
    if n_cols < theta.shape[0]:
        pad = True
        if stat_type == "t":
            warn(
                f"t contrasts should be of length P={theta.shape[0]}, "
                f"but it has length {n_cols}. "
                "The rest of the contrast was padded with zeros.",
                category=UserWarning,
                stacklevel=3,
            )
        if stat_type == "F":
            warn(
                f"F contrasts should have {theta.shape[0]} colmuns, "
                f"but it has only {n_cols}. "
                "The rest of the contrast was padded with zeros.",
                category=UserWarning,
                stacklevel=3,
            )

    if pad:
        if stat_type == "t" or (stat_type == "F" and con_val.shape[0] == 1):
            padding = np.zeros((1, theta.shape[0] - n_cols))
        elif stat_type == "F":
            padding = np.zeros((con_val.shape[0], theta.shape[0] - n_cols))
        con_val = np.hstack((con_val, padding))

    return con_val
