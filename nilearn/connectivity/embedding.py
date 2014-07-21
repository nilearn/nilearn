import copy

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance
from manifold import geometric_mean, stack_newaxis, inv, inv_sqrtm, logm


def sym_to_vec(sym, isometry=True):
    """Return the flattened lower triangular part of an array.

    If isometry is True, the off-diagonal terms are multiplied by sqrt(2).
    Acts on the last two dimensions of the array.

    Parameters
    ==========
    sym: numpy.ndarray
        Input array, shape (..., n, n).
    isometry: bool, optional (default to True)
        Off diagonal terms of sym are multiplied by sqrt(2) if True.

    Returns
    =======
    numpy.ndarray
        The output flattened lower triangular part, shape
        (..., n * (n + 1) /2).
    """
    p = sym.shape[-1]
    tril_mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    sym_copy = copy.copy(sym)
    if isometry:
        off_diag_mask = (np.ones((p, p)) - np.eye(p)).astype(np.bool)
        sym_copy[..., off_diag_mask] *= np.sqrt(2)

    return sym_copy[..., tril_mask]


def vec_to_sym(vec, isometry=True):
    """Return the symmetric 2D array given its flattened lower triangular part.

    If isometry is True, divides the off-diagonal terms by sqrt(2).
    Acts on the last dimension of the array.

    Parameters
    ==========
    vec: numpy.ndarray
        The input array, shape (..., p * (p + 1) /2).
    isometry: bool, optional (default to True)
        Off diagonal terms of sym are divided by sqrt(2) if True.

    Returns
    =======
    sym: numpy.ndarray
        The output symmetric array, shape (..., p, p).
    """
    n = vec.shape[-1]
    # solve p * (p + 1) / 2 = n subj. to p > 0
    # p ** 2 + p - 2n = 0 & p > 0
    # p = - 1 / 2 + sqrt( 1 + 8 * n) / 2
    p = (np.sqrt(8 * n + 1) - 1.) / 2
    try:
        np.testing.assert_almost_equal(p, int(p))
    except AssertionError:
        raise ValueError("Vector size unsuitable, can not transform vector to "
                         "symmetric matrix")

    p = int(p)
    tril_mask = np.tril(np.ones((p, p))).astype(np.bool)
    off_diag_mask = (np.ones((p, p)) - np.eye(p)).astype(np.bool)
    sym = np.zeros(vec.shape[:-1] + (p, p), dtype=np.float)
    sym[..., tril_mask] = vec
    (sym.swapaxes(-1, -2))[..., tril_mask] = vec
    if isometry:
        sym[..., off_diag_mask] /= np.sqrt(2)

    return sym


def cov_to_corr(cov):
    """Return correlation matrix for a given covariance matrix.

    Parameters
    ==========
    cov: numpy.ndarray
        The 2-D input array, covariance matrix.

    Returns
    =======
    corr: numpy.ndarray
        The 2-D ouput array, correlation matrix.
    """
    d = np.diag(cov)
    corr = cov * d ** (-1. / 2) * (d ** (-1. / 2))[..., np.newaxis]
    return corr


def prec_to_partial(prec):
    """Return partial correlation matrix for a given precision matrix.

    Parameters
    ==========
    prec: numpy.ndarray
        The 2D input array, precision matrix.

    Returns
    =======
    partial: numpy.ndarray
        The 2D ouput array, partial correlation matrix.
    """
    partial = -cov_to_corr(prec)
    np.fill_diagonal(partial, 1.)
    return partial


class CovEmbedding(BaseEstimator, TransformerMixin):
    """
    Tranformer that returns the coefficients on a flat space to
    perform the analysis.

    Parameters
    ----------
    cov_estimator: estimator object, optional (default to
        sklearn.covariance.EmpiricalCovariance)
        The covariance estimator.
    kind: {"correlation", "partial correlation", "tangent", "precision"},
        optional (default to "covariance")
        The connectivity measure, default to "tangent".

    Attributes
    ----------
    `cov_estimator_` : estimator object
        A new covariance estimator with the same parameters as cov_estimator.

    `mean_cov_` : numpy.ndarray
        The geometric mean of the covariance matrices.

    `whitening_` : numpy.ndarray
        The inverted square-rooted geometric mean.
    """

    def __init__(self, cov_estimator=None, kind=None):
        self.cov_estimator = cov_estimator
        self.kind = kind

    def fit(self, X, y=None):
        """Fits the group sparse precision model according to the given
        training data and parameters.

        Parameters
        ----------
        X: list of numpy.ndarray with shapes (n_samples, n_features)
            The input subjects.

        Attributes
        ----------
        `cov_estimator_` : estimator object
            A new covariance estimator with the same parameters as
            cov_estimator.

        `mean_cov_` : numpy.ndarray
            The geometric mean of the covariance matrices.

        `whitening_` : numpy.ndarray
            The inverted square-rooted geometric mean.

        Returns
        -------
        self : CovEmbedding instance
            The object itself. Useful for chaining operations.
        """

        if self.cov_estimator is None:
            self.cov_estimator_ = EmpiricalCovariance(assume_centered=True)
        else:
            self.cov_estimator_ = clone(self.cov_estimator)

        if self.kind is None:
            self.kind = 'covariance'
        elif self.kind == 'tangent':
            covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
            self.mean_cov_ = geometric_mean(covs, max_iter=30, tol=1e-7)
            self.whitening_ = inv_sqrtm(self.mean_cov_)

        return self

    def transform(self, X):
        """Apply transform to covariances

        Parameters
        ----------
        X: list of numpy.ndarray with shapes (n_samples, n_features)
            The input subjects.

        Returns
        -------
        numpy.ndarray, transformed covariance matrices, shape
            (len(X), n_features * (n_features + 1) / 2, )
        """
        covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
        covs = stack_newaxis(covs)
        if self.kind == 'covariance':
            pass
        elif self.kind == 'tangent':
            covs = [logm(self.whitening_.dot(c).dot(self.whitening_))
                    for c in covs]
        elif self.kind == 'precision':
            covs = [inv(g) for g in covs]
        elif self.kind == 'partial correlation':
            covs = [prec_to_partial(inv(g)) for g in covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        else:
            raise ValueError("Unknown connectivity measure.")

        return np.array([sym_to_vec(c) for c in covs])