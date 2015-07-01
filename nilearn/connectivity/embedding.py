import warnings
from math import floor, sqrt

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance
from .._utils.extmath import is_spd


def check_mat(mat, prop):
    """Raise a ValueError if the input matrix does not satisfy the property.

    Parameters
    ----------
    mat : numpy.ndarray
        Input array.

    prop : {'square', 'symmetric', 'spd'}
        Property to check.
    """
    if prop == 'square':
        if mat.ndim != 2 or (mat.shape[0] != mat.shape[-1]):
            raise ValueError('Expected a square matrix, got array of shape' +
                             ' {0}.'.format(mat.shape))
    if prop == 'symmetric':
        if not np.allclose(mat, mat.T):
            raise ValueError('Expected a symmetric matrix.')

    if prop == 'spd':
        if not is_spd(mat, decimal=7):
            raise ValueError('Expected a symmetric positive definite matrix.')


def map_eig(function, vals, vecs):
    """Return the symmetric matrix with eigenvectors vecs and eigenvalues
    obtained by applying the function to vals.

    Parameters
    ----------
    function : function
        The function to apply.

    vals : numpy.ndarray, shape (M, )
        Input argument of the function.

    vecs : numpy.ndarray, shape (M, M)
        Unitary matrix.

    Returns
    -------
    output : numpy.ndarray, shape (M, M)
        The symmetric matrix obtained after transforming the eigenvalues, with
        eigenvectors the columns of vecs.
    """
    return np.dot(vecs * function(vals), vecs.T)


def map_sym(function, sym):
    """Matrix function, for real symmetric matrices. The function is applied
    to the eigenvalues of sym.

    Parameters
    ----------
    function : function
        The function to apply.

    sym : numpy.ndarray, shape (M, M)
        The matrix to be transformed.

    Returns
    -------
    output : numpy.ndarray, shape (M, M)
        The new symmetric matrix obtained after transforming the eigenvalues.

    Note
    ----
    If input matrix is not real symmetric, no error is reported but result will
    be wrong.
    """
    vals, vecs = linalg.eigh(sym)
    return map_eig(function, vals, vecs)


def geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """Compute the geometric mean of symmetric positive definite matrices.

    The geometric mean is the minimizer of the sum of squared distances from an
    arbitrary matrix to each input matrix in the manifold of symmetric positive
    definite matrices.

    Minimization of the objective function is done by an intrinsic gradient
    descent in the manifold: moving from the current point to the next one
    along a short geodesic arc in the opposite direction of the
    covariant derivative.

    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ----------
    mats : list of numpy.ndarray, shape of each (n_features, n_features)
        List of matrices whose geometric mean to compute. Raise an error if the
        arrays are not all symmetric positive definite of the same shape.

    init : numpy.ndarray, shape (n_features, n_features) or None, optional
        Initialization matrix. Raise an error if the array is not symmetric
        positive definite of the same shape as the elements of mats. Set to the
        arithmetic mean of mats if None.

    max_iter : int, optional (default to 10)
        Maximal number of iterations.

    tol : float, optional (default to 1e-7)
        Tolerance.

    Returns
    -------
    geo : numpy.ndarray, shape (n_features, n_features)
        Geometric mean of the matrices.
    """
    # Shape and symmetry positive definiteness checks
    n_features = mats[0].shape[0]
    for mat in mats:
        check_mat(mat, 'square')
        if mat.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")
        check_mat(mat, 'spd')

    # Initialization
    mats = np.array(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        check_mat(init, 'square')
        if init.shape[0] != n_features:
            raise ValueError("Initialization has not the correct shape.")
        check_mat(init, 'spd')
        geo = init

    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
            for mat in mats]
        logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo = geo_sqrt.dot(map_eig(np.exp, vals_log * step, vecs_log)).dot(
            geo_sqrt)  # Move along the geodesic with step size step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / geo.size < tol:
            break
    if tol is not None and norm / geo.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without " \
                      "getting to the requested tolerance level " \
                      "{1}.".format(max_iter, tol))

    return geo


def grad_geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """Return the norm of the covariant derivative at each iteration step of
    geometric_mean. See its docstring for details.

    Norm is intrinsic norm on the tangent space of the manifold of symmetric
    positive definite matrices.

    Returns
    -------
    grad_norm : list of float
        Norm of the covariant derivative in the tangent space at each step.
    """
    mats = np.array(mats)

    # Initialization
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        geo = init
    norm_old = np.inf
    step = 1.
    grad_norm = []
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
            for mat in mats]
        logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo = geo_sqrt.dot(map_eig(np.exp, vals_log * step, vecs_log)).dot(
            geo_sqrt)  # Move along the geodesic with step size step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old

        grad_norm.append(norm / geo.size)
        if tol is not None and norm / geo.size < tol:
            break

    return grad_norm


def sym_to_vec(sym, isometry=True):
    """Return the flattened lower triangular part of an array.

    Acts on the last two dimensions of the array if not 2-dimensional.

    Parameters
    ----------
    sym : numpy.ndarray, shape (..., p, p)
        Input array.

    isometry : bool, optional (default to True)
        If True, off diagonal terms of sym are multiplied by sqrt(2).

    Returns
    -------
    output : numpy.ndarray, shape (..., p * (p + 1) / 2)
        The output flattened lower triangular part of sym.
    """
    tril_mask = np.tril(np.ones(sym.shape[-2:]), -1).astype(np.bool)
    if isometry:
        sym = sym.copy()
        sym[..., tril_mask] *= sqrt(2)

    tril_mask.flat[::sym.shape[-1] + 1] = True
    return sym[..., tril_mask]


def vec_to_sym(vec, isometry=True):
    """Return the symmetric array given its flattened lower triangular part.

    Acts on the last dimension of the array if not 1-dimensional.

    Parameters
    ----------
    vec : numpy.ndarray, shape (..., p * (p + 1) /2)
        The input array.

    isometry : bool, optional (default to True)
        If True, off diagonal terms of the output array are divided by sqrt(2).

    Returns
    -------
    sym : numpy.ndarray, shape (..., p, p)
        The output symmetric array.
    """
    n = vec.shape[-1]
    # solve p * (p + 1) / 2 = n subj. to p > 0
    # p ** 2 + p - 2n = 0 & p > 0
    # p = - 1 / 2 + sqrt( 1 + 8 * n) / 2
    p = (sqrt(8 * n + 1) - 1.) / 2
    if p > floor(p):
        raise ValueError("Vector size unsuitable, can not transform vector to "
                         "symmetric matrix.")

    p = int(p)
    mask = np.tril(np.ones((p, p))).astype(np.bool)
    sym = np.zeros(vec.shape[:-1] + (p, p))
    sym[..., mask] = vec
    sym.swapaxes(-1, -2)[..., mask] = vec
    if isometry:
        mask.flat[::p + 1] = False
        mask = mask + mask.T
        sym[..., mask] /= sqrt(2)

    return sym


def cov_to_corr(cov):
    """Return correlation matrix for a given covariance matrix.

    Parameters
    ----------
    cov : 2D numpy.ndarray
        The input covariance matrix.

    Returns
    -------
    corr : 2D numpy.ndarray
        The ouput correlation matrix.
    """
    d = np.atleast_2d(1. / np.sqrt(np.diag(cov)))
    corr = cov * d * d.T
    return corr


def prec_to_partial(prec):
    """Return partial correlation matrix for a given precision matrix.

    Parameters
    ----------
    prec : 2D numpy.ndarray
        The input precision matrix.

    Returns
    -------
    partial : 2D numpy.ndarray
        The 2D ouput partial correlation matrix.
    """
    partial = -cov_to_corr(prec)
    np.fill_diagonal(partial, 1.)
    return partial


class CovEmbedding(BaseEstimator, TransformerMixin):
    """Tranformer that returns the coefficients on a flat space to perform the
    analysis.

    Parameters
    ----------
    cov_estimator : estimator object, optional
        The covariance estimator.

    kind : {"correlation", "partial correlation", "tangent", "precision"}, \
        optional
        The connectivity measure.

    Attributes
    ----------
    `cov_estimator_` : estimator object
        A new covariance estimator with the same parameters as cov_estimator.

    `mean_cov_` : numpy.ndarray
        The geometric mean of the covariance matrices.

    `whitening_` : numpy.ndarray
        The inverted square-rooted geometric mean of the covariance matrices.
    """

    def __init__(self, cov_estimator=EmpiricalCovariance(assume_centered=True),
                 kind='covariance'):
        self.cov_estimator = cov_estimator
        self.kind = kind

    def fit(self, X, y=None):
        """Fits the group sparse precision model according to the given
        training data and parameters.

        Parameters
        ----------
        X : list of numpy.ndarray, shapes (n_samples, n_features)
            The input subjects.

        Attributes
        ----------
        `cov_estimator_` : estimator object
            A new covariance estimator with the same parameters as
            cov_estimator.

        `mean_cov_` : numpy.ndarray
            The geometric mean of the covariance matrices.

        `whitening_` : numpy.ndarray
            The inverted square-rooted geometric mean of the covariance
            matrices.

        Returns
        -------
        self : CovEmbedding instance
            The object itself. Useful for chaining operations.
        """
        self.cov_estimator_ = clone(self.cov_estimator)

        if self.kind == 'tangent':
            covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
            self.mean_cov_ = geometric_mean(covs, max_iter=30, tol=1e-7)
            self.whitening_ = map_sym(lambda x: 1. / np.sqrt(x),
                                      self.mean_cov_)

        return self

    def transform(self, X):
        """Apply transform to covariances.

        Parameters
        ----------
        X : list of numpy.ndarray with shapes (n_samples, n_features)
            The input subjects.

        Returns
        -------
        output : numpy.ndarray, shape (len(X), \
                                       n_features * (n_features + 1) / 2)
             The transformed covariance matrices.
        """
        covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
        covs = np.array(covs)
        if self.kind == 'covariance':
            pass
        elif self.kind == 'tangent':
            covs = [map_sym(np.log, self.whitening_.dot(c).dot(
                            self.whitening_)) for c in covs]
        elif self.kind == 'precision':
            covs = [map_sym(lambda x: 1. / x, g) for g in covs]
        elif self.kind == 'partial correlation':
            covs = [prec_to_partial(map_sym(lambda x: 1. / x, g))
                for g in covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        else:
            raise ValueError("Unknown connectivity measure.")

        return np.array([sym_to_vec(c) for c in covs])