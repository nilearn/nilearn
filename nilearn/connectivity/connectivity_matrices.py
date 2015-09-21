import warnings
from math import sqrt

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance
from .._utils.extmath import is_spd


def _check_matrix(matrix, prop):
    """Raise a ValueError if the input matrix does not satisfy the property.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.

    prop : {'square', 'symmetric', 'spd'}
        Property to check.
    """
    if prop == 'square':
        if matrix.ndim != 2 or (matrix.shape[0] != matrix.shape[-1]):
            raise ValueError('Expected a square matrix, got array of shape' +
                             ' {0}.'.format(matrix.shape))
    if prop == 'symmetric':
        if not np.allclose(matrix, matrix.T):
            raise ValueError('Expected a symmetric matrix.')

    if prop == 'spd':
        if not is_spd(matrix, decimal=7):
            raise ValueError('Expected a symmetric positive definite matrix.')


def _form_symmetric(function, eigen_values, eigen_vectors):
    """Return the symmetric matrix with eigenvectors eigen_vectors and
    eigenvalues obtained by applying the function to eigen_values.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    eigen_values : numpy.ndarray, shape (n_features, )
        Input argument of the function.

    eigen_vectors : numpy.ndarray, shape (n_features, n_features)
        Unitary matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The symmetric matrix obtained after transforming the eigenvalues, while
        keeping the same eigenvectors.
    """
    return np.dot(eigen_vectors * function(eigen_values), eigen_vectors.T)


def _map_eigenvalues(function, symmetric):
    """Matrix function, for real symmetric matrices. The function is applied
    to the eigenvalues of symmetric.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    symmetric : numpy.ndarray, shape (n_features, n_features)
        The input symmetric matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The new symmetric matrix obtained after transforming the eigenvalues,
        while keeping the same eigenvectors.

    Note
    ----
    If input matrix is not real symmetric, no error is reported but result will
    be wrong.
    """
    eigen_values, eigen_vectors = linalg.eigh(symmetric)
    return _form_symmetric(function, eigen_values, eigen_vectors)


def _geometric_mean(matrices, init=None, max_iter=10, tol=1e-7):
    """Compute the geometric mean of symmetric positive definite matrices.

    The geometric mean of n positive definite matrices
    M_1, ..., M_n is the minimizer of the sum of squared distances from an
    arbitrary matrix to each input matrix M_k

    gmean(M_1, ..., M_n) = argmin_X sum_{k=1}^N dist(X, M_k)^2

    where the used distance is related to matrices logarithm

    dist(X, M_k) = ||log(X^{-1/2} M_k X^{-1/2)}||

    In case of positive numbers, this mean is the usual geometric mean.

    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ----------
    matrices : list of numpy.ndarray, shape of each (n_features, n_features)
        List of matrices whose geometric mean to compute. Raise an error if the
        arrays are not all symmetric positive definite of the same shape.

    init : numpy.ndarray, shape (n_features, n_features) or None, optional
        Initialization matrix. Raise an error if the array is not symmetric
        positive definite of the same shape as the elements of matrices. Set to
        the arithmetic mean of matrices if None.

    max_iter : int, optional (default to 10)
        Maximal number of iterations.

    tol : float, optional (default to 1e-7)
        Tolerance.

    Returns
    -------
    gmean : numpy.ndarray, shape (n_features, n_features)
        Geometric mean of the matrices.
    """
    # Shape and symmetry positive definiteness checks
    n_features = matrices[0].shape[0]
    for matrix in matrices:
        _check_matrix(matrix, 'square')
        if matrix.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")
        _check_matrix(matrix, 'spd')

    # Initialization
    matrices = np.array(matrices)
    if init is None:
        gmean = np.mean(matrices, axis=0)
    else:
        _check_matrix(init, 'square')
        if init.shape[0] != n_features:
            raise ValueError("Initialization has incorrect shape.")
        _check_matrix(init, 'spd')
        gmean = init

    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_gmean, vecs_gmean)
        whitened_matrices = [gmean_inv_sqrt.dot(matrix).dot(gmean_inv_sqrt)
                             for matrix in matrices]
        logs = [_map_eigenvalues(np.log, w_mat) for w_mat in whitened_matrices]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - gmean.dot(logms_mean)
        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point gmean

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        # Move along the geodesic
        gmean = gmean_sqrt.dot(
            _form_symmetric(np.exp, vals_log * step, vecs_log)).dot(gmean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / gmean.size < tol:
            break
    if tol is not None and norm / gmean.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without "
                      "getting to the requested tolerance level "
                      "{1}.".format(max_iter, tol))

    return gmean


def sym_to_vec(symmetric):
    """Return the flattened lower triangular part of an array.

    Acts on the last two dimensions of the array if not 2-dimensional.

    Parameters
    ----------
    symmetric : numpy.ndarray, shape (..., n_features, n_features)
        Input array.

    Returns
    -------
    output : numpy.ndarray, shape (..., n_features * (n_features + 1) / 2)
        The output flattened lower triangular part of symmetric.
    """
    tril_mask = np.tril(np.ones(symmetric.shape[-2:]), -1).astype(np.bool)
    symmetric = symmetric.copy()
    symmetric[..., tril_mask] *= sqrt(2)
    tril_mask.flat[::symmetric.shape[-1] + 1] = True
    return symmetric[..., tril_mask]


def _cov_to_corr(covariance):
    """Return correlation matrix for a given covariance matrix.

    Parameters
    ----------
    covariance : 2D numpy.ndarray
        The input covariance matrix.

    Returns
    -------
    correlation : 2D numpy.ndarray
        The ouput correlation matrix.
    """
    diagonal = np.atleast_2d(1. / np.sqrt(np.diag(covariance)))
    correlation = covariance * diagonal * diagonal.T
    return correlation


def _prec_to_partial(precision):
    """Return partial correlation matrix for a given precision matrix.

    Parameters
    ----------
    precision : 2D numpy.ndarray
        The input precision matrix.

    Returns
    -------
    partial_correlation : 2D numpy.ndarray
        The 2D ouput partial correlation matrix.
    """
    partial_correlation = -_cov_to_corr(precision)
    np.fill_diagonal(partial_correlation, 1.)
    return partial_correlation


class ConnectivityMeasure(BaseEstimator, TransformerMixin):
    """A class that computes different kinds of functional connectivity
    matrices.

    Parameters
    ----------
    cov_estimator : estimator object, optional (default to
        sklearn.covariance.EmpiricalCovariance()).
        The covariance estimator.

    kind : {"correlation", "partial correlation", "robust dispersion",
            "covariance", "precision"}, optional (default to 'covariance')
        The matrix kind.

    Attributes
    ----------
    `cov_estimator_` : estimator object
        A new covariance estimator with the same parameters as cov_estimator.

    `robust_mean_` : numpy.ndarray
        The mean connectivity for the robust dispersion kind.

    `whitening_` : numpy.ndarray
        The inverted square-rooted geometric mean of the covariance matrices.

    References
    ----------
    For the use of "robust dispersion", see the paper:
    G. Varoquaux et al. "Detection of brain functional-connectivity difference
    in post-stroke patients using group-level covariance modeling, MICCAI 2010.
    """

    def __init__(self, cov_estimator=EmpiricalCovariance(assume_centered=True),
                 kind='covariance'):
        self.cov_estimator = cov_estimator
        self.kind = kind

    def fit(self, X, y=None):
        """Fit the covariance estimator to the given time series for each
        subject.

        Parameters
        ----------
        X : list of numpy.ndarray, shapes (n_samples, n_features)
            The input subjects time series.

        Returns
        -------
        self : ConnectivityMatrix instance
            The object itself. Useful for chaining operations.
        """
        self.cov_estimator_ = clone(self.cov_estimator)

        if self.kind == 'robust dispersion':
            covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]
            self.robust_mean_ = _geometric_mean(covariances, max_iter=30,
                                                tol=1e-7)
            self.whitening_ = _map_eigenvalues(lambda x: 1. / np.sqrt(x),
                                               self.robust_mean_)

        return self

    def transform(self, X):
        """Apply transform to covariances matrices to get the connectivity
        matrices for the chosen kind.

        Parameters
        ----------
        X : list of numpy.ndarray with shapes (n_samples, n_features)
            The input subjects time series.

        Returns
        -------
        output : numpy.ndarray, shape (n_samples, n_features, n_features)
             The transformed connectivity matrices.
        """
        covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]
        covariances = np.array(covariances)
        if self.kind == 'covariance':
            connectivities = covariances
        elif self.kind == 'robust dispersion':
            connectivities = [_map_eigenvalues(np.log, self.whitening_.dot(
                                               cov).dot(self.whitening_))
                              for cov in covariances]
        elif self.kind == 'precision':
            connectivities = [linalg.inv(cov) for cov in covariances]
        elif self.kind == 'partial correlation':
            connectivities = [_prec_to_partial(linalg.inv(cov))
                              for cov in covariances]
        elif self.kind == 'correlation':
            connectivities = [_cov_to_corr(cov) for cov in covariances]
        else:
            raise ValueError("Unknown connectivity kind.")

        return np.array(connectivities)
