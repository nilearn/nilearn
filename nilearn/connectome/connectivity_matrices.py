import warnings
from math import sqrt

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from .. import signal
from .._utils.extmath import is_spd


def _check_square(matrix):
    """Raise a ValueError if the input matrix is square.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.
    """
    if matrix.ndim != 2 or (matrix.shape[0] != matrix.shape[-1]):
        raise ValueError('Expected a square matrix, got array of shape'
                         ' {0}.'.format(matrix.shape))


def _check_spd(matrix):
    """Raise a ValueError if the input matrix is not symmetric positive
    definite.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.
    """
    if not is_spd(matrix, decimal=7):
        raise ValueError('Expected a symmetric positive definite matrix.')


def _form_symmetric(function, eigenvalues, eigenvectors):
    """Return the symmetric matrix with the given eigenvectors and
    eigenvalues transformed by function.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    eigenvalues : numpy.ndarray, shape (n_features, )
        Input argument of the function.

    eigenvectors : numpy.ndarray, shape (n_features, n_features)
        Unitary matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The symmetric matrix obtained after transforming the eigenvalues, while
        keeping the same eigenvectors.
    """
    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)


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
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    return _form_symmetric(function, eigenvalues, eigenvectors)


def _geometric_mean(matrices, init=None, max_iter=10, tol=1e-7):
    """Compute the geometric mean of symmetric positive definite matrices.

    The geometric mean of n positive definite matrices
    M_1, ..., M_n is the minimizer of the sum of squared distances from an
    arbitrary matrix to each input matrix M_k

    gmean(M_1, ..., M_n) = argmin_X sum_{k=1}^N dist(X, M_k)^2

    where the used distance is related to matrices logarithm

    dist(X, M_k) = ||log(X^{-1/2} M_k X^{-1/2)}||

    In case of positive numbers, this mean is the usual geometric mean.

    References
    ----------
    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ----------
    matrices : list of numpy.ndarray, all of shape (n_features, n_features)
        List of matrices whose geometric mean to compute. Raise an error if the
        matrices are not all symmetric positive definite of the same shape.

    init : numpy.ndarray, shape (n_features, n_features), optional
        Initialization matrix, default to the arithmetic mean of matrices.
        Raise an error if the matrix is not symmetric positive definite of the
        same shape as the elements of matrices.

    max_iter : int, optional
        Maximal number of iterations.

    tol : positive float or None, optional
        The tolerance to declare convergence: if the gradient norm goes below
        this value, the gradient descent is stopped. If None, no  check is
        performed.

    Returns
    -------
    gmean : numpy.ndarray, shape (n_features, n_features)
        Geometric mean of the matrices.
    """
    # Shape and symmetry positive definiteness checks
    n_features = matrices[0].shape[0]
    for matrix in matrices:
        _check_square(matrix)
        if matrix.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")
        _check_spd(matrix)

    # Initialization
    matrices = np.array(matrices)
    if init is None:
        gmean = np.mean(matrices, axis=0)
    else:
        _check_square(init)
        if init.shape[0] != n_features:
            raise ValueError("Initialization has incorrect shape.")
        _check_spd(init)
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
        elif norm > norm_old:
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
    """Return the flattened lower triangular part of an array, after
    multiplying above the diagonal elements by sqrt(2).

    Acts on the last two dimensions of the array if not 2-dimensional.

    .. versionadded:: 0.2

    Parameters
    ----------
    symmetric : numpy.ndarray, shape (..., n_features, n_features)
        Input array.

    Returns
    -------
    output : numpy.ndarray, shape (..., n_features * (n_features + 1) / 2)
        The output flattened lower triangular part of symmetric.
    """
    scaling = sqrt(2) * np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, 1.)
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(np.bool)
    return symmetric[..., tril_mask] * scaling[tril_mask]


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

    .. versionadded:: 0.2

    Parameters
    ----------
    cov_estimator : estimator object, optional.
        The covariance estimator. By default the LedoitWolf estimator
        is used. This implies that correlations are slightly shrunk
        towards zero compared to a maximum-likelihood estimate

    kind : {"correlation", "partial correlation", "tangent",\
            "covariance", "precision"}, optional
        The matrix kind.

    Attributes
    ----------
    `cov_estimator_` : estimator object
        A new covariance estimator with the same parameters as cov_estimator.

    `mean_` : numpy.ndarray
        The mean connectivity for the tangent kind.

    `whitening_` : numpy.ndarray
        The inverted square-rooted geometric mean of the covariance matrices.

    References
    ----------
    For the use of "tangent", see the paper:
    G. Varoquaux et al. "Detection of brain functional-connectivity difference
    in post-stroke patients using group-level covariance modeling, MICCAI 2010.
    """

    def __init__(self, cov_estimator=LedoitWolf(store_precision=False),
                 kind='covariance'):
        self.cov_estimator = cov_estimator
        self.kind = kind

    def fit(self, X, y=None):
        """Fit the covariance estimator to the given time series for each
        subject.

        Parameters
        ----------
        X : list of numpy.ndarray, shape for each (n_samples, n_features)
            The input subjects time series.

        Returns
        -------
        self : ConnectivityMatrix instance
            The object itself. Useful for chaining operations.
        """
        self.cov_estimator_ = clone(self.cov_estimator)
        if not hasattr(X, "__iter__"):
            raise ValueError("'subjects' input argument must be an iterable. "
                             "You provided {0}".format(X.__class__))

        subjects_types = [type(s) for s in X]
        if set(subjects_types) != set([np.ndarray]):
            raise ValueError("Each subject must be 2D numpy.ndarray.\n You "
                             "provided {0}".format(str(subjects_types)))

        subjects_dims = [s.ndim for s in X]
        if set(subjects_dims) != set([2]):
            raise ValueError("Each subject must be 2D numpy.ndarray.\n You"
                             "provided arrays of dimensions "
                             "{0}".format(str(subjects_dims)))

        n_subjects = [s.shape[1] for s in X]
        if len(set(n_subjects)) > 1:
            raise ValueError("All subjects must have the same number of "
                             "features.\nYou provided: "
                             "{0}".format(str(n_subjects)))

        if self.kind == 'tangent':
            covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]
            self.mean_ = _geometric_mean(covariances, max_iter=30, tol=1e-7)
            self.whitening_ = _map_eigenvalues(lambda x: 1. / np.sqrt(x),
                                               self.mean_)

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
        if self.kind == 'correlation':
            covariances_std = [self.cov_estimator_.fit(
                signal._standardize(x, detrend=False, normalize=True)
                ).covariance_ for x in X]
            connectivities = [_cov_to_corr(cov) for cov in covariances_std]
        else:
            covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]
            if self.kind == 'covariance':
                connectivities = covariances
            elif self.kind == 'tangent':
                connectivities = [_map_eigenvalues(np.log, self.whitening_.dot(
                                                   cov).dot(self.whitening_))
                                  for cov in covariances]
            elif self.kind == 'precision':
                connectivities = [linalg.inv(cov) for cov in covariances]
            elif self.kind == 'partial correlation':
                connectivities = [_prec_to_partial(linalg.inv(cov))
                                  for cov in covariances]
            else:
                raise ValueError('Allowed connectivity kinds are '
                                 '"correlation", '
                                 '"partial correlation", "tangent", '
                                 '"covariance" and "precision", got kind '
                                 '"{}"'.format(self.kind))

        return np.array(connectivities)
