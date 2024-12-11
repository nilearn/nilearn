"""Connectivity matrices."""

import warnings
from math import floor, sqrt

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf

from nilearn._utils.docs import fill_doc

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
        raise ValueError(
            f"Expected a square matrix, got array of shape {matrix.shape}."
        )


def _check_spd(matrix):
    """Raise a ValueError if the input matrix is not symmetric positive \
    definite.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.

    """
    if not is_spd(matrix, decimal=7):
        raise ValueError("Expected a symmetric positive definite matrix.")


def _form_symmetric(function, eigenvalues, eigenvectors):
    """Return the symmetric matrix with the given eigenvectors and \
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
    """Matrix function, for real symmetric matrices.

    The function is applied to the eigenvalues of symmetric.

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

    Notes
    -----
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

    See Algorithm 3 of :footcite:t:`Fletcher2007`.

    References
    ----------
    .. footbibliography::

    Parameters
    ----------
    matrices : list of numpy.ndarray, all of shape (n_features, n_features)
        List of matrices whose geometric mean to compute. Raise an error if the
        matrices are not all symmetric positive definite of the same shape.

    init : numpy.ndarray, shape (n_features, n_features), optional
        Initialization matrix, default to the arithmetic mean of matrices.
        Raise an error if the matrix is not symmetric positive definite of the
        same shape as the elements of matrices.

    max_iter : int, default=10
        Maximal number of iterations.

    tol : positive float or None, default=1e-7
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
    step = 1.0

    # Gradient descent
    for _ in range(max_iter):
        # Computation of the gradient
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_gmean, vecs_gmean)
        whitened_matrices = [
            gmean_inv_sqrt.dot(matrix).dot(gmean_inv_sqrt)
            for matrix in matrices
        ]
        logs = [_map_eigenvalues(np.log, w_mat) for w_mat in whitened_matrices]
        # Covariant derivative is - gmean.dot(logms_mean)
        logs_mean = np.mean(logs, axis=0)
        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        # Norm of the covariant derivative on the tangent space at point gmean
        norm = np.linalg.norm(logs_mean)

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        # Move along the geodesic
        gmean = gmean_sqrt.dot(
            _form_symmetric(np.exp, vals_log * step, vecs_log)
        ).dot(gmean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.0
            norm = norm_old
        if tol is not None and norm / gmean.size < tol:
            break
    if tol is not None and norm / gmean.size >= tol:
        warnings.warn(
            f"Maximum number of iterations {max_iter} reached without "
            f"getting to the requested tolerance level {tol}."
        )

    return gmean


def sym_matrix_to_vec(symmetric, discard_diagonal=False):
    """Return the flattened lower triangular part of an array.

    If diagonal is kept, diagonal elements are divided by sqrt(2) to conserve
    the norm.

    Acts on the last two dimensions of the array if not 2-dimensional.

    .. versionadded:: 0.3

    Parameters
    ----------
    symmetric : numpy.ndarray or list of numpy arrays, shape\
        (..., n_features, n_features)
        Input array.

    discard_diagonal : boolean, default=False
        If True, the values of the diagonal are not returned.

    Returns
    -------
    output : numpy.ndarray
        The output flattened lower triangular part of symmetric. Shape is
        (..., n_features * (n_features + 1) / 2) if discard_diagonal is False
        and (..., (n_features - 1) * n_features / 2) otherwise.

    """
    if discard_diagonal:
        # No scaling, we directly return the values
        tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(bool)
        return symmetric[..., tril_mask]
    scaling = np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, sqrt(2.0))
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(bool)
    return symmetric[..., tril_mask] / scaling[tril_mask]


def vec_to_sym_matrix(vec, diagonal=None):
    """Return the symmetric matrix given its flattened lower triangular part.

    Acts on the last dimension of the array if not 1-dimensional.
    Diagonal can be encompassed in vec or given separately. In both cases, note
    that diagonal elements are multiplied by sqrt(2).

    .. versionadded:: 0.3

    Parameters
    ----------
    vec : numpy.ndarray or list of numpy arrays, shape \
        (..., n_columns * (n_columns + 1) /2) or
        (..., (n_columns - 1) * n_columns / 2) if diagonal is given separately.
        The input array.

    diagonal : numpy.ndarray, shape (..., n_columns), optional
        The diagonal array to be stacked to vec. If None, the diagonal is
        assumed to be included in vec.

    Returns
    -------
    sym : numpy.ndarray, shape (..., n_columns, n_columns).
        The output symmetric matrix.

    Notes
    -----
    This function is meant to be the inverse of sym_matrix_to_vec. If you have
    discarded the diagonal in sym_matrix_to_vec, you need to provide it
    separately to reconstruct the symmetric matrix. For instance this can be
    useful for correlation matrices for which we know the diagonal is 1.

    See Also
    --------
    nilearn.connectome.sym_matrix_to_vec

    """
    n = vec.shape[-1]
    # Compute the number of the symmetric matrix columns
    # solve n_columns * (n_columns + 1) / 2 = n subject to n_columns > 0
    n_columns = (sqrt(8 * n + 1) - 1.0) / 2
    if diagonal is not None:
        n_columns += 1

    if n_columns > floor(n_columns):
        raise ValueError(
            f"Vector of unsuitable shape {vec.shape} cannot be transformed to "
            "a symmetric matrix."
        )

    n_columns = int(n_columns)
    first_shape = vec.shape[:-1]
    if diagonal is not None and (
        diagonal.shape[:-1] != first_shape or diagonal.shape[-1] != n_columns
    ):
        raise ValueError(
            f"diagonal of shape {diagonal.shape} incompatible "
            f"with vector of shape {vec.shape}"
        )

    sym = np.zeros((*first_shape, n_columns, n_columns))

    # Fill lower triangular part
    skip_diagonal = diagonal is not None
    mask = np.tril(np.ones((n_columns, n_columns)), k=-skip_diagonal).astype(
        bool
    )
    sym[..., mask] = vec

    # Fill upper triangular part
    sym.swapaxes(-1, -2)[..., mask] = vec

    # (Fill and) rescale diagonal terms
    mask.fill(False)
    np.fill_diagonal(mask, True)
    if diagonal is not None:
        sym[..., mask] = diagonal

    sym[..., mask] *= sqrt(2)

    return sym


def cov_to_corr(covariance):
    """Return correlation matrix for a given covariance matrix.

    Parameters
    ----------
    covariance : 2D numpy.ndarray
        The input covariance matrix.

    Returns
    -------
    correlation : 2D numpy.ndarray
        The output correlation matrix.

    """
    diagonal = np.atleast_2d(1.0 / np.sqrt(np.diag(covariance)))
    correlation = covariance * diagonal * diagonal.T

    # Force exact 1. on diagonal
    np.fill_diagonal(correlation, 1.0)
    return correlation


def prec_to_partial(precision):
    """Return partial correlation matrix for a given precision matrix.

    Parameters
    ----------
    precision : 2D numpy.ndarray
        The input precision matrix.

    Returns
    -------
    partial_correlation : 2D numpy.ndarray
        The 2D output partial correlation matrix.

    """
    partial_correlation = -cov_to_corr(precision)
    np.fill_diagonal(partial_correlation, 1.0)
    return partial_correlation


@fill_doc
class ConnectivityMeasure(TransformerMixin, BaseEstimator):
    """A class that computes different kinds of \
       :term:`functional connectivity` matrices.

    .. versionadded:: 0.2

    Parameters
    ----------
    cov_estimator : estimator object, \
                    default=LedoitWolf(store_precision=False)
        The covariance estimator.
        This implies that correlations are slightly shrunk
        towards zero compared to a maximum-likelihood estimate

    kind : {"covariance", "correlation", "partial correlation",\
            "tangent", "precision"}, default='covariance'
        The matrix kind.
        For the use of "tangent" see :footcite:t:`Varoquaux2010b`.

    vectorize : bool, default=False
        If True, connectivity matrices are reshaped into 1D arrays and only
        their flattened lower triangular parts are returned.

    discard_diagonal : bool, default=False
        If True, vectorized connectivity coefficients do not include the
        matrices diagonal elements. Used only when vectorize is set to True.

    %(standardize)s

        .. note::

            Added to control passing value to `standardize` of ``signal.clean``
            to call new behavior since passing "zscore" or True (default) is
            deprecated. This parameter will be deprecated in version 0.13 and
            removed in version 0.15.

    Attributes
    ----------
    cov_estimator_ : estimator object, default=None
        A new covariance estimator with the same parameters as cov_estimator.
        If ``None`` is passed,
        defaults to ``LedoitWolf(store_precision=False)``.

    mean_ : numpy.ndarray
        The mean connectivity matrix across subjects. For 'tangent' kind,
        it is the geometric mean of covariances (a group covariance
        matrix that captures information from both correlation and partial
        correlation matrices). For other values for "kind", it is the
        mean of the corresponding matrices

    whitening_ : numpy.ndarray
        The inverted square-rooted geometric mean of the covariance matrices.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        cov_estimator=None,
        kind="covariance",
        vectorize=False,
        discard_diagonal=False,
        standardize=True,
    ):
        self.cov_estimator = cov_estimator
        self.kind = kind
        self.vectorize = vectorize
        self.discard_diagonal = discard_diagonal
        self.standardize = standardize

    def _check_input(self, X, confounds=None):
        if not hasattr(X, "__iter__"):
            raise ValueError(
                "'subjects' input argument must be an iterable. "
                f"You provided {X.__class__}"
            )

        subjects_types = [type(s) for s in X]
        if set(subjects_types) != {np.ndarray}:
            raise ValueError(
                "Each subject must be 2D numpy.ndarray.\n "
                f"You provided {subjects_types}"
            )

        subjects_dims = [s.ndim for s in X]
        if set(subjects_dims) != {2}:
            raise ValueError(
                "Each subject must be 2D numpy.ndarray.\n "
                f"You provided arrays of dimensions {subjects_dims}"
            )

        features_dims = [s.shape[1] for s in X]
        if len(set(features_dims)) > 1:
            raise ValueError(
                "All subjects must have the same number of features.\n"
                f"You provided: {features_dims}"
            )

        if confounds is not None and not hasattr(confounds, "__iter__"):
            raise ValueError(
                "'confounds' input argument must be an iterable. "
                f"You provided {confounds.__class__}"
            )

    def fit(
        self,
        X,
        y=None,  # noqa: ARG002
    ):
        """Fit the covariance estimator to the given time series for each \
        subject.

        Parameters
        ----------
        X : list of numpy.ndarray, shape for each (n_samples, n_features)
            The input subjects time series. The number of samples may differ
            from one subject to another.

        Returns
        -------
        self : ConnectivityMatrix instance
            The object itself. Useful for chaining operations.

        """
        self._fit_transform(X, do_fit=True)
        return self

    def _fit_transform(
        self, X, do_transform=False, do_fit=False, confounds=None
    ):
        """Avoid duplication of computation."""
        if self.cov_estimator is None:
            self.cov_estimator = LedoitWolf(store_precision=False)

        self._check_input(X, confounds=confounds)

        if do_fit:
            self.cov_estimator_ = clone(self.cov_estimator)

        # Compute all the matrices, stored in "connectivities"
        if self.kind == "correlation":
            covariances_std = [
                self.cov_estimator_.fit(
                    signal.standardize_signal(
                        x,
                        detrend=False,
                        standardize=self.standardize,
                    )
                ).covariance_
                for x in X
            ]
            connectivities = [cov_to_corr(cov) for cov in covariances_std]
        else:
            covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]
            if self.kind in ("covariance", "tangent"):
                connectivities = covariances
            elif self.kind == "precision":
                connectivities = [linalg.inv(cov) for cov in covariances]
            elif self.kind == "partial correlation":
                connectivities = [
                    prec_to_partial(linalg.inv(cov)) for cov in covariances
                ]
            else:
                allowed_kinds = (
                    "correlation",
                    "partial correlation",
                    "tangent",
                    "covariance",
                    "precision",
                )
                raise ValueError(
                    f"Allowed connectivity kinds are {allowed_kinds}. "
                    f"Got kind {self.kind}."
                )

        # Store the mean
        if do_fit:
            if self.kind == "tangent":
                self.mean_ = _geometric_mean(
                    covariances, max_iter=30, tol=1e-7
                )
                self.whitening_ = _map_eigenvalues(
                    lambda x: 1.0 / np.sqrt(x), self.mean_
                )
            else:
                self.mean_ = np.mean(connectivities, axis=0)
                # Fight numerical instabilities: make symmetric
                self.mean_ = self.mean_ + self.mean_.T
                self.mean_ *= 0.5

        # Compute the vector we return on transform
        if do_transform:
            if self.kind == "tangent":
                connectivities = [
                    _map_eigenvalues(
                        np.log, self.whitening_.dot(cov).dot(self.whitening_)
                    )
                    for cov in connectivities
                ]

            connectivities = np.array(connectivities)

            if confounds is not None and not self.vectorize:
                error_message = (
                    "'confounds' are provided but vectorize=False. "
                    "Confounds are only cleaned on vectorized matrices "
                    "as second level connectome regression "
                    "but not on symmetric matrices."
                )
                raise ValueError(error_message)

            if self.vectorize:
                connectivities = sym_matrix_to_vec(
                    connectivities, discard_diagonal=self.discard_diagonal
                )
                if confounds is not None:
                    connectivities = signal.clean(
                        connectivities, confounds=confounds
                    )

        return connectivities

    def fit_transform(
        self,
        X,
        y=None,  # noqa: ARG002
        confounds=None,
    ):
        """Fit the covariance estimator to the given time series \
        for each subject. \
        Then apply transform to covariance matrices for the chosen kind.

        Parameters
        ----------
        X : list of n_subjects numpy.ndarray with shapes \
            (n_samples, n_features)
            The input subjects time series. The number of samples may differ
            from one subject to another.

        confounds : np.ndarray with shape (n_samples) or \
                    (n_samples, n_confounds), or pandas DataFrame, optional
            Confounds to be cleaned on the vectorized matrices. Only takes
            into effect when vetorize=True.
            This parameter is passed to signal.clean. Please see the related
            documentation for details.

        Returns
        -------
        output : numpy.ndarray, shape (n_subjects, n_features, n_features) or \
            (n_subjects, n_features * (n_features + 1) / 2) if vectorize \
            is set to True.
            The transformed individual connectivities, as matrices or vectors.
            Vectors are cleaned when vectorize=True and confounds are provided.

        """
        if self.kind == "tangent" and len(X) <= 1:
            # Check that people are applying fit_transform to a group of
            # subject
            # We can only impose this in fit_transform, as it is legit to
            # fit only on a single given reference point
            raise ValueError(
                "Tangent space parametrization can only "
                "be applied to a group of subjects, as it returns "
                f"deviations to the mean. You provided {X!r}"
            )
        return self._fit_transform(
            X, do_fit=True, do_transform=True, confounds=confounds
        )

    def transform(self, X, confounds=None):
        """Apply transform to covariances matrices to get the connectivity \
        matrices for the chosen kind.

        Parameters
        ----------
        X : list of n_subjects numpy.ndarray with shapes \
            (n_samples, n_features)
            The input subjects time series. The number of samples may differ
            from one subject to another.

        confounds : numpy.ndarray with shape (n_samples) or \
                    (n_samples, n_confounds), optional
            Confounds to be cleaned on the vectorized matrices. Only takes
            into effect when vetorize=True.
            This parameter is passed to signal.clean. Please see the related
            documentation for details.

        Returns
        -------
        output : numpy.ndarray, shape (n_subjects, n_features, n_features) or \
            (n_subjects, n_features * (n_features + 1) / 2) if vectorize \
            is set to True.
            The transformed individual connectivities, as matrices or vectors.
            Vectors are cleaned when vectorize=True and confounds are provided.

        """
        self._check_fitted()
        return self._fit_transform(X, do_transform=True, confounds=confounds)

    def __sklearn_is_fitted__(self):
        return not hasattr(self, "cov_estimator_")

    def _check_fitted(self):
        if self.__sklearn_is_fitted__():
            raise ValueError(
                f"It seems that {self.__class__.__name__} "
                "has not been fitted. "
                "You must call fit() before calling transform()."
            )

    def inverse_transform(self, connectivities, diagonal=None):
        """Return connectivity matrices from connectivities, \
        vectorized or not.

        If kind is 'tangent', the covariance matrices are reconstructed.

        Parameters
        ----------
        connectivities : list of n_subjects numpy.ndarray with shapes\
            (n_features, n_features) or (n_features * (n_features + 1) / 2,)
            or ((n_features - 1) * n_features / 2,)
            Connectivities of each subject, vectorized or not.

        diagonal : numpy.ndarray, shape (n_subjects, n_features), optional
            The diagonals of the connectivity matrices.

        Returns
        -------
        output : numpy.ndarray, shape (n_subjects, n_features, n_features)
            The corresponding connectivity matrices. If kind is 'correlation'/
            'partial correlation', the correlation/partial correlation
            matrices are returned.
            If kind is 'tangent', the covariance matrices are reconstructed.

        """
        self._check_fitted()

        connectivities = np.array(connectivities)
        if self.vectorize:
            if self.discard_diagonal and diagonal is None:
                if self.kind in ["correlation", "partial correlation"]:
                    diagonal = np.ones(
                        (connectivities.shape[0], self.mean_.shape[0])
                    ) / sqrt(2.0)
                else:
                    raise ValueError(
                        "diagonal values has been discarded and are unknown "
                        f"for {self.kind} kind, "
                        "cannot reconstruct connectivity matrices."
                    )

            connectivities = vec_to_sym_matrix(
                connectivities, diagonal=diagonal
            )

        if self.kind == "tangent":
            mean_sqrt = _map_eigenvalues(np.sqrt, self.mean_)
            connectivities = [
                mean_sqrt.dot(_map_eigenvalues(np.exp, displacement)).dot(
                    mean_sqrt
                )
                for displacement in connectivities
            ]
            connectivities = np.array(connectivities)

        return connectivities
