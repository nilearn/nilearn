import numpy as np
from scipy.linalg import pinv, eigh, inv
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets.base import Bunch
from sklearn.covariance import LedoitWolf
from nilearn.connectome.connectivity_matrices import (
    _check_spd,
    sym_matrix_to_vec,
    _geometric_mean,
    _map_eigenvalues,
)


def regularized_eigenvalue_decomposition(C, explained_variance_threshold):
    """Approximate C through eigenvalue decomposition, see notes.

    Parameters
    ----------
    C : numpy.ndarray (n_features, n_features)
        Symmetric positive definite matrix to be approximated

    explained_variance_threshold : float
        Threshold of the cumulative ratio of the variance explained by the
        components of the eigenvalue decomposition

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        alpha, eigenvalues, eigenvectors
    
    Notes
    -----
    The eigenvalue approximation of C is
    Cr = alpha*I + Q L Q.T 
    where Q is the truncated eigenvector matrix, 
    L is the diagonal matrix of the truncated eigenvalues,
    alpha is set such that trace(Cr) = trace(C)
    """
    if explained_variance_threshold > 1 or explained_variance_threshold < 0:
        raise ValueError(
            "Threshold of the explained variance eigenvalue"
            "decomposition should be between 0 and 1 instead"
            " of {}".format(explained_variance_threshold)
        )

    # eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(C)

    # select k eigenvalues that explains has variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio[::-1])
    k = np.searchsorted(cumulative_variance_ratio, 
                        explained_variance_threshold) + 1

    # Cr = alpha*I + Q L Q.T
    Cr = np.dot(eigenvalues[:k] * eigenvectors[:, :k], eigenvectors[:, :k].T)

    # set alpha s.t. trace(C) = trace(Cr)
    n_features = C.shape[0]
    alpha = (np.trace(C) - np.trace(Cr)) / n_features

    return Bunch(
        eigenvalues=eigenvalues[:k],
        eigenvectors=eigenvectors[:, :k],
        alpha=alpha
    )


def shrunk_covariance_embedding(cov_embedding, prior_cov_approx, shrinkage):
    """Population-based shrinkage of a covariance estimate in the tangent space

    Parameters
    ----------
    cov_embedding : numpy.ndarray (n_features*(n_features-1)/2, 1)
        Tangent-space estimate of the covariance to be shrunk

    prior_cov_approx : Bunch
        Eigenvalue-approximation of the prior dispersion

    shrinkage : float, 0 <= shrinkage <= 1
        coefficient of regularization

    Returns
    -------
    shrunk_cov_embedding : numpy.ndarray (n_features*(n_features-1)/2, 1)
        Shrunk tangent-space estimate of the covariance

    Notes
    -----
    The general formula of PoSCE is 
    shrunk_cov_embedding = 
    (likelihood_cov^-1 + prior_cov^-1)^-1 likelihood_cov^-1 cov_embedding
    where 
    likelihood_cov is assumed to be shrinkage*Identity
    """
    n_features = cov_embedding.shape[0]

    # use approximated prior covariance eigenvalue approximation
    alpha = prior_cov_approx.alpha
    U = prior_cov_approx.eigenvectors
    v = prior_cov_approx.eigenvalues

    # use Woodbury matrix identity to approximate :
    # (prior_covariance^-1 + likelihood_covariance^-1)^-1
    # assume likelihood_covariance = shrinkage*Identity
    coef = shrinkage / (shrinkage + alpha)
    prior_cov_inv_approx = (alpha * coef) * np.eye(n_features) + np.dot(
        coef * np.dot(U, pinv(np.diag(1 / v) + (coef / shrinkage) * np.dot(U.T, U))),
        coef * U.T,
    )
    shrunk_cov_embedding = (1.0 / shrinkage) * prior_cov_inv_approx.dot(cov_embedding)
    return shrunk_cov_embedding


class PopulationShrunkCovariance(BaseEstimator, TransformerMixin):
    """Compute population shrinkage of covariance embedding

        Parameters
        ----------
        cov_estimator : estimator object, optional
            The covariance estimator. By default the LedoitWolf estimator
            is used. This implies that correlations are slightly shrunk
            towards zero compared to a maximum-likelihood estimate
        
        prior_mean_type : {"geometric", "empirical"}, optional
            Kind of the prior to be computed

        shrinkage : float, 0 <= shrinkage <= 1
            coefficient of regularization            

        explained_variance_threshold : float, optional
            Threshold of the cumulative ratio of the variance explained by the
            components of the eigenvalue decomposition

        Attributes
        ----------
        `cov_estimator_` : estimator object
            A new covariance estimator with the same parameters as cov_estimator.
    
        `prior_mean_` : numpy.ndarray
            The prior mean in the covariance space.

        `prior_whitening_` : numpy.ndarray
            The inverted square-rooted geometric mean of the covariance matrices.

        `prior_cov_` : numpy.ndarray
            The dispersion of the prior.

        References
        ----------
        M. Rahim, B. Thirion and G. Varoquaux. Population
        shrinkage of covariance (PoSCE) for better individual brain
        functional-connectivity estimation, in Medical Image Analysis (2019).
    """

    def __init__(
        self,
        cov_estimator=LedoitWolf(store_precision=False),
        prior_mean_type="geometric",
        shrinkage=0.5,
        explained_variance_threshold=0.7,
    ):
        self.cov_estimator = cov_estimator
        self.prior_mean_type = prior_mean_type
        self.shrinkage = shrinkage
        self.explained_variance_threshold = explained_variance_threshold

    def fit(self, X, y=None):
        """Fit PoSCE to the given time series for each subject

        Parameters
        ----------
        X : list of n_subjects numpy.ndarray, shapes (n_samples, n_features)
            The input subjects time series. The number of samples may differ
            from one subject to another

        Returns
        -------
        self : PopulationShrunkCovariance instance
            The object itself. Useful for chaining operations.
        """
        # compute covariances from timeseries
        self.cov_estimator_ = clone(self.cov_estimator)
        covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]

        # compute prior mean
        if self.prior_mean_type == "geometric":
            self.prior_mean_ = _geometric_mean(covariances, 
                                               max_iter=30, 
                                               tol=1e-7)
        elif self.prior_mean_type == "empirical":
            self.prior_mean_ = np.mean(covariances, axis=0)
        else:
            raise ValueError(
                "Allowed mean types are"
                '"geometric", "euclidean"'
                ', got type "{}"'.format(self.prior_mean_type)
            )
        self.prior_whitening_ = _map_eigenvalues(
            lambda x: 1.0 / np.sqrt(x), self.prior_mean_
        )
        self.prior_whitening_inv_ = _map_eigenvalues(
            lambda x: np.sqrt(x), self.prior_mean_
        )

        # compute the population prior dispersion
        connectivities = [
            _map_eigenvalues(
                np.log, 
                self.prior_whitening_.dot(cov).dot(self.prior_whitening_)
            )
            for cov in covariances
        ]
        connectivities = np.array(connectivities)
        connectivities = sym_matrix_to_vec(connectivities)
        self.prior_cov_ = np.mean(
            [np.expand_dims(c, 1).dot(np.expand_dims(c, 0)) 
             for c in connectivities],
            axis=0,
        )
        # approximate the population prior dispersion
        self.prior_cov_approx_ = regularized_eigenvalue_decomposition(
            self.prior_cov_, 
            explained_variance_threshold=0.7
        )
        return self

    def transform(self, X):
        """Transform subjects timeseries to shrunk covariances in the tangent 
        space using the population prior.

        Parameters
        ----------
        X : list of n_subjects numpy.ndarray, shapes (n_samples, n_features)
            The input subjects time series. The number of samples may differ
            from one subject to another

        Returns
        -------
        shrunk_connectivities : numpy.ndarray, shape 
        (n_subjects, n_features * (n_features + 1) / 2).
            Shrunk individual connectivities as vectors.
        """
        # compute covariances from timeseries
        covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]

        # transform in the tangent space
        connectivities = [
            _map_eigenvalues(
                np.log, 
                self.prior_whitening_.dot(cov).dot(self.prior_whitening_)
            )
            for cov in covariances
        ]

        connectivities = np.array(connectivities)
        connectivities = sym_matrix_to_vec(connectivities)

        shrunk_connectivities = [
            shrunk_covariance_embedding(
                cov_embedding=c,
                prior_cov_approx=self.prior_cov_approx_,
                shrinkage=self.shrinkage,
            )
            for c in connectivities
        ]
        return np.array(shrunk_connectivities)
