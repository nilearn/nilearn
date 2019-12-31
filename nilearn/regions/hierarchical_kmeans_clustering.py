import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, ClusterMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def _remove_empty_labels(labels):
    '''Remove empty values label values from labels list'''
    vals = np.unique(labels)
    inverse_vals = - np.ones(labels.max() + 1).astype(np.int)
    inverse_vals[vals] = np.arange(len(vals))
    return inverse_vals[labels]


def _hierarchical_k_means(X, n_clusters, init="k-means++", batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0, random_state=0):
    """ Use a recursive k-means to cluster X
    Parameters
    ----------
    X: nd array (n_samples, n_features)
        Data to cluster

    n_clusters: int,
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    batch_size : int, optional, default: 100
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """

    n_big_clusters = int(np.sqrt(n_clusters))
    mbk = MiniBatchKMeans(init=init, n_clusters=n_big_clusters, batch_size=batch_size,
                          n_init=n_init, max_no_improvement=max_no_improvement, verbose=verbose,
                          random_state=random_state).fit(X)
    coarse_labels = mbk.labels_
    fine_labels = np.zeros_like(coarse_labels)
    q = 0
    for i in range(n_big_clusters):
        n_small_clusters = int(
            n_clusters * np.sum(coarse_labels == i) * 1. / X.shape[0])
        n_small_clusters = np.maximum(1, n_small_clusters)
        mbk = MiniBatchKMeans(init=init, n_clusters=n_small_clusters,
                              batch_size=batch_size, n_init=n_init,
                              max_no_improvement=max_no_improvement, verbose=verbose,
                              random_state=random_state).fit(X[coarse_labels == i])
        fine_labels[coarse_labels == i] = q + mbk.labels_
        q += n_small_clusters

    return _remove_empty_labels(fine_labels)


class HierarchicalKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Hierarchical KMeans:
    First clusterize the samples into big clusters. Then clusterize the samples
    inside these big clusters into smaller ones.

    Parameters
    ----------
    n_clusters: int
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    batch_size : int, optional, default: 100
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    scaling: bool, optional (default False)
        If scaling is True, each cluster is scaled by the square root of its
        size, preserving the l2-norm of the image.

    verbose: int, optional (default 0)
        Verbosity level.

    Attributes
    ----------
    `labels_ `: ndarray, shape = [n_features]
        cluster labels for each feature.

    `sizes_`: ndarray, shape = [n_features]
        It contains the size of each cluster.

    """

    def __init__(self, n_clusters, init="k-means++", batch_size=1000,
                 n_init=10, max_no_improvement=10, verbose=0, random_state=0, scaling=False):
        self.n_clusters = n_clusters
        self.init = init
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose
        self.random_state = random_state
        self.scaling = scaling

    def fit(self, X, y=None):
        """Compute clustering of the data.

        Parameters
        ----------
        X: ndarray, shape = [n_samples, n_features]
            Training data.
        y: Ignored

        Returns
        -------
        self
        """

        X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
                        estimator=self)
        n_features = X.shape[1]

        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        if self.n_clusters > n_features:
            self.n_clusters = n_features
            warnings.warn("n_clusters should be at most the number of "
                          "features. Taking n_clusters = %s instead."
                          % str(n_features))
        self.labels_ = _hierarchical_k_means(X, self.n_clusters, self.init, self.batch_size,
                                             self.n_init, self.max_no_improvement, self.verbose, self.random_state)
        sizes = np.bincount(self.labels_)
        sizes = sizes[sizes > 0]
        self.sizes_ = sizes
        self.n_clusters = len(np.unique(self.labels_))
        return self

    def transform(self, X, y=None):
        """Apply clustering, reduce the dimensionality of the data.

        Parameters
        ----------
        X: ndarray, shape = [n_samples, n_features]
            Data to transform with the fitted clustering.

        Returns
        -------
        X_red: ndarray, shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster
        """

        #check_is_fitted(self, "labels_")
        unique_labels = np.unique(self.labels_)

        mean_cluster = []
        for label in unique_labels:
            mean_cluster.append(np.mean(X[:, self.labels_ == label], axis=1))

        X_red = np.array(mean_cluster).T

        if self.scaling:
            X_red = X_red * np.sqrt(self.sizes_)

        return X_red

    def inverse_transform(self, X_red):
        """Send the reduced 2D data matrix back to the original feature
        space (voxels).

        Parameters
        ----------
        X_red: ndarray , shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster

        Returns
        -------
        X_inv: ndarray, shape = [n_samples, n_features]
            Data reduced expanded to the original feature space
        """

        check_is_fitted(self, "labels_")

        _, inverse = np.unique(self.labels_, return_inverse=True)

        if self.scaling:
            X_red = X_red / np.sqrt(self.sizes_)
        X_inv = X_red[..., inverse]

        return X_inv
