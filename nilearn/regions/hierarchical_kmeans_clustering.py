"""Hierarchical k-means clustering."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from nilearn._utils.tags import SKLEARN_LT_1_6


def _remove_empty_labels(labels):
    """Remove empty values label values from labels list.

    Returns labels mapped to np.arange(n_unique),
    where n_unique is the number of unique values in labels
    """
    vals = np.unique(labels)
    inverse_vals = -np.ones(labels.max() + 1, dtype=int)
    inverse_vals[vals] = np.arange(len(vals))
    return inverse_vals[labels]


def _adjust_small_clusters(array, n_clusters):
    """Take a ndarray of floats summing to n_clusters \
    and try to round it while enforcing rounded array still sum \
    to n_clusters and every element is at least 1.
    """
    array_round = np.rint(array).astype(int)
    array_round = np.maximum(array_round, 1)

    if np.sum(array_round) < n_clusters:
        while np.sum(array_round) != n_clusters:
            idx = np.argmax(array - array_round)
            array_round[idx] += 1
    elif np.sum(array_round) == n_clusters:
        pass
    elif np.sum(array_round) > n_clusters:
        parent_idx_ = np.arange(array_round.shape[0])
        while np.sum(array_round) != n_clusters:
            # prevent element rounded to 1 to be decreased in edge cases
            mask = array_round != 1
            idx = np.argmin(array[mask] - array_round[mask])
            parent_idx = parent_idx_[mask][idx]
            array_round[parent_idx] -= 1
    return array_round


def hierarchical_k_means(
    X,
    n_clusters,
    init="k-means++",
    batch_size=1000,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
    random_state=0,
):
    """Use a recursive k-means to cluster X.

    First clustering in sqrt(n_clusters) parcels,
    and Kmeans a second time on each parcel.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Data to cluster

    n_clusters : int,
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}, default='k-means++'
        Method for initialization.
        'k-means++' : selects initial cluster centers for k-means
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    batch_size : int, optional, default: 1000
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=10
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

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    n_big_clusters = int(np.sqrt(n_clusters))
    mbk = MiniBatchKMeans(
        init=init,
        n_clusters=n_big_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=max_no_improvement,
        verbose=verbose,
        random_state=random_state,
    ).fit(X)
    coarse_labels = mbk.labels_
    fine_labels = np.zeros_like(coarse_labels)
    q = 0
    counts = np.bincount(coarse_labels)
    exact_clusters = np.asarray(
        [
            n_clusters * counts[i] * 1.0 / X.shape[0]
            for i in range(n_big_clusters)
        ]
    )

    adjusted_clusters = _adjust_small_clusters(exact_clusters, n_clusters)
    for i, n_small_clusters in enumerate(adjusted_clusters):
        mbk = MiniBatchKMeans(
            init=init,
            n_clusters=n_small_clusters,
            batch_size=batch_size,
            random_state=random_state,
            max_no_improvement=max_no_improvement,
            verbose=verbose,
            n_init=n_init,
        ).fit(X[coarse_labels == i])
        fine_labels[coarse_labels == i] = q + mbk.labels_
        q += n_small_clusters

    return _remove_empty_labels(fine_labels)


class HierarchicalKMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Hierarchical KMeans.

    First clusterize the samples into big clusters. Then clusterize the samples
    inside these big clusters into smaller ones.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}, default='k-means++'
        Method for initialization.

        * 'k-means++' : selects initial cluster centers for k-means
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.

        * 'random': choose k observations (rows) at random from data for
          the initial centroids.

        * If an ndarray is passed, it should be of shape (n_clusters,
          n_features) and gives the initial centers.

    batch_size : int, optional, default: 1000
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=10
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

    scaling : bool, optional (default False)
        If scaling is True, each cluster is scaled by the square root of its
        size during transform(), preserving the l2-norm of the image.
        inverse_transform() will apply inversed scaling to yield an image with
        same l2-norm as input.

    verbose : int, optional (default 0)
        Verbosity level.

    Attributes
    ----------
    labels_ : ndarray, shape = [n_features]
        cluster labels for each feature.

    sizes_ : ndarray, shape = [n_features]
        It contains the size of each cluster.

    """

    def __init__(
        self,
        n_clusters,
        init="k-means++",
        batch_size=1000,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
        random_state=0,
        scaling=False,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose
        self.random_state = random_state
        self.scaling = scaling

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags()

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags()
        return tags

    def fit(
        self,
        X,
        y=None,  # noqa: ARG002
    ):
        """Compute clustering of the data.

        Parameters
        ----------
        X : ndarray, shape = [n_samples, n_features]
            Training data.
        y : Ignored

        Returns
        -------
        self
        """
        X = check_array(
            X, ensure_min_features=2, ensure_min_samples=2, estimator=self
        )
        # Transpose the data so that we can cluster features (voxels)
        # and input them as samples to the sklearn's clustering algorithm
        # This is because sklearn's clustering algorithm does clustering
        # on samples and not on features
        X = X.T
        # n_features for the sklearn's clustering algorithm would be the
        # number of samples in the input data
        n_features = X.shape[1]

        if self.n_clusters <= 0:
            raise ValueError(
                "n_clusters should be an integer greater than 0."
                f" {self.n_clusters} was provided."
            )

        if self.n_clusters > n_features:
            self.n_clusters = n_features
            warnings.warn(
                "n_clusters should be at most the number of "
                f"features. Taking n_clusters = {n_features} instead.",
                stacklevel=2,
            )
        self.labels_ = hierarchical_k_means(
            X,
            self.n_clusters,
            self.init,
            self.batch_size,
            self.n_init,
            self.max_no_improvement,
            self.verbose,
            self.random_state,
        )
        sizes = np.bincount(self.labels_)

        self.sizes_ = sizes
        self.n_clusters = len(sizes)
        return self

    def transform(
        self,
        X,
        y=None,  # noqa: ARG002
    ):
        """Apply clustering, reduce the dimensionality of the data.

        Parameters
        ----------
        X : ndarray, shape = [n_samples, n_features]
            Data to transform with the fitted clustering.

        Returns
        -------
        X_red : ndarray, shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster
        """
        check_is_fitted(self, "labels_")

        # Transpose the data so that we can cluster features (voxels)
        # and input them as samples to the sklearn's clustering algorithm
        X = X.T
        unique_labels = np.arange(self.n_clusters)

        mean_cluster = np.empty(
            (len(unique_labels), X.shape[1]), dtype=X.dtype
        )
        for label in unique_labels:
            mean_cluster[label] = np.mean(X[self.labels_ == label], axis=0)

        X_red = np.array(mean_cluster)

        if self.scaling:
            X_red = X_red * np.sqrt(self.sizes_[:, np.newaxis])

        # Transpose the data back to the original shape i.e.
        # (n_samples, n_clusters)
        X_red = X_red.T
        return X_red

    def inverse_transform(self, X_red):
        """Send the reduced 2D data matrix back to the original feature \
        space (voxels).

        Parameters
        ----------
        X_red : ndarray , shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster

        Returns
        -------
        X_inv : ndarray, shape = [n_samples, n_features]
            Data reduced expanded to the original feature space
        """
        check_is_fitted(self, "labels_")
        X_red = X_red.T
        inverse = self.labels_
        if self.scaling:
            X_red = X_red / np.sqrt(self.sizes_[:, np.newaxis])
        X_inv = X_red[inverse, ...]
        X_inv = X_inv.T
        return X_inv
