"""Recursive Neighbor Agglomeration (ReNA).

Fastclustering for approximation of structured signals
"""
# Author: Andres Hoyos idrobo, Gael Varoquaux, Jonas Kahn and  Bertrand Thirion
# License: simplified BSD

import warnings

import numpy as np
from joblib import Memory
from nibabel import Nifti1Image
from nilearn._utils import fill_doc
from nilearn.image import get_data
from nilearn.masking import _unmask_from_to_3d_array
from scipy.sparse import coo_matrix, csgraph, dia_matrix
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def _compute_weights(X, mask_img):
    """Compute the weights in direction of each axis using Euclidean distance.

    i.e. weights = (weight_deep, weights_right, weight_down).

    Notes
    -----
    Here we assume a square lattice (no diagonal connections).

    Parameters
    ----------
    X : ndarray, shape = [n_samples, n_features]
        Training data.

    mask_img : Niimg-like object
        Object used for masking the data.

    Returns
    -------
    weights : ndarray
        Weights corresponding to all edges in the mask.
        shape: (n_edges,).

    """
    n_samples, n_features = X.shape

    mask = get_data(mask_img).astype("bool")
    shape = mask.shape

    data = np.empty((shape[0], shape[1], shape[2], n_samples))
    for sample in range(n_samples):
        data[:, :, :, sample] = _unmask_from_to_3d_array(
            X[sample].copy(), mask
        )

    weights_deep = np.sum(np.diff(data, axis=2) ** 2, axis=-1).ravel()
    weights_right = np.sum(np.diff(data, axis=1) ** 2, axis=-1).ravel()
    weights_down = np.sum(np.diff(data, axis=0) ** 2, axis=-1).ravel()

    weights = np.hstack([weights_deep, weights_right, weights_down])

    return weights


def _make_3d_edges(vertices, is_mask):
    """Create the edges set: Returns a list of edges for a 3D image.

    Parameters
    ----------
    vertices : ndarray
        The indices of the voxels.

    is_mask : boolean
        If is_mask is true, it returns the mask of edges.
        Returns 1 if the edge is contained in the mask, 0 otherwise.

    Returns
    -------
    edges : ndarray
        Edges corresponding to the image or mask.
        shape: (1, n_edges) if_mask,
               (2, n_edges) otherwise.

    """
    if is_mask:
        edges_deep = np.logical_and(
            vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()
        )
        edges_right = np.logical_and(
            vertices[:, :-1].ravel(), vertices[:, 1:].ravel()
        )
        edges_down = np.logical_and(
            vertices[:-1].ravel(), vertices[1:].ravel()
        )
    else:
        edges_deep = np.vstack(
            [vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()]
        )
        edges_right = np.vstack(
            [vertices[:, :-1].ravel(), vertices[:, 1:].ravel()]
        )
        edges_down = np.vstack([vertices[:-1].ravel(), vertices[1:].ravel()])

    edges = np.hstack([edges_deep, edges_right, edges_down])

    return edges


def _make_edges_and_weights(X, mask_img):
    """Compute the weights to all edges in the mask.

    Parameters
    ----------
    X : ndarray, shape = [n_samples, n_features]
        Training data.

    mask_img : Niimg-like object
        Object used for masking the data.

    Returns
    -------
    edges : ndarray
        Array containing [edges_deep, edges_right, edges_down]

    weights : ndarray
        Weights corresponding to all edges in the mask.
        shape: (n_edges,).

    """
    mask = get_data(mask_img)
    shape = mask.shape
    n_vertices = np.prod(shape)

    # Indexing each voxel
    vertices = np.arange(n_vertices).reshape(shape)

    weights_unmasked = _compute_weights(X, mask_img)

    edges_unmasked = _make_3d_edges(vertices, is_mask=False)
    edges_mask = _make_3d_edges(mask, is_mask=True)

    # Apply mask to edges and weights
    weights = np.copy(weights_unmasked[edges_mask])
    edges = np.copy(edges_unmasked[:, edges_mask])

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weights


def _weighted_connectivity_graph(X, mask_img):
    """Create a symmetric weighted graph.

    Data and topology are encoded by a connectivity matrix.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Training data. shape = [n_samples, n_features]

    mask_img : Niimg-like object
        Object used for masking the data.

    Returns
    -------
    connectivity : a CSR matrix
        Sparse matrix representation of the weighted adjacency graph.

    """
    n_features = X.shape[1]

    edges, weight = _make_edges_and_weights(X, mask_img)

    connectivity = coo_matrix(
        (weight, edges), (n_features, n_features)
    ).tocsr()

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def _nn_connectivity(connectivity, threshold=1e-7):
    """Fast implementation of nearest neighbor connectivity.

    Parameters
    ----------
    connectivity : a sparse matrix in COOrdinate format.
        Sparse matrix representation of the weighted adjacency graph.

    threshold : float in the close interval [0, 1], optional
        The threshold is set to handle eccentricities.
        Default=1e-7.

    Returns
    -------
    nn_connectivity : a sparse matrix in COOrdinate format.

    """
    n_features = connectivity.shape[0]

    connectivity_ = coo_matrix(
        (1.0 / connectivity.data, connectivity.nonzero()),
        (n_features, n_features),
    ).tocsr()

    # maximum on the axis = 0
    max_connectivity = connectivity_.max(axis=0).toarray()[0]
    inv_max = dia_matrix(
        (1.0 / max_connectivity, 0), shape=(n_features, n_features)
    )

    connectivity_ = inv_max * connectivity_

    # Dealing with eccentricities, there are probably many nearest neighbors
    edge_mask = connectivity_.data > 1 - threshold

    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    # Set weights to 1
    weight = np.ones_like(j_idx)
    edges = np.array([i_idx, j_idx])

    nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    return nn_connectivity


def _reduce_data_and_connectivity(
    X, labels, n_components, connectivity, threshold=1e-7
):
    """Perform feature grouping and reduce the connectivity matrix.

    During the reduction step one changes the value of each cluster
    by their mean.
    In addition, connected nodes are merged.

    Parameters
    ----------
    X : ndarray, shape = [n_samples, n_features]
        Training data.

    labels : ndarray
        Contains the label assignation for each voxel.

    n_components : int
        The number of clusters in the current iteration.

    connectivity : a sparse matrix in COOrdinate format.
        Sparse matrix representation of the weighted adjacency graph.

    threshold : float in the close interval [0, 1], optional
        The threshold is set to handle eccentricities.
        Default=1e-7.

    Returns
    -------
    reduced_connectivity : a sparse matrix in COOrdinate format.

    reduced_X : ndarray
        Data reduced with agglomerated signal for each cluster.

    """
    n_features = len(labels)

    incidence = coo_matrix(
        (np.ones(n_features), (labels, np.arange(n_features))),
        shape=(n_components, n_features),
        dtype=np.float32,
    ).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1.0 / incidence.sum(axis=1)).squeeze(), 0),
        shape=(n_components, n_components),
    )

    incidence = inv_sum_col * incidence

    reduced_X = (incidence * X.T).T
    reduced_connectivity = (incidence * connectivity) * incidence.T

    reduced_connectivity = reduced_connectivity - dia_matrix(
        (reduced_connectivity.diagonal(), 0),
        shape=(reduced_connectivity.shape),
    )

    i_idx, j_idx = reduced_connectivity.nonzero()

    weights_ = np.sum((reduced_X[:, i_idx] - reduced_X[:, j_idx]) ** 2, axis=0)
    weights_ = np.maximum(threshold, weights_)
    reduced_connectivity.data = weights_

    return reduced_connectivity, reduced_X


def _nearest_neighbor_grouping(X, connectivity, n_clusters, threshold=1e-7):
    """Cluster using nearest neighbor agglomeration.

    Merge clusters according to their nearest neighbors,
    then the data and the connectivity are reduced.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Training data. shape = [n_samples, n_features]

    connectivity : a sparse matrix in COOrdinate format.
        Sparse matrix representation of the weighted adjacency graph.

    n_clusters : :obj:`int`
        The number of clusters to find.

    threshold : :obj:`float` in the close interval [0, 1], optional
        The threshold is set to handle eccentricities.
        Default=1e-7.

    Returns
    -------
    reduced_connectivity : a sparse matrix in COOrdinate format.

    reduced_X : :class:`numpy.ndarray`
        Data reduced with agglomerated signal for each cluster.

    labels : :class:`numpy.ndarray`, shape = [n_features]
        It contains the clusters assignation.

    """
    # Nearest neighbor connectivity
    nn_connectivity = _nn_connectivity(connectivity, threshold)
    n_features = connectivity.shape[0]
    n_components = n_features - (nn_connectivity + nn_connectivity.T).nnz / 2

    if n_components < n_clusters:
        # remove edges so that the final number of clusters is not less than
        # n_clusters (to achieve the desired number of clusters)
        n_edges = n_features - n_clusters
        nn_connectivity = nn_connectivity + nn_connectivity.T

        i_idx, j_idx = nn_connectivity.nonzero()
        edges = np.array([i_idx, j_idx])

        # select n_edges to merge.
        edge_mask = np.argsort(i_idx - j_idx)[:n_edges]
        # Set weights to 1, and the connectivity matrix symmetrical.
        weight = np.ones(2 * n_edges)
        edges = np.hstack([edges[:, edge_mask], edges[::-1, edge_mask]])

        nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    # Clustering step: getting the connected components of the nn matrix
    n_components, labels = csgraph.connected_components(nn_connectivity)

    # Reduction step: reduction by averaging
    reduced_connectivity, reduced_X = _reduce_data_and_connectivity(
        X, labels, n_components, connectivity, threshold
    )

    return reduced_connectivity, reduced_X, labels


def recursive_neighbor_agglomeration(
    X, mask_img, n_clusters, n_iter=10, threshold=1e-7, verbose=0
):
    """Recursive neighbor agglomeration (:term:`ReNA`).

    It performs iteratively the nearest neighbor grouping.
    See :footcite:`Hoyos2019`.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Training data. shape = [n_samples, n_features]

    mask_img : Niimg-like object
        Object used for masking the data.

    n_clusters : :obj:`int`
        The number of clusters to find.

    n_iter : :obj:`int`, optional
        Number of iterations. Default=10.

    threshold : :obj:`float` in the close interval [0, 1], optional
        The threshold is set to handle eccentricities.
        Default=1e-7.

    verbose : :obj:`int`, optional
        Verbosity level. Default=0.

    Returns
    -------
    n_components : :obj:`int`
        Number of clusters.

    labels : :class:`numpy.ndarray`
        Cluster assignation. shape = [n_features]

    References
    ----------
    .. footbibliography::

    """
    connectivity = _weighted_connectivity_graph(X, mask_img)

    # Initialization
    labels = np.arange(connectivity.shape[0])
    n_components = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, X, reduced_labels = _nearest_neighbor_grouping(
            X, connectivity, n_clusters, threshold
        )

        labels = reduced_labels[labels]
        n_components = connectivity.shape[0]

        if verbose > 0:
            print(
                f"After iteration number {i + 1}, features are "
                f" grouped into {n_components} clusters"
            )

        if n_components <= n_clusters:
            break

    return n_components, labels


@fill_doc
class ReNA(BaseEstimator, ClusterMixin, TransformerMixin):
    """Recursive Neighbor Agglomeration (:term:`ReNA`).

    Recursively merges the pair of clusters according to 1-nearest neighbors
    criterion.
    See :footcite:`Hoyos2019`.

    Parameters
    ----------
    %(mask_img)s
    n_clusters : :obj:`int`, optional
        The number of clusters to find. Default=2.

    scaling : :obj:`bool`, optional
        If scaling is True, each cluster is scaled by the square root of its
        size, preserving the l2-norm of the image. Default=False.

    n_iter : :obj:`int`, optional
        Number of iterations of the recursive neighbor agglomeration.
        Default=10.

    threshold : :obj:`float` in the open interval (0., 1.), optional
        Threshold used to handle eccentricities.
        Default=1e-7.
    %(memory)s
    %(memory_level1)s
    %(verbose0)s

    Attributes
    ----------
    `labels_ ` : :class:`numpy.ndarray`, shape = [n_features]
        Cluster labels for each feature.

    `n_clusters_` : :obj:`int`
        Number of clusters.

    `sizes_` : :class:`numpy.ndarray`, shape = [n_features]
        It contains the size of each cluster.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        mask_img,
        n_clusters=2,
        scaling=False,
        n_iter=10,
        threshold=1e-7,
        memory=None,
        memory_level=1,
        verbose=0,
    ):
        self.mask_img = mask_img
        self.n_clusters = n_clusters
        self.scaling = scaling
        self.n_iter = n_iter
        self.threshold = threshold
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def _more_tags(self):
        return BaseEstimator._more_tags()

    def fit(self, X, y=None):
        """Compute clustering of the data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`, shape = [n_samples, n_features]
            Training data.

        y : Ignored

        Returns
        -------
        self : `ReNA` object

        """
        X = check_array(
            X, ensure_min_features=2, ensure_min_samples=2, estimator=self
        )
        n_features = X.shape[1]

        if not isinstance(self.mask_img, (str, Nifti1Image)):
            raise ValueError(
                "The mask image should be a Niimg-like"
                f"object. Instead a {type(self.mask_img)} object was provided."
            )

        if self.memory is None or isinstance(self.memory, str):
            self.memory_ = Memory(
                location=self.memory, verbose=max(0, self.verbose - 1)
            )
        else:
            self.memory_ = self.memory

        if self.n_clusters <= 0:
            raise ValueError(
                "n_clusters should be an integer greater than 0."
                f" {self.n_clusters} was provided."
            )

        if self.n_iter <= 0:
            raise ValueError(
                "n_iter should be an integer greater than 0."
                f" {self.n_iter} was provided."
            )

        if self.n_clusters > n_features:
            self.n_clusters = n_features
            warnings.warn(
                "n_clusters should be at most the number of features. "
                f"Taking n_clusters = {n_features} instead."
            )

        n_components, labels = self.memory_.cache(
            recursive_neighbor_agglomeration
        )(
            X,
            self.mask_img,
            self.n_clusters,
            n_iter=self.n_iter,
            threshold=self.threshold,
            verbose=self.verbose,
        )

        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]

        self.labels_ = labels
        self.n_clusters_ = np.unique(self.labels_).shape[0]
        self.sizes_ = sizes

        return self

    def transform(self, X, y=None):
        """Apply clustering, reduce the dimensionality of the data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`, shape = [n_samples, n_features]
            Data to transform with the fitted clustering.

        Returns
        -------
        X_red : :class:`numpy.ndarray`, shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster.

        """
        check_is_fitted(self, "labels_")

        unique_labels = np.unique(self.labels_)

        mean_cluster = [
            np.mean(X[:, self.labels_ == label], axis=1)
            for label in unique_labels
        ]
        X_red = np.array(mean_cluster).T

        if self.scaling:
            X_red = X_red * np.sqrt(self.sizes_)

        return X_red

    def inverse_transform(self, X_red):
        """Send the reduced 2D data matrix back to the original feature \
        space (:term:`voxels<voxel>`).

        Parameters
        ----------
        X_red : :class:`numpy.ndarray`, shape = [n_samples, n_clusters]
            Data reduced with agglomerated signal for each cluster.

        Returns
        -------
        X_inv : :class:`numpy.ndarray`, shape = [n_samples, n_features]
            Data reduced expanded to the original feature space.

        """
        check_is_fitted(self, "labels_")

        _, inverse = np.unique(self.labels_, return_inverse=True)

        if self.scaling:
            X_red = X_red / np.sqrt(self.sizes_)
        X_inv = X_red[..., inverse]

        return X_inv
