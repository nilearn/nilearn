"""Recursive nearest agglomeration (ReNA):
    fastclustering for approximation of structured signals
"""
import numpy as np
import warnings
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.base import BaseEstimator, clone
from .._utils.fixes import check_array, check_is_fitted
from ..input_data import NiftiMasker, MultiNiftiMasker


def _compute_weights(masker, data_matrix):
    """Compute the weights in the direction of each axis using the Euclidean
    distance.

    Note: Here we are assuming a square lattice (no diagonal connections)
    """
    dims = len(masker.mask_img_.shape)
    data_graph = masker.inverse_transform(data_matrix).get_data()
    weights = []

    for axis in range(dims):
        weights.append(
            np.sum(np.diff(data_graph, axis=axis) ** 2, axis=-1).ravel())

    return np.hstack(weights)


def _compute_edges(data_graph, is_mask=False):
    """
    """
    dims = len(data_graph.shape)
    edges = []
    for axis in range(dims):
        vertices_axis = np.swapaxes(data_graph, 0, axis)

        if is_mask:
            edges.append(np.logical_and(
                vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                vertices_axis[1:].swapaxes(axis, 0).ravel()))
        else:
            edges.append(
                np.vstack([vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                           vertices_axis[1:].swapaxes(axis, 0).ravel()]))
    edges = np.hstack(edges)

    return edges


def _create_ordered_edges(masker, data_matrix):
    """
    """
    mask = masker.mask_img_.get_data()
    shape = mask.shape
    n_features = np.prod(shape)

    vertices = np.arange(n_features).reshape(shape)
    weights = _compute_weights(masker, data_matrix)
    edges = _compute_edges(vertices)
    edges_mask = _compute_edges(mask, is_mask=True)

    # Apply the mask
    weights = weights[edges_mask]
    edges = edges[:, edges_mask]

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weights, edges_mask


def weighted_connectivity_graph(masker, data_matrix):
    """ Creating weighted graph

    data and topology, encoded by a connectivity matrix

    """
    n_features = masker.mask_img_.get_data().sum()

    edges, weight, edges_mask = _create_ordered_edges(masker, data_matrix)
    connectivity = coo_matrix(
        (weight, edges), (n_features, n_features)).tocsr()

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def _nn_connectivity(connectivity, threshold):
    """ Fast implementation of nearest neighbor connectivity

    connectivity: weighted connectivity matrix
    """
    n_features = connectivity.shape[0]

    connectivity_ = coo_matrix(
        (1. / connectivity.data, connectivity.nonzero()),
        (n_features, n_features)).tocsr()

    inv_max = dia_matrix((1. / connectivity_.max(axis=0).toarray()[0], 0),
                         shape=(n_features, n_features))

    connectivity_ = inv_max * connectivity_

    # Dealing with eccentricities
    edge_mask = connectivity_.data > 1 - threshold

    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    weight = np.ones_like(j_idx)
    edges = np.array((i_idx, j_idx))

    nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    return nn_connectivity


def reduce_data_and_connectivity(labels, n_labels, connectivity, data_matrix,
                                 threshold):
    """
    """
    n_features = len(labels)

    incidence = coo_matrix(
        (np.ones(n_features), (labels, np.arange(n_features))),
        shape=(n_labels, n_features), dtype=np.float32).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
        shape=(n_labels, n_labels))

    incidence = inv_sum_col * incidence

    # reduced data
    reduced_data_matrix = (incidence * data_matrix.T).T
    reduced_connectivity = (incidence * connectivity) * incidence.T

    reduced_connectivity = reduced_connectivity - dia_matrix(
        (reduced_connectivity.diagonal(), 0),
        shape=(reduced_connectivity.shape))

    i_idx, j_idx = reduced_connectivity.nonzero()

    data_matrix_ = np.maximum(threshold, np.sum(
        (reduced_data_matrix[:, i_idx] - reduced_data_matrix[:, j_idx]) ** 2, 0))
    reduced_connectivity.data = data_matrix_

    return reduced_connectivity, reduced_data_matrix


def nearest_neighbor_grouping(connectivity, data_matrix, n_clusters,
                              threshold):
    """ Cluster according to nn and reduce the data and connectivity
    """
    # Nearest neighbor conenctivity
    nn_connectivity = _nn_connectivity(connectivity, threshold)
    n_features = connectivity.shape[0]

    n_labels = n_features - (nn_connectivity + nn_connectivity.T).nnz / 2

    if n_labels < n_clusters:
        # cut some links to achieve the desired number of clusters
        alpha = n_features - n_clusters

        nn_connectivity = nn_connectivity + nn_connectivity.T

        edges_ = np.array(nn_connectivity.nonzero())

        plop = edges_[0] - edges_[1]

        select = np.argsort(plop)[:alpha]

        nn_connectivity = coo_matrix(
            (np.ones(2 * alpha),
             np.hstack((edges_[:, select], edges_[::-1, select]))),
            (n_features, n_features))

    # Clustering step: getting the connected components of the nn matrix
    n_labels, labels = csgraph.connected_components(nn_connectivity)

    # Reduction step: reduction by averaging
    reduced_connectivity, reduced_data_matrix = reduce_data_and_connectivity(
        labels, n_labels, connectivity, data_matrix, threshold)

    return reduced_connectivity, reduced_data_matrix, labels


def recursive_nearest_agglomeration(masker, data_matrix, n_clusters, n_iter,
                                    threshold):
    """
    """
    # Weighted connectivity matrix
    connectivity = weighted_connectivity_graph(masker, data_matrix)

    # Initialization
    labels = np.arange(connectivity.shape[0])
    n_labels = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, data_matrix, reduced_labels = nearest_neighbor_grouping(
            connectivity, data_matrix, n_clusters, threshold)

        labels = reduced_labels[labels]
        n_labels = connectivity.shape[0]

        if n_labels <= n_clusters:
            break

    return n_labels, labels


class ReNA(BaseEstimator):
    """
    Recursive nearest agglomeration.
    Recursively merges the pair of clusters according to 1-nearest neighbors
    criterion.

    Parameters
    ----------
    mask : filename, niimg, NiftiMasker instance, optional default None)
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is it will be computed
        automatically by a NiftiMasker.

    n_cluster: int, optional (default 2)
        The number of clusters to find.

    scaling: bool, optional (default False)

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose : int, optional (default 1)
        Verbosity level.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine : 3x3 or 4x4 matrix, optional (default None)
        This parameter is passed to image.resample_img. An important use-case
        of this parameter is for downsampling the input data to a coarser
        resolution (to speed of the model fit). Please see the related
        documentation for details.

    target_shape : 3-tuple of integers, optional (default None)
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    n_iter: int, optional (default 10)
        Number of iterations of the recursive nearest agglomeration

    threshold: float in the opened interval (0., 1.), optional (default 1e-7)
        Threshold used to handle eccentricities.

    Attributes
    ----------
    `masker_` : instance of NiftiMasker
        The nifti masker used to mask the data.

    `mask_img_` : Nifti like image
        The mask of the data. If no mask was supplied by the user,
        this attribute is the mask image computed automatically from the
        data `X`.

    `labels_ ` : array-like, (n_features,)
        cluster labels for each feature.

    `n_clusters_` : int
        Number of clusters

    `sizes_`: array-like (n_features,)
        It contains the size of each cluster

    """
    def __init__(self, n_clusters=2, mask=None, smoothing_fwhm=None,
                 standardize=True, target_affine=None, target_shape=None,
                 mask_strategy='background', memory=None, memory_level=0,
                 verbose=False, scaling=False, n_iter=10, threshold=1e-7,):
        self.n_clusters = n_clusters
        self.scaling = scaling
        self.n_iter = n_iter
        self.mask = mask
        self.threshold = threshold
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute clustering of the data

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        self : `ReNA` object
        """

        if self.memory is None or isinstance(self.memory, six.string_types):
            self.memory_ = Memory(cachedir=self.memory,
                                  verbose=max(0, self.verbose - 1))
        else:
            self.memory_ = self.memory

        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        self.masker_ = _check_masking(self.mask, self.smoothing_fwhm,
                                      self.target_affine, self.target_shape,
                                      self.standardize, self.mask_strategy,
                                      self.memory_, self.memory_level)

        X = self.masker_.fit_transform(X)

        X = check_array(X, ensure_min_features=2)

        n_labels, labels = self.memory_.cache(recursive_nearest_agglomeration)(
            self.masker_, X, self.n_clusters, n_iter=self.n_iter,
            threshold=self.threshold)

        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]

        self.labels_ = labels
        self.n_clusters_ = np.unique(self.labels_).shape[0]
        self.sizes_ = sizes

        return self

    def transform(self, X):
        """Apply clustering, reduce the dimensionality of the data

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        X : array

        """

        check_is_fitted(self, "masker_")
        check_is_fitted(self, "labels_")

        X = self.masker_.transform(X)

        unique_labels = np.unique(self.labels_)

        nX = []
        for l in unique_labels:
            nX.append(np.mean(X[:, self.labels_ == l], axis=1))
        Xred = np.array(nX).T

        if self.scaling:
            Xred = Xred * np.sqrt(self.sizes_)

        return Xred

    def fit_transform(self, X, y=None):
        """Fit to data, then perform the clustering (transformation)

        Parameters
        ----------
        X : list of Niimg-like objects
            See http://nilearn.github.io/manipulating_images/input_output.html
            Data on which model is to be fitted. If this is a list,
            the affine is considered the same for all.

        Returns
        -------
        X : array

        """

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Xred):
        """
        Parameters
        ----------
        Xred : array like
        """

        check_is_fitted(self, "labels_")

        _, inverse = np.unique(self.labels_, return_inverse=True)

        if self.scaling:
            Xred = Xred / np.sqrt(self.sizes_)
        return Xred[..., inverse]


# XXX this code is also replicated in the Metaestimator PR
def _check_masking(mask, smoothing_fwhm, target_affine, target_shape,
                   standardize, mask_strategy, memory, memory_level):
    """Setup a nifti masker."""
    # mask is an image, not a masker
    if mask is None or isinstance(mask, six.string_types):
        masker = NiftiMasker(mask_img=mask,
                             smoothing_fwhm=smoothing_fwhm,
                             target_affine=target_affine,
                             target_shape=target_shape,
                             standardize=standardize,
                             mask_strategy=mask_strategy,
                             memory=memory,
                             memory_level=memory_level)
    # mask is a masker object
    elif isinstance(mask, (NiftiMasker, MultiNiftiMasker)):
        try:
            masker = clone(mask)
            if hasattr(mask, 'mask_img_'):
                mask_img = mask.mask_img_
                masker.set_params(mask_img=mask_img)
                masker.fit()
        except TypeError as e:
            # Workaround for a joblib bug: in joblib 0.6, a Memory object
            # with cachedir = None cannot be cloned.
            masker_memory = mask.memory
            if masker_memory.cachedir is None:
                mask.memory = None
                masker = clone(mask)
                mask.memory = masker_memory
                masker.memory = Memory(cachedir=None)
            else:
                # The error was raised for another reason
                raise e

        for param_name in ['target_affine', 'target_shape',
                           'smoothing_fwhm', 'mask_strategy',
                           'memory', 'memory_level']:
            if getattr(mask, param_name) is not None:
                warnings.warn('Parameter %s of the masker overriden'
                              % param_name)
                masker.set_params(**{param_name: getattr(mask, param_name)})
        if hasattr(mask, 'mask_img_'):
            warnings.warn('The mask_img_ of the masker will be copied')
    return masker


