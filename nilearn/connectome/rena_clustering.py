"""Recursive nearest agglomeration (ReNA):
    fastclustering for approximation of structured signals
"""
# Author: Andres Hoyos idrobo, Gael Varoquaux, Jonas Kahn and  Bertrand Thirion
# License: simplified BSD

import numpy as np
import warnings
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from sklearn.base import TransformerMixin, ClusterMixin
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.base import BaseEstimator, clone
from .._utils.fixes import check_array, check_is_fitted
from ..input_data import NiftiMasker, MultiNiftiMasker


def _compute_weights(masker, masked_data):
    """Compute the weights in the direction of each axis using the Euclidean
    distance --i.e. weights = (weight_deep, weights_right, weight_down).

    Note: Here we assume a square lattice (no diagonal connections).

    Parameters
    ----------
    masker : NiftiMasker
        The nifti masker used to mask the data.

    masked_data : numpy array of shape [n_samples, n_features]
        Image in brain space transformed into 2D data matrix.

    Returns
    -------
    weights : numpy array
        Weights corresponding to all edges in the mask.
        shape: (n_edges,)
    """
    data = masker.inverse_transform(masked_data).get_data()

    weights_deep = np.sum(np.diff(data, axis=2) ** 2, axis=-1).ravel()
    weights_right = np.sum(np.diff(data, axis=1) ** 2, axis=-1).ravel()
    weights_down = np.sum(np.diff(data, axis=0) ** 2, axis=-1).ravel()

    weights = np.hstack([weights_deep, weights_right, weights_down])

    return weights


def _make_edges_3d(vertices, is_mask):
    """Create the edges set: Returns a list of edges for a 3D image.

    Parameters
    ----------
    vertices : numpy.ndarray
        The indices of the voxels.

    is_mask : boolean
        If is_mask is true, it returns the mask of edges.
        Retruns 1 if the edge is contained in the mask, 0 otherwise.

    Returns
    -------
    edges : numpy array
        Edges corresponding to the image or mask.
        shape: (1, n_edges) if_mask,
               (2, n_edges) otherwise.
    """

    if is_mask:
        edges_deep = np.logical_and(vertices[:, :, :-1].ravel(),
                                    vertices[:, :, 1:].ravel())
        edges_right = np.logical_and(vertices[:, :-1].ravel(),
                                     vertices[:, 1:].ravel())
        edges_down = np.logical_and(vertices[:-1].ravel(),
                                    vertices[1:].ravel())
    else:
        edges_deep = np.vstack([vertices[:, :, :-1].ravel(),
                                vertices[:, :, 1:].ravel()])
        edges_right = np.vstack([vertices[:, :-1].ravel(),
                                 vertices[:, 1:].ravel()])
        edges_down = np.vstack([vertices[:-1].ravel(),
                                vertices[1:].ravel()])

    edges = np.hstack([edges_deep, edges_right, edges_down])

    return edges


def _make_edges_and_weights(masker, masked_data):
    """Compute the weights to all edges in the mask.

    Parameters
    ----------
    masker : NiftiMasker
        The nifti masker used to mask the data.

    masked_data : numpy array of shape [n_samples, n_features]
        Image in brain space transformed into 2D data matrix.

    Returns
    -------
    edges : numpy array

    weights : numpy array

    edges_mask : numpy array
    """
    mask = masker.mask_img_.get_data()
    shape = mask.shape
    n_features = np.prod(shape)

    # Indexing each voxel
    vertices = np.arange(n_features).reshape(shape)

    weights = _compute_weights(masker, masked_data)

    edges = _make_edges_3d(vertices, is_mask=False)
    edges_mask = _make_edges_3d(mask, is_mask=True)

    # Apply mask to edges and weights
    weights = weights[edges_mask]
    edges = edges[:, edges_mask]

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weights, edges_mask


def weighted_connectivity_graph(masker, masked_data):
    """ Creating weighted graph: data and topology are encoded by a
    connectivity matrix.

    Parameters
    ----------
    masker : NiftiMasker instance

    masked_data : numpy array of shape [n_samples, n_features]
        Image in brain space transformed into 2D data matrix

    Returns
    -------
    connectivity : a sparse COO matrix
    """
    n_features = masker.mask_img_.get_data().sum()

    edges, weight, edges_mask = _make_edges_and_weights(masker, masked_data)

    connectivity = coo_matrix((weight, edges),
                              (n_features, n_features)).tocsr()

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def _nn_connectivity(connectivity, threshold):
    """ Fast implementation of nearest neighbor connectivity

    Parameters
    ----------
    connectivity : a sparse matrix in COOrdinate format.
        Weighted connectivity matrix

    threshold : float in the close interval [0, 1]
        The treshold is setted to handle eccentricities.
        In practice it is 1e-7.

    Returns
    -------
    nn_connectivity : a sparse matrix in COOrdinate format.
    """
    n_features = connectivity.shape[0]

    connectivity_ = coo_matrix(
        (1. / connectivity.data, connectivity.nonzero()),
        (n_features, n_features)).tocsr()

    inv_max = dia_matrix((1. / connectivity_.max(axis=0).toarray()[0], 0),
                         shape=(n_features, n_features))

    connectivity_ = inv_max * connectivity_

    # Dealing with eccentricities, there are probably many neares neighbors
    edge_mask = connectivity_.data > 1 - threshold

    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    # Set weights to 1
    weight = np.ones_like(j_idx)
    edges = np.array((i_idx, j_idx))

    nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    return nn_connectivity


def _reduce_data_and_connectivity(labels, n_labels, connectivity, masked_data,
                                  threshold):
    """Perform feature grouping and reduce the connectivity matrix: during the
    reduction step one changes the value of each cluster by their mean.
    In addition, connected nodes are merged.

    Parameters
    ----------
    labels : array like
        Containts the label assignation for each voxel.

    n_labels : int
        The number of clusters in the current iteration.

    connectivity : a sparse matrix in COOrdinate format.

    masked_data : array like
        2D data matrix.

    threshold : float in the close interval [0, 1]
        The treshold is setted to handle eccentricities.
        In practice it is 1e-7.

    Returns
    -------
    reduced_connectivity : a sparse matrix in COOrdinate format.

    reduced_masked_data: array like
        2D data matrix.
    """
    n_features = len(labels)

    incidence = coo_matrix(
        (np.ones(n_features), (labels, np.arange(n_features))),
        shape=(n_labels, n_features), dtype=np.float32).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
        shape=(n_labels, n_labels))

    incidence = inv_sum_col * incidence

    reduced_masked_data = (incidence * masked_data.T).T
    reduced_connectivity = (incidence * connectivity) * incidence.T

    reduced_connectivity = reduced_connectivity - dia_matrix(
        (reduced_connectivity.diagonal(), 0),
        shape=(reduced_connectivity.shape))

    i_idx, j_idx = reduced_connectivity.nonzero()

    weights_ = np.sum(
        (reduced_masked_data[:, i_idx] - reduced_masked_data[:, j_idx]) ** 2,
        axis=0)
    weights_ = np.maximum(threshold, weights_)
    reduced_connectivity.data = weights_

    return reduced_connectivity, reduced_masked_data


def nearest_neighbor_grouping(connectivity, masked_data, n_clusters,
                              threshold):
    """Cluster using nearest agglomeration: merge clusters according to their
    nearest neighbors, then the data and the connectivity are reduced.

    Parameters
    ----------
    connectivity : a sparse matrix in COOrdinate format.
        Weighted connectivity matrix

    masked_data : numpy array of shape [n_samples, n_features]
        Image in brain space transformed into 2D data matrix

    n_clusters : int
        The number of clusters to find.

    threshold : float in the close interval [0, 1]
        The treshold is setted to handle eccentricities.
        In practice it is 1e-7.

    Returns
    -------
    reduced_connectivity : a sparse matrix in COOrdinate format.

    reduced_masked_data :  array like
        2D data matrix.

    labels : array like
        It contains the clusters assignation.
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
    reduced_connectivity, reduced_masked_data = _reduce_data_and_connectivity(
        labels, n_labels, connectivity, masked_data, threshold)

    return reduced_connectivity, reduced_masked_data, labels


def recursive_nearest_agglomeration(masker, masked_data, n_clusters, n_iter,
                                    threshold):
    """Recursive nearest agglomeration: it performs iteratively the nearest
    neighbor grouping.

    Parameters
    ----------
    masker : NiftiMasker instance

    masked_data : numpy array of shape [n_samples, n_features]
        Image in brain space transformed into 2D data matrix

    n_clusters : int
        The number of clusters to find.

    n_iter : int
        Number of iterations.

    threshold : float in the close interval [0, 1]
        The treshold is setted to handle eccentricities.
        In practice it is 1e-7.

    Returns
    -------
    n_labels : int
        Number of clusters.

    labels : array
        Cluster assignation.
    """
    connectivity = weighted_connectivity_graph(masker, masked_data)

    # Initialization
    labels = np.arange(connectivity.shape[0])
    n_labels = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, masked_data, reduced_labels = nearest_neighbor_grouping(
            connectivity, masked_data, n_clusters, threshold)

        labels = reduced_labels[labels]
        n_labels = connectivity.shape[0]

        if n_labels <= n_clusters:
            break

    return n_labels, labels


class ReNA(BaseEstimator, ClusterMixin, TransformerMixin):
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

    n_clusters : int, optional (default 2)
        The number of clusters to find.

    scaling : bool, optional (default False)

    memory : instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional (default 1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose : int, optional (default 1)
        Verbosity level.

    smoothing_fwhm : float, optional
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

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : float, optional (default None)
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    n_iter : int, optional (default 10)
        Number of iterations of the recursive nearest agglomeration

    threshold : float in the opened interval (0., 1.), optional (default 1e-7)
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
        Number of clusters.

    `sizes_` : array-like (n_features,)
        It contains the size of each cluster.
    """
    def __init__(self, n_clusters=2, mask=None, smoothing_fwhm=None,
                 standardize=True, target_affine=None, target_shape=None,
                 mask_strategy='background', memory=None, memory_level=1,
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
        X : numpy array
            2D data matrix of shape [n_sampels, n_clusters]
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

    def inverse_transform(self, Xred):
        """
        Parameters
        ----------
        Xred : numpy array
            2D data matrix of shape [n_samples, n_features]

        Returns
        -------
        X_inv : Niimg
        """

        check_is_fitted(self, "labels_")

        _, inverse = np.unique(self.labels_, return_inverse=True)

        if self.scaling:
            Xred = Xred / np.sqrt(self.sizes_)
        X_inv = Xred[..., inverse]

        return self.masker_.inverse_transform(X_inv)


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


