"""Random walker segmentation algorithm.

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

This code is mostly adapted from scikit-image 0.11.3 release.
Location of file in scikit image: random_walker function and its supporting
sub functions in skimage.segmentation
"""

import warnings

import numpy as np
from scipy import __version__, sparse
from scipy import ndimage as ndi
from scipy.sparse.linalg import cg
from sklearn.utils import as_float_array

from nilearn._utils.helpers import compare_version


def _make_graph_edges_3d(n_x, n_y, n_z):
    """Return a list of edges for a 3D image.

    Parameters
    ----------
    n_x : integer
        The size of the grid in the x direction.

    n_y : integer
        The size of the grid in the y direction.

    n_z : integer
        The size of the grid in the z direction.

    Returns
    -------
    edges : (2, N) ndarray
        With the total number of edges:

            N = n_x * n_y * (nz - 1) +
                n_x * (n_y - 1) * nz +
                (n_x - 1) * n_y * nz

        Graph edges with each column describing a node-id pair.

    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack(
        (vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel())
    )
    edges_right = np.vstack(
        (vertices[:, :-1].ravel(), vertices[:, 1:].ravel())
    )
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_weights_3d(data, spacing, beta=130, eps=1.0e-6):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = sum(
        _compute_gradients_3d(data[..., channel], spacing) ** 2
        for channel in range(data.shape[-1])
    )
    # All channels considered together in this standard deviation
    beta /= 10 * data.std()
    gradients *= beta
    weights = np.exp(-gradients)
    weights += eps
    return weights


def _compute_gradients_3d(data, spacing):
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel() / spacing[2]
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel() / spacing[1]
    gr_down = np.abs(data[:-1] - data[1:]).ravel() / spacing[0]
    return np.r_[gr_deep, gr_right, gr_down]


def _make_laplacian_sparse(edges, weights):
    """Sparse implementation."""
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix(
        (data, (i_indices, j_indices)), shape=(pixel_nb, pixel_nb)
    )
    connect = -np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix(
        (
            np.hstack((data, connect)),
            (np.hstack((i_indices, diag)), np.hstack((j_indices, diag))),
        ),
        shape=(pixel_nb, pixel_nb),
    )
    return lap.tocsr()


def _clean_labels_ar(X, labels):
    X = X.astype(labels.dtype)
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels


def _build_ab(lap_sparse, labels):
    """Build the matrix A and rhs B of the linear system to solve.

    A and B are two block of the laplacian of the image graph.
    """
    labels = labels[labels >= 0]
    indices = np.arange(labels.size)
    unlabeled_indices = indices[labels == 0]
    seeds_indices = indices[labels > 0]
    # The following two lines take most of the time in this function
    B = lap_sparse[unlabeled_indices][:, seeds_indices]
    lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels + 1):
        mask = labels[seeds_indices] == lab
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        rhs.append(B * fs)
    return lap_sparse, rhs


def _mask_edges_weights(edges, weights, mask):
    """Remove edges of the graph connected to masked nodes, \
    as well as corresponding weights of the edges.
    """
    mask0 = np.hstack(
        (mask[:, :, :-1].ravel(), mask[:, :-1].ravel(), mask[:-1].ravel())
    )
    mask1 = np.hstack(
        (mask[:, :, 1:].ravel(), mask[:, 1:].ravel(), mask[1:].ravel())
    )
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    max_node_index = edges.max()
    # Reassign edges labels to 0, 1, ... edges_number - 1
    order = np.searchsorted(
        np.unique(edges.ravel()), np.arange(max_node_index + 1)
    )
    edges = order[edges.astype(np.int64)]
    return edges, weights


def _build_laplacian(data, spacing, mask=None, beta=50):
    l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.0e-10)
    if mask is not None:
        edges, weights = _mask_edges_weights(edges, weights, mask)
    lap = _make_laplacian_sparse(edges, weights)
    del edges, weights
    return lap


def random_walker(data, labels, beta=130, tol=1.0e-3, copy=True, spacing=None):
    """Random walker algorithm for segmentation from markers.

    Parameters
    ----------
    data : array_like
        Image to be segmented in phases.
        Data spacing is assumed isotropic unless
        the `spacing` keyword argument is used.

    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive.

    beta : float, default=130
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).

    tol : float, default=1e-3
        Tolerance to achieve when solving the linear system, in
        cg' mode.

    copy : bool, default=True
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.

    spacing : iterable of floats, optional
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.

    Returns
    -------
    output : ndarray
        An array of ints of same shape as `data`, in which each pixel has
        been labeled according to the marker that reached the pixel first
        by anisotropic diffusion.

    Notes
    -----
    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.

    The algorithm was first proposed in [1]_.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    References
    ----------
    .. [1] Random walks for image segmentation, Leo Grady, IEEE Trans Pattern
       Anal Mach Intell. 2006 Nov;28(11):1768-83.

    """
    out_labels = np.copy(labels)
    if (labels != 0).all():
        warnings.warn(
            "Random walker only segments unlabeled areas, where "
            "labels == 0. No zero valued areas in labels were "
            "found. Returning provided labels."
        )
        return out_labels

    if (labels == 0).all():
        warnings.warn(
            "Random walker received no seed label. Returning provided labels."
        )
        return out_labels

    # We take multichannel as always False since we are not strictly using
    # for image processing as such with RGB values.
    multichannel = False
    if not multichannel:
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(
                "For non-multichannel input, data must be of "
                "dimension 2 or 3."
            )
        dims = data.shape  # To reshape final labeled result
        data = np.atleast_3d(as_float_array(data))[..., np.newaxis]

    # Spacing kwarg checks
    if spacing is None:
        spacing = np.asarray((1.0,) * 3)
    elif len(spacing) == len(dims):
        spacing = (
            np.r_[spacing, 1.0] if len(spacing) == 2 else np.asarray(spacing)
        )
    else:
        raise ValueError(
            "Input argument `spacing` incorrect, should be an "
            "iterable with one number per spatial dimension."
        )

    if copy:
        labels = np.copy(labels)
    label_values = np.unique(labels)

    # Reorder label values to have consecutive integers (no gaps)
    if np.any(np.diff(label_values) != 1):
        mask = labels >= 0
        labels[mask] = np.searchsorted(
            np.unique(labels[mask]), labels[mask]
        ).astype(labels.dtype)
    labels = labels.astype(np.int32)
    # If the array has pruned zones, we can have two problematic situations:
    #   - isolated zero-labeled pixels that cannot be determined because they
    #     are not connected to any seed.
    #   - isolated seeds, that is pixels with labels > 0
    #     in connected components without any zero-labeled pixel
    #     to determine.
    #     This causes errors when computing the Laplacian of the graph.
    # For both cases, the problematic pixels are ignored (label is set to -1).
    if np.any(labels < 0):
        # Handle the isolated zero-labeled pixels first
        filled = ndi.binary_propagation(labels > 0, mask=labels >= 0)
        labels[np.logical_and(np.logical_not(filled), labels == 0)] = -1
        del filled
        # Handle the isolated seeds
        filled = ndi.binary_propagation(labels == 0, mask=labels >= 0)
        isolated = np.logical_and(labels > 0, np.logical_not(filled))
        labels[isolated] = -1
        del filled

    # If the operations above yield only -1 pixels
    if (labels == -1).all():
        warnings.warn(
            "Random walker only segments unlabeled areas, where "
            "labels == 0. Data provided only contains isolated seeds "
            "and isolated pixels. Returning provided labels."
        )
        return out_labels

    labels = np.atleast_3d(labels)
    if np.any(labels < 0):
        lap_sparse = _build_laplacian(
            data, spacing, mask=labels >= 0, beta=beta
        )
    else:
        lap_sparse = _build_laplacian(data, spacing, beta=beta)

    lap_sparse, B = _build_ab(lap_sparse, labels)

    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives
    # first at pixel j by anisotropic diffusion.
    X = _solve_cg(lap_sparse, B, tol=tol)

    # Clean up results
    X = _clean_labels_ar(X + 1, labels).reshape(dims)
    return X


def _solve_cg(lap_sparse, B, tol):
    """Solve lap_sparse X_i = B_i for each phase i, using the conjugate \
    gradient method.

    For each pixel, the label i corresponding to the maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    X = [
        cg(lap_sparse, -b_i.todense(), rtol=tol, atol=0)[0]
        # TODO
        # when support scipy to >= 1.12
        # See https://github.com/nilearn/nilearn/pull/4394
        if compare_version(__version__, ">=", "1.12")
        else cg(lap_sparse, -b_i.todense(), tol=tol, atol="legacy")[0]
        for b_i in B
    ]

    X = np.array(X)
    X = np.argmax(X, axis=0)
    return X
