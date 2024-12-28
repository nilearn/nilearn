def _compute_adjacency_matrix(surface, values="ones", dtype=None):
    """Compute the adjacency matrix for a surface.

    The adjacency matrix is a matrix
    with one row and one column for each vertex
    such that the value of a cell `(u,v)` in the matrix is 1
    if nodes `u` and `v` are adjacent and 0 otherwise.

    Parameters
    ----------
    surface : Surface-like
        The surface whose adjacency matrix is to be computed.

    values : { 'len' | 'invlen' | 'ones'}, optional
        If `values` is `'ones'` (the default), then the returned matrix
        contains uniform values in the cells representing edges.
        If the value is `'len'` then the cells contain
        the edge length of the represented edge.
        If the value is `'invlen'`, then the the inverse of the distances
        are returned.

    dtype : numpy dtype-like or None, optional
        The dtype that should be used for the returned sparse matrix.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        A sparse matrix representing the edge relationships in `surface`.

    """
    from scipy.sparse import csr_matrix

    surface = load_surf_mesh(surface)
    # This is a bit of a hack to quickly find a unique set of all edges.
    n = surface.coordinates.shape[0]
    edges = np.vstack(
        [
            surface.faces[:, [0, 1]],
            surface.faces[:, [0, 2]],
            surface.faces[:, [1, 2]],
        ]
    )
    edges = edges.astype(np.int64)
    bigcol = edges[:, 0] > edges[:, 1]
    lilcol = ~bigcol
    edges = np.concatenate(
        [
            edges[bigcol, 0] + edges[bigcol, 1] * n,
            edges[lilcol, 1] + edges[lilcol, 0] * n,
        ]
    )
    edges = np.unique(edges)
    (u, v) = (edges // n, edges % n)
    # Calculate distances between pairs.
    # We use this as a weighting to make sure that
    # smoothing takes into account the distance between each vertex neighbor
    if values == "len" or values == "invlen":
        coords = surface.coordinates
        edge_lens = np.sqrt(np.sum((coords[u, :] - coords[v, :]) ** 2, axis=1))
        if dtype is None:
            dtype = edge_lens.dtype
        else:
            edge_lens = edge_lens.astype(dtype)
        if values == "invlen":
            edge_lens = 1 / edge_lens
    elif values == "ones":
        if dtype is None:
            edge_lens = np.ones_like(edges)
        else:
            edge_lens = np.ones(edges.shape, dtype=dtype)
    else:
        raise ValueError(f"unrecognized values argument: {values}")
    # We can now make a sparse matrix.
    ee = np.concatenate([edge_lens, edge_lens])
    uv = np.concatenate([u, v])
    vu = np.concatenate([v, u])
    return csr_matrix((ee, (uv, vu)), shape=(n, n))


def _compute_vertex_neighborhoods(surface):
    """For each vertex, compute the neighborhood.

    The neighborhood is defined as all the vertices
    that are connected by a face.

    Parameters
    ----------
    surface : Surface-like
        The surface whose vertex neighborhoods are to be computed.

    Returns
    -------
    neighbors : list
        A list of all the vertices that are connected to each vertex
    """
    from scipy.sparse import find

    matrix = _compute_adjacency_matrix(surface)
    return [find(row)[1] for row in matrix]


def smooth_surface_data(
    surface,
    surf_data,
    iterations=1,
    distance_weights=False,
    vertex_weights=None,
    return_vertex_weights=False,
    center_surround_knob=0,
    match="sum",
):
    """Smooth values along the surface.

    Parameters
    ----------
    surface : Surface-like
        The surface on which the data `surf_data` are to be smoothed.

    surf_data : array-like
        The array of values at each vertex that is being smoothed.
        This may either be a vector of length `n`
        or a matrix with `n` rows.
        In the case of fMRI data, `n` could be the number of timepoints.
        Each column is smoothed independently.

    iterations : :obj:`int`, optional
        The number of times to repeat the smoothing operation
        (it must be a positive value).
        Defaults to 1

    distance_weights : :obj:`bool`, optional
        Whether to add distance-based weighting to the smoothing.
        With such weights, the value calculated for each vertex
        at each iteration is the weighted sum of neighboring vertices
        where the weight on each neighbor is the inverse
        of the distances to it.

    vertex_weights : array-like or None, optional
        A vector of weights, one per vertex.
        These weights are normalized and
        applied to the smoothing
        after the application of center-surround weights.

    return_vector_weights : :obj:`bool`, optional
        If `True` then `(smoothed_data, smoothed_vertex_weights)` are returned.
        The default is `False`.

    center_surround_knob : :obj:`float`, optional
        The relative weighting of the center and the surround
        in each iteration of the smoothing.
        If the value of the knob is `k`,
        then the total weight of vertices that are neighbors
        of a given vertex (the vertex's surround)
        is `2**k` times the weight of the vertex itself (the center).
        A value of 0 (the default) means that, in each smoothing iteration,
        each vertex is updated with the average of its value
        and the average value of its neighbors.
        A value of `-inf` results in no smoothing because the entire
        weight is on the center, so each vertex is updated with its own value.
        A value of `inf` results in each vertex being updated
        with the average of its neighbors without including its own value.

    match : { 'sum' | 'mean' | 'var' | 'dist' | None }, optional
        What properties of the input data should be matched in the output data.
        `None` indicates that the smoothed output should be
        returned without transformation. If the value is `'sum'`, then the
        output is rescaled to have the same sum as `surf_data`.
        If the value is `'mean'`,
        then the output is shifted to match the mean of the input.
        If the value is `'var'` or `'std'`,
        then the variance of the output is matched.
        Finally, if the value is `'dist'`,
        then the mean and the variance are matched. Default is `'sum'`

    Returns
    -------
    surf_data_smooth : array
        The array of smoothed values at each vertex.

    Examples
    --------
    >>> from nilearn import datasets, surface, plotting
    >>> fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    >>> white_left = surface.load_surf_mesh(fsaverage.white_left)
    >>> curv = surface.load_surf_data(fsaverage.curv_left)
    >>> curv_smooth = surface.smooth_surface_data(surface=white_left,
    ... surf_data=curv, iterations=50)
    >>> plotting.plot_surf(white_left, surf_map = curv_smooth)

    """
    surface = load_surf_mesh(surface)
    # First, calculate the center and surround weights for the
    # center-surround knob.
    center_weight = 1 / (1 + np.exp2(-center_surround_knob))
    surround_weight = 1 - center_weight
    if surround_weight == 0:
        # There's nothing to do in this case.
        return np.array(surf_data)
    # Calculate the adjacency matrix either weighting
    # by inverse distance or not weighting (ones)
    values = "invlen" if distance_weights else "ones"
    matrix = _compute_adjacency_matrix(surface, values=values)

    # If there are vertex weights, get them ready.
    if vertex_weights:
        w = np.array(vertex_weights)
        w /= np.sum(w)
    else:
        w = np.ones(matrix.shape[0])
    # We need to normalize the matrix columns, and we can do this now by
    # normalizing everything but the diagonal to the surround weight, then
    # adding the center weight along the diagonal.
    colsums = matrix.sum(axis=1)
    colsums = np.asarray(colsums).flatten()
    matrix = matrix.multiply(surround_weight / colsums[:, None])

    # Add in the diagonal.
    matrix.setdiag(center_weight)
    # Run the iteratioons of smooothing.
    data = surf_data
    for _ in range(iterations):
        data = matrix.dot(data)
    # Convert back into numpy array.
    data = np.reshape(np.asarray(data), np.shape(surf_data))
    # Rescale it if needed.
    if match == "sum":
        sum0 = np.nansum(surf_data, axis=0)
        sum1 = np.nansum(data, axis=0)
        data = data * (sum0 / sum1)
    elif match == "mean":
        mu0 = np.nanmean(surf_data, axis=0)
        mu1 = np.nanmean(data, axis=0)
        data = data + (mu0 - mu1)
    elif match in ("var", "std", "variance", "stddev", "sd"):
        std0 = np.nanstd(surf_data, axis=0)
        std1 = np.nanstd(data, axis=0)
        mu1 = np.nanmean(data, axis=0)
        data = (data - mu1) * (std0 / std1) + mu1
    elif match in ("dist", "meanvar", "meanstd", "meansd"):
        std0 = np.nanstd(surf_data, axis=0)
        std1 = np.nanstd(data, axis=0)
        mu0 = np.nanmean(surf_data, axis=0)
        mu1 = np.nanmean(data, axis=0)
        data = (data - mu1) * (std0 / std1) + mu0
    elif match is not None:
        raise ValueError(f"invalid match argument: {match}")
    if return_vertex_weights:
        w /= np.sum(w)
        return (data, w)
    else:
        return data
