"""Compute clusters on surface."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_adjacency_matrix(mesh, dtype=None):
    """Compute the adjacency matrix for a surface.

    The adjacency matrix is a matrix
    with one row and one column for each vertex
    such that the value of a cell `(u,v)` in the matrix is 1
    if nodes `u` and `v` are adjacent and 0 otherwise.

    Parameters
    ----------
    mesh : InMemoryMesh

    dtype : numpy dtype-like or None, default=None
        The dtype that should be used for the returned sparse matrix.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        A sparse matrix representing the edge relationships in `surface`.

    """
    n = mesh.coordinates.shape[0]

    edges = np.vstack(
        [
            mesh.faces[:, [0, 1]],
            mesh.faces[:, [0, 2]],
            mesh.faces[:, [1, 2]],
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

    if dtype is None:
        edge_lens = np.ones_like(edges)
    else:
        edge_lens = np.ones(edges.shape, dtype=dtype)

    # We can now make a sparse matrix.
    ee = np.concatenate([edge_lens, edge_lens])
    uv = np.concatenate([u, v])
    vu = np.concatenate([v, u])

    return csr_matrix((ee, (uv, vu)), shape=(n, n))


def find_surface_clusters(mesh, mask) -> tuple[pd.DataFrame, np.ndarray]:
    """Find clusters of truthy vertices on a surface mesh.

    Parameters
    ----------
    mesh : InMemoryMesh
        Surface mesh providing coordinates and faces.

    mask : (n_vertices,) array_like of bool
        Boolean mask, True where vertex is part of a cluster.

    Returns
    -------
    clusters : pandas.DataFrame
        One row per cluster with:
          - 'label': cluster ID (1..n_clusters)
          - 'size': number of vertices in the cluster

    labels : np.ndarray of shape (n_vertices,)
        Integer labels per vertex.
        0 means background (mask is False).
        Positive integers index the cluster ID (1..n_clusters).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != mesh.n_vertices:
        raise ValueError(
            f"Mask length {mask.shape[0]} does not match "
            f"mesh.n_vertices {mesh.n_vertices}"
        )

    adj = compute_adjacency_matrix(mesh)
    sub_adj = adj[mask][:, mask]

    _, labels_sub = connected_components(sub_adj, directed=False)

    # full label array (0 = background)
    labels = np.zeros(mesh.n_vertices, dtype=int)
    labels[mask] = labels_sub + 1

    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    clusters = pd.DataFrame({"label": unique, "size": counts})

    return clusters, labels
