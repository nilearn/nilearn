import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

VALID_REORDER_VALUES = (True, False, "single", "complete", "average")
VALID_TRI_VALUES = ("full", "lower", "diag")


def mask_matrix(mat, tri):
    """Help for plot_matrix.

    This function masks the matrix depending on the provided
    value of ``tri``.
    """
    if tri == "lower":
        mask = np.tri(mat.shape[0], k=-1, dtype=bool) ^ True
    else:
        mask = np.tri(mat.shape[0], dtype=bool) ^ True
    return np.ma.masked_array(mat, mask)


def sanitize_labels(mat_shape, labels):
    """Help for plot_matrix."""
    # we need a list so an empty one will be cast to False
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if labels and len(labels) != mat_shape[0]:
        raise ValueError(
            f"Length of labels ({len(labels)}) "
            f"unequal to length of matrix ({mat_shape[0]})."
        )
    return labels


def sanitize_reorder(reorder):
    """Help for plot_matrix."""
    if reorder not in VALID_REORDER_VALUES:
        param_to_print = []
        for item in VALID_REORDER_VALUES:
            if isinstance(item, str):
                param_to_print.append(f'"{item}"')
            else:
                param_to_print.append(str(item))
        raise ValueError(
            "Parameter reorder needs to be one of:"
            f"\n{', '.join(param_to_print)}."
        )
    reorder = "average" if reorder is True else reorder
    return reorder


def sanitize_tri(tri, allowed_values=None):
    """Help for plot_matrix."""
    if allowed_values is None:
        allowed_values = VALID_TRI_VALUES
    if tri not in allowed_values:
        raise ValueError(
            f"Parameter tri needs to be one of: {', '.join(allowed_values)}."
        )


def reorder_matrix(mat, labels, reorder):
    """Help for plot_matrix.

    This function reorders the provided matrix.
    """
    if not labels:
        raise ValueError("Labels are needed to show the reordering.")

    linkage_matrix = linkage(mat, method=reorder)
    ordered_linkage = optimal_leaf_ordering(linkage_matrix, mat)
    index = leaves_list(ordered_linkage)
    # make sure labels is an ndarray and copy it
    labels = np.array(labels).copy()
    mat = mat.copy()
    # and reorder labels and matrix
    labels = labels[index].tolist()
    mat = mat[index, :][:, index]
    return mat, labels
