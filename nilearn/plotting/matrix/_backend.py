import numpy as np

from nilearn.plotting.matrix._matplotlib_backend import (
    _sanitize_figure_and_axes,
)

VALID_TRI_VALUES = ("full", "lower", "diag")


def _sanitize_labels(mat_shape, labels):
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


def _sanitize_inputs_plot_matrix(
    mat_shape, tri, labels, reorder, figure, axes
):
    """Help for plot_matrix.

    This function makes sure the inputs to plot_matrix are valid.
    """
    _sanitize_tri(tri)
    labels = _sanitize_labels(mat_shape, labels)
    reorder = _sanitize_reorder(reorder)
    fig, axes, own_fig = _sanitize_figure_and_axes(figure, axes)
    return labels, reorder, fig, axes, own_fig


def _sanitize_reorder(reorder):
    """Help for plot_matrix."""
    VALID_REORDER_ARGS = (True, False, "single", "complete", "average")
    if reorder not in VALID_REORDER_ARGS:
        param_to_print = []
        for item in VALID_REORDER_ARGS:
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


def _sanitize_tri(tri, allowed_values=None):
    """Help for plot_matrix."""
    if allowed_values is None:
        allowed_values = VALID_TRI_VALUES
    if tri not in allowed_values:
        raise ValueError(
            f"Parameter tri needs to be one of: {', '.join(allowed_values)}."
        )
