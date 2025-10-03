"""Matrix plotting utilities available to other nilearn modules."""

import warnings

import numpy as np

from nilearn._utils.logger import find_stack_level
from nilearn.glm.contrasts import expression_to_contrast_vector


def pad_contrast_matrix(contrast_def, design_matrix):
    """Pad contrasts with zeros.

    Parameters
    ----------
    contrast_def : :class:`numpy.ndarray`, str
        Contrast to be padded

    design_matrix : :class:`pandas.DataFrame`
        Design matrix to use.

    Returns
    -------
    axes : :class:`numpy.ndarray`
        Padded contrast

    """
    design_column_names = design_matrix.columns.tolist()
    if isinstance(contrast_def, str):
        contrast_def = expression_to_contrast_vector(
            contrast_def, design_column_names
        )
    n_columns_design_matrix = len(design_column_names)
    n_columns_contrast_def = (
        contrast_def.shape[0]
        if contrast_def.ndim == 1
        else contrast_def.shape[1]
    )
    horizontal_padding = n_columns_design_matrix - n_columns_contrast_def
    if horizontal_padding == 0:
        return contrast_def
    warnings.warn(
        (
            f"Contrasts will be padded with {horizontal_padding} "
            "column(s) of zeros."
        ),
        category=UserWarning,
        stacklevel=find_stack_level(),
    )
    contrast_def = np.pad(
        contrast_def,
        ((0, 0), (0, horizontal_padding)),
        "constant",
        constant_values=(0, 0),
    )
    return contrast_def
