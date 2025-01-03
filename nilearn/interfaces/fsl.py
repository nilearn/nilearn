"""Functions for working with the FSL library."""

from pathlib import Path

import numpy as np
import pandas as pd


def get_design_from_fslmat(fsl_design_matrix_path, column_names=None):
    """Extract design matrix dataframe from FSL mat file.

    Parameters
    ----------
    fsl_design_matrix_path : :obj:`str`
        Path to the FSL design matrix file.
    column_names : None or :obj:`list` of :obj:`str`, default=None
        The names of the columns in the design matrix.

    Returns
    -------
    design_matrix : :obj:`pandas.DataFrame`
        A DataFrame containing the design matrix.
    """
    with Path(fsl_design_matrix_path).open() as design_matrix_file:
        # Based on the openneuro example this seems to be the right
        # marker to start extracting the matrix until the end of the file
        # Conventions of FSL mat files should be verified in more detail for
        # a general case
        for line in design_matrix_file:
            if "/Matrix" in line:
                break

        design_matrix = np.array(
            [
                [float(val) for val in line.replace("\t\n", "").split("\t")]
                for line in design_matrix_file
            ]
        )
        design_matrix = pd.DataFrame(design_matrix, columns=column_names)

    return design_matrix
