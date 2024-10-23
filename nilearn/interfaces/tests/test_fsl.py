"""Tests for the nilearn.interfaces.fsl submodule."""

import numpy as np

from nilearn.interfaces.fsl import get_design_from_fslmat


def test_get_design_from_fslmat(tmp_path):
    fsl_mat_path = tmp_path / "fsl_mat.txt"
    matrix = np.ones((5, 5))
    with fsl_mat_path.open("w") as fsl_mat:
        fsl_mat.write("/Matrix\n")
        for row in matrix:
            for val in row:
                fsl_mat.write(str(val) + "\t")
            fsl_mat.write("\n")

    design_matrix = get_design_from_fslmat(fsl_mat_path)
    assert design_matrix.shape == matrix.shape
