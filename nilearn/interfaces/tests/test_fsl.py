"""Tests for the nilearn.interfaces.fsl submodule."""

import os

import numpy as np

from nilearn.interfaces.fsl import get_design_from_fslmat


def test_get_design_from_fslmat(tmp_path):
    fsl_mat_path = os.path.join(str(tmp_path), "fsl_mat.txt")
    matrix = np.ones((5, 5))
    with open(fsl_mat_path, "w") as fsl_mat:
        fsl_mat.write("/Matrix\n")
        for row in matrix:
            for val in row:
                fsl_mat.write(str(val) + "\t")
            fsl_mat.write("\n")

    design_matrix = get_design_from_fslmat(fsl_mat_path)
    assert design_matrix.shape == matrix.shape
