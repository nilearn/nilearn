"""
Tests for :func:`nilearn.plotting.plot_prob_atlas`.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from nibabel import Nifti1Image
from nilearn.plotting import plot_prob_atlas


@pytest.mark.parametrize("params",
                         [{"view_type": 'contours'},
                          {"view_type": 'filled_contours', "threshold": .2},
                          {"view_type": "continuous"},
                          {"view_type": 'filled_contours', "colorbar": True},
                          {"threshold": None}])
def test_plot_prob_atlas(params):
    affine = np.eye(4)
    shape = (6, 8, 10, 5)
    rng = np.random.RandomState(42)
    data_rng = rng.normal(size=shape)
    img = Nifti1Image(data_rng, affine)
    plot_prob_atlas(img, **params)
    plt.close()