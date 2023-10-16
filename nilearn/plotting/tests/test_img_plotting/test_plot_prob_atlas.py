"""Tests for :func:`nilearn.plotting.plot_prob_atlas`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.plotting import plot_prob_atlas


@pytest.mark.parametrize(
    "params",
    [
        {"view_type": "contours"},
        {"view_type": "filled_contours", "threshold": 0.2},
        {"view_type": "continuous"},
        {"view_type": "filled_contours", "colorbar": True},
        {"threshold": None},
    ],
)
def test_plot_prob_atlas(affine_eye, params):
    """Smoke tests for plot_prob_atlas.

    Tests different combinations of parameters `view_type`, `threshold`,
    and `colorbar`.
    """
    rng = np.random.RandomState(42)
    data_rng = rng.normal(size=(6, 8, 10, 5))
    plot_prob_atlas(Nifti1Image(data_rng, affine_eye), **params)
    plt.close()


def test_plot_prob_atlas_radiological_view(img_4D_rand_eye):
    """Smoke test for radiological view."""
    result = plot_prob_atlas(img_4D_rand_eye, radiological=True)
    assert result.axes.get("y").radiological is True
    plt.close()
