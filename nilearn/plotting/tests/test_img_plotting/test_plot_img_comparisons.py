"""Tests for :func:`nilearn.plotting.plot_img_comparison`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.image import iter_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_img_comparison


def test_plot_img_comparison(rng):
    """Tests for plot_img_comparision."""
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    kwargs = {"shape": (3, 2, 4), "length": 5}
    query_images, mask_img = generate_fake_fmri(random_state=rng, **kwargs)
    # plot_img_comparison doesn't handle 4d images ATM
    query_images = list(iter_img(query_images))
    target_images, _ = generate_fake_fmri(random_state=rng, **kwargs)
    target_images = list(iter_img(target_images))
    target_images[0] = query_images[0]
    masker = NiftiMasker(mask_img).fit()
    correlations = plot_img_comparison(
        target_images, query_images, masker, axes=axes, src_label="query"
    )
    assert len(correlations) == len(query_images)
    assert correlations[0] == pytest.approx(1.0)
    ax_0, ax_1 = axes
    # 5 scatterplots
    assert len(ax_0.collections) == 5
    assert len(
        ax_0.collections[0].get_edgecolors()
        == masker.transform(target_images[0]).ravel().shape[0]
    )
    assert ax_0.get_ylabel() == "query"
    assert ax_0.get_xlabel() == "image set 1"
    # 5 regression lines
    assert len(ax_0.lines) == 5
    assert ax_0.lines[0].get_linestyle() == "--"
    assert ax_1.get_title() == "Histogram of imgs values"
    assert len(ax_1.patches) == 5 * 2 * 128
    correlations_1 = plot_img_comparison(
        target_images, query_images, masker, plot_hist=False
    )
    assert np.allclose(correlations, correlations_1)
