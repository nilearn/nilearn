"""Tests for :func:`nilearn.plotting.plot_carpet`."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nilearn.plotting import plot_carpet


def test_plot_carpet(img_4d_mni, img_3d_ones_mni, img_4d_long_mni):
    """Check contents of plot_carpet figure against data in image."""
    display = plot_carpet(
        img_4d_mni, img_3d_ones_mni, detrend=False, title="TEST"
    )
    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    plotted_array = ax.images[0].get_array()
    assert plotted_array.shape == (
        np.prod(img_4d_mni.shape[:-1]),
        img_4d_mni.shape[-1],
    )
    # Make sure that the values in the figure match the values in the image
    np.testing.assert_almost_equal(
        plotted_array.sum(), img_4d_mni.get_fdata().sum(), decimal=3
    )
    # Save execution time and memory
    plt.close(display)

    fig, ax = plt.subplots()
    display = plot_carpet(
        img_4d_long_mni,
        img_3d_ones_mni,
        t_r=None,
        detrend=True,
        title="TEST",
        figure=fig,
        axes=ax,
    )
    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    plotted_array = ax.images[0].get_array()
    # Check size
    n_items = np.prod(img_4d_long_mni.shape[:-1]) * np.ceil(
        img_4d_long_mni.shape[-1] / 2
    )
    assert plotted_array.size == n_items
    plt.close(display)


def test_plot_carpet_with_atlas(img_4d_mni, img_atlas):
    """Test plot_carpet when using an atlas."""
    # Test atlas - labels
    # t_r is set explicitly for this test as well
    display = plot_carpet(
        img_4d_mni,
        mask_img=img_atlas["img"],
        t_r=2,
        detrend=False,
        title="TEST",
    )

    # Check the output
    # Two axes: 1 for colorbar and 1 for imshow
    assert len(display.axes) == 2
    # The y-axis label of the imshow should be 'voxels' since atlas labels are
    # unknown
    ax = display.axes[1]
    assert ax.get_ylabel() == "voxels"

    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    colorbar = ax.images[0].get_array()
    assert len(np.unique(colorbar)) == len(img_atlas["labels"])

    # Save execution time and memory
    plt.close(display)

    # Test atlas + labels
    fig, ax = plt.subplots()
    display = plot_carpet(
        img_4d_mni,
        mask_img=img_atlas["img"],
        mask_labels=img_atlas["labels"],
        detrend=True,
        title="TEST",
        figure=fig,
        axes=ax,
    )
    # Check the output
    # Two axes: 1 for colorbar and 1 for imshow
    assert len(display.axes) == 2
    ax = display.axes[0]

    # The ytick labels of the colorbar should match the atlas labels
    yticklabels = ax.get_yticklabels()
    yticklabels = [yt.get_text() for yt in yticklabels]
    assert set(yticklabels) == set(img_atlas["labels"].keys())

    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    colorbar = ax.images[0].get_array()
    assert len(np.unique(colorbar)) == len(img_atlas["labels"])
    plt.close(display)


def test_plot_carpet_standardize(img_4d_mni, img_3d_ones_mni):
    """Check warning is raised and then suppressed with setting standardize."""
    match = "default strategy for standardize"

    with pytest.deprecated_call(match=match):
        plot_carpet(img_4d_mni, mask_img=img_3d_ones_mni)

    with warnings.catch_warnings(record=True) as record:
        plot_carpet(
            img_4d_mni, mask_img=img_3d_ones_mni, standardize="zscore_sample"
        )
        for m in record:
            assert match not in m.message
