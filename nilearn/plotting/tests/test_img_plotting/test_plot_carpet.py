"""Tests for :func:`nilearn.plotting.plot_carpet`."""

import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting import plot_carpet


def test_plot_carpet(testdata_4d_for_plotting):
    """Check contents of plot_carpet figure against data in image."""
    img_4d = testdata_4d_for_plotting["img_4d"]
    img_4d_long = testdata_4d_for_plotting["img_4d_long"]
    mask_img = testdata_4d_for_plotting["img_mask"]
    display = plot_carpet(img_4d, mask_img, detrend=False, title="TEST")
    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    plotted_array = ax.images[0].get_array()
    assert plotted_array.shape == (
        np.prod(img_4d.shape[:-1]),
        img_4d.shape[-1],
    )
    # Make sure that the values in the figure match the values in the image
    np.testing.assert_almost_equal(
        plotted_array.sum(), img_4d.get_fdata().sum(), decimal=3
    )
    # Save execution time and memory
    plt.close(display)

    fig, ax = plt.subplots()
    display = plot_carpet(
        img_4d_long,
        mask_img,
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
    n_items = np.prod(img_4d_long.shape[:-1]) * np.ceil(
        img_4d_long.shape[-1] / 4
    )
    assert plotted_array.size == n_items
    plt.close(display)


def test_plot_carpet_with_atlas(testdata_4d_for_plotting):
    """Test plot_carpet when using an atlas."""
    img_4d = testdata_4d_for_plotting["img_4d"]
    mask_img = testdata_4d_for_plotting["img_atlas"]
    atlas_labels = testdata_4d_for_plotting["atlas_labels"]

    # Test atlas - labels
    # t_r is set explicitly for this test as well
    display = plot_carpet(img_4d, mask_img, t_r=2, detrend=False, title="TEST")

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
    assert len(np.unique(colorbar)) == len(atlas_labels)

    # Save execution time and memory
    plt.close(display)

    # Test atlas + labels
    fig, ax = plt.subplots()
    display = plot_carpet(
        img_4d,
        mask_img,
        mask_labels=atlas_labels,
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
    assert set(yticklabels) == set(atlas_labels.keys())

    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    colorbar = ax.images[0].get_array()
    assert len(np.unique(colorbar)) == len(atlas_labels)
    plt.close(display)
