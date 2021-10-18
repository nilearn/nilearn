"""Tests for :func:`nilearn.plotting.plot_stat_map`."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import get_data
from nilearn.image.resampling import coord_transform
from nilearn.plotting.find_cuts import find_cut_slices
from nilearn.datasets import load_mni152_template
from nibabel import Nifti1Image
from .testing_utils import MNI_AFFINE, testdata_3d  # noqa:F401


def test_plot_stat_map_bad_input(testdata_3d, tmpdir):  # noqa:F811
    """Test for bad input arguments (cf. #510)."""
    filename = str(tmpdir.join('temp.png'))
    ax = plt.subplot(111, rasterized=True)
    plot_stat_map(testdata_3d['img'], symmetric_cbar=True,
                  output_file=filename, axes=ax, vmax=np.nan)
    plt.close()


@pytest.mark.parametrize("params",
                         [{}, {'display_mode': 'x', 'cut_coords': 3}])
def test_save_plot_stat_map(params, testdata_3d, tmpdir):  # noqa:F811
    """Test saving figure to file in different ways."""
    filename = str(tmpdir.join('test.png'))
    display = plot_stat_map(testdata_3d['img'], output_file=filename, **params)
    assert display is None
    display = plot_stat_map(testdata_3d['img'], **params)
    display.savefig(filename)
    plt.close()


@pytest.mark.parametrize("display_mode,cut_coords",
                         [('ortho', (80, -120, -60)),
                          ('y', 2), ('yx', None)])
def test_plot_stat_map_cut_coords_and_display_mode(display_mode, cut_coords,
                                                   testdata_3d):  # noqa:F811
    """Smoke-tests for plot_stat_map.

    Tests different combinations of parameters `cut_coords`
    and `display_mode`.
    """
    plot_stat_map(
        testdata_3d['img'], display_mode=display_mode, cut_coords=cut_coords
    )
    plt.close()


def test_plot_stat_map_with_masked_image(testdata_3d):  # noqa:F811
    """Smoke test coordinate finder with mask."""
    masked_img = Nifti1Image(
        np.ma.masked_equal(get_data(testdata_3d['img']), 0), MNI_AFFINE
    )
    plot_stat_map(masked_img, display_mode='x')
    plt.close()


@pytest.mark.parametrize("data",
                         [np.zeros((91, 109, 91)),
                          np.random.RandomState(42).standard_normal(
                              size=(91, 109, 91))])
def test_plot_stat_map_threshold(data):
    """Tests plot_stat_map with threshold (see #510)."""
    plot_stat_map(Nifti1Image(data, np.eye(4)), threshold=1000, colorbar=True)
    plt.close()


def test_plot_stat_map_threshold_for_affine_with_rotation():
    """Tests for plot_stat_map with thresholding and resampling.

    Threshold was not being applied when affine has a rotation.
    See https://github.com/nilearn/nilearn/issues/599 for more details.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(size=(10, 10, 10))
    # matrix with rotation
    affine = np.array([[-3., 1., 0., 1.],
                       [-1., -3., 0., -2.],
                       [0., 0., 3., 3.],
                       [0., 0., 0., 1.]])
    img = Nifti1Image(data, affine)
    display = plot_stat_map(img, bg_img=None, threshold=1.,
                            display_mode='z', cut_coords=1)
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
    plotted_array = ax.images[0].get_array()
    # Given the high threshold the array should be partly masked
    assert plotted_array.mask.any()
    # Save execution time and memory
    plt.close()


@pytest.mark.parametrize("params",
                         [{},
                          {"symmetric_cbar": True},
                          {"symmetric_cbar": False},
                          {"symmetric_cbar": False, "vmax": 10},
                          {"symmetric_cbar": True, "vmax": 10},
                          {"colorbar": False}])
def test_plot_stat_map_colorbar_variations(params, testdata_3d):  # noqa:F811
    """Smoke test for plot_stat_map with different colorbar configurations."""
    img_positive = testdata_3d['img']
    data_positive = get_data(img_positive)
    rng = np.random.RandomState(42)
    data_negative = -data_positive
    data_heterogeneous = data_positive * rng.standard_normal(
        size=data_positive.shape
    )
    img_negative = Nifti1Image(data_negative, MNI_AFFINE)
    img_heterogeneous = Nifti1Image(data_heterogeneous, MNI_AFFINE)
    for img in [img_positive, img_negative, img_heterogeneous]:
        plot_stat_map(img, cut_coords=(80, -120, -60), **params)
        plt.close()


@pytest.mark.parametrize("shape,direction",
                         [((1, 6, 7), 'x'),
                          ((5, 1, 7), 'y'),
                          ((5, 6, 1), 'z')])
def test_plot_stat_map_singleton_ax_dim(shape, direction):
    """Tests for plot_stat_map and singleton display mode."""
    plot_stat_map(
        Nifti1Image(np.ones(shape), np.eye(4)), None, display_mode=direction
    )
    plt.close()


def test_outlier_cut_coords():
    """Test to plot a subset of a large set of cuts found for a small area."""
    bg_img = load_mni152_template()
    data = np.zeros((79, 95, 79))
    affine = np.array([[-2., 0., 0., 78.],
                       [0., 2., 0., -112.],
                       [0., 0., 2., -70.],
                       [0., 0., 0., 1.]])
    # Color a cube around a corner area:
    x, y, z = 20, 22, 60
    x_map, y_map, z_map = coord_transform(x, y, z, np.linalg.inv(affine))
    data[int(x_map) - 1:int(x_map) + 1,
         int(y_map) - 1:int(y_map) + 1,
         int(z_map) - 1:int(z_map) + 1] = 1
    img = Nifti1Image(data, affine)
    cuts = find_cut_slices(img, n_cuts=20, direction='z')
    plot_stat_map(img, display_mode='z', cut_coords=cuts[-4:],
                  bg_img=bg_img)


def test_plotting_functions_with_dim_invalid_input(testdata_3d):  # noqa:F811
    """Test whether error raises with bad error to input."""
    pytest.raises(ValueError, plot_stat_map, testdata_3d['img'], dim='-10')
