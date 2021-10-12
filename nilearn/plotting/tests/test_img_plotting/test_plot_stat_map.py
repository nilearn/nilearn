"""
Tests for :func:`nilearn.plotting.plot_stat_map`.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import get_data
from nilearn.image.resampling import coord_transform
from nilearn.plotting.find_cuts import find_cut_slices
from nilearn.datasets import load_mni152_template
from nibabel import Nifti1Image
from .testing_utils import MNI_AFFINE, testdata_3d


def test_plot_stat_map_bad_input(testdata_3d, tmpdir):
    # test for bad input arguments (cf. #510)
    filename = str(tmpdir.join('temp.png'))
    ax = plt.subplot(111, rasterized=True)
    plot_stat_map(testdata_3d['img'], symmetric_cbar=True,
                  output_file=filename, axes=ax, vmax=np.nan)
    plt.close()


@pytest.mark.parametrize("params",
                         [{}, {'display_mode': 'x', 'cut_coords': 3}])
def test_save_plot_stat_map(params, testdata_3d, tmpdir):
    filename = str(tmpdir.join('test.png'))
    display = plot_stat_map(testdata_3d['img'], output_file=filename, **params)
    assert display is None
    display = plot_stat_map(testdata_3d['img'], **params)
    display.savefig(filename)
    plt.close()


def test_plot_stat_map(testdata_3d):
    img = testdata_3d['img']
    plot_stat_map(img, cut_coords=(80, -120, -60))

    # Smoke test coordinate finder, with and without mask
    masked_img = Nifti1Image(np.ma.masked_equal(get_data(img), 0), MNI_AFFINE)
    plot_stat_map(masked_img, display_mode='x')
    plot_stat_map(img, display_mode='y', cut_coords=2)

    # 'yx' display_mode
    plot_stat_map(img, display_mode='yx')

    # regression test #510
    data = np.zeros((91, 109, 91))
    aff = np.eye(4)
    new_img = Nifti1Image(data, aff)
    plot_stat_map(new_img, threshold=1000, colorbar=True)

    rng = np.random.RandomState(42)
    data = rng.standard_normal(size=(91, 109, 91))
    new_img = Nifti1Image(data, aff)
    plot_stat_map(new_img, threshold=1000, colorbar=True)
    plt.close()


def test_plot_stat_map_threshold_for_affine_with_rotation():
    # threshold was not being applied when affine has a rotation
    # see https://github.com/nilearn/nilearn/issues/599 for more details
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
def test_plot_stat_map_colorbar_variations(params, testdata_3d):
    # This is only a smoke test
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


def test_singleton_ax_dim():
    for axis, direction in enumerate("xyz"):
        shape = [5, 6, 7]
        shape[axis] = 1
        img = Nifti1Image(np.ones(shape), np.eye(4))
        plot_stat_map(img, None, display_mode=direction)
        plt.close()


def test_outlier_cut_coords():
    """Test to plot a subset of a large set of cuts found for a small area."""
    bg_img = load_mni152_template()
    data = np.zeros((79, 95, 79))
    affine = np.array([[  -2.,    0.,    0.,   78.],
                       [   0.,    2.,    0., -112.],
                       [   0.,    0.,    2.,  -70.],
                       [   0.,    0.,    0.,    1.]])
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


def test_plotting_functions_with_dim_invalid_input(testdata_3d):
    # Test whether error raises with bad error to input
    img = testdata_3d['img']
    pytest.raises(ValueError, plot_stat_map, img, dim='-10')