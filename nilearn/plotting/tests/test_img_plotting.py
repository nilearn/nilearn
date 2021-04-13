
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from functools import partial
from distutils.version import LooseVersion

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import nibabel
import numpy as np
import pytest

from scipy import sparse

from nilearn import _utils
from nilearn.image.resampling import coord_transform, reorder_img
from nilearn._utils import data_gen
from nilearn.image import get_data
from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_template
from nilearn.plotting.find_cuts import find_cut_slices
from nilearn.plotting.img_plotting import (MNI152TEMPLATE, plot_anat, plot_img,
                                           plot_roi, plot_stat_map, plot_epi,
                                           plot_glass_brain, plot_connectome,
                                           plot_connectome_strength,
                                           plot_markers, plot_prob_atlas,
                                           plot_carpet,
                                           _get_colorbar_and_data_ranges)
from nilearn import plotting

mni_affine = np.array([[-2.,    0.,    0.,   90.],
                       [0.,    2.,    0., -126.],
                       [0.,    0.,    2.,  -72.],
                       [0.,    0.,    0.,    1.]])


@pytest.fixture()
def testdata_3d():
    """A random 3D image for testing figures.
    """
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    img_3d = nibabel.Nifti1Image(data_positive, mni_affine)
    data = {
        'img': img_3d,
    }
    return data


@pytest.fixture()
def testdata_4d():
    """Random 4D images for testing figures for multivolume data.
    """
    rng = np.random.RandomState(42)
    img_4d = nibabel.Nifti1Image(rng.uniform(size=(7, 7, 3, 10)), mni_affine)
    img_4d_long = nibabel.Nifti1Image(
        rng.uniform(size=(7, 7, 3, 1777)), mni_affine
    )
    img_mask = nibabel.Nifti1Image(np.ones((7, 7, 3), int), mni_affine)
    atlas = np.ones((7, 7, 3), int)
    atlas[2:5, :, :] = 2
    atlas[5:8, :, :] = 3
    img_atlas = nibabel.Nifti1Image(atlas, mni_affine)
    atlas_labels = {
        "gm": 1,
        "wm": 2,
        "csf": 3,
    }
    data = {
        'img_4d': img_4d,
        'img_4d_long': img_4d_long,
        'img_mask': img_mask,
        'img_atlas': img_atlas,
        'atlas_labels': atlas_labels,
    }
    return data


def test_mni152template_is_reordered():
    """See issue #2550"""
    reordered_mni = reorder_img(load_mni152_template())
    assert np.allclose(get_data(reordered_mni), get_data(MNI152TEMPLATE))
    assert np.allclose(reordered_mni.affine, MNI152TEMPLATE.affine)
    assert np.allclose(reordered_mni.shape, MNI152TEMPLATE.shape)


def demo_plot_roi(**kwargs):
    """ Demo plotting an ROI
    """
    mni_affine = MNI152TEMPLATE.get_affine()
    data = np.zeros((91, 109, 91))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z,
                                          np.linalg.inv(mni_affine))
    data[int(x_map) - 5:int(x_map) + 5, int(y_map) - 3:int(y_map) + 3,
         int(z_map) - 10:int(z_map) + 10] = 1
    img = nibabel.Nifti1Image(data, mni_affine)
    return plot_roi(img, title="Broca's area", **kwargs)


def test_plot_roi_view_types():
    # This is only a smoke test contours rois
    demo_plot_roi(view_type='contours')
    # This is only a smoke test contours rois
    demo_plot_roi(view_type='continuous')

    # Test error message for invalid view_type
    with pytest.raises(ValueError,
                       match='Unknown view type:'
                       ):
        demo_plot_roi(view_type='flled')
    plt.close()


def test_plot_roi_contours():
    display = plot_roi(None)
    data = np.zeros((91, 109, 91))
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z,
                                          np.linalg.inv(mni_affine))
    data[int(x_map) - 5:int(x_map) + 5, int(y_map) - 3:int(y_map) + 3,
         int(z_map) - 10:int(z_map) + 10] = 1
    img = nibabel.Nifti1Image(data, mni_affine)
    plot_roi(img, cmap='RdBu', alpha=0.1, view_type='contours',
             linewidths=2.)
    plt.close()


def test_demo_plot_roi(tmpdir):
    # This is only a smoke test
    demo_plot_roi()
    # Test the black background code path
    demo_plot_roi(black_bg=True)
    # Test whether the function accepts a threshold argument
    demo_plot_roi(threshold=0.2)

    # Save execution time and memory
    plt.close()
    filename = str(tmpdir.join('test.png'))
    with open(filename, 'wb') as fp:
        out = demo_plot_roi(output_file=fp)
    assert out is None


def test_plot_anat(testdata_3d, tmpdir):
    img = testdata_3d['img']

    # Test saving with empty plot
    z_slicer = plot_anat(anat_img=False, display_mode='z')
    filename = str(tmpdir.join('test.png'))
    z_slicer.savefig(filename)

    z_slicer = plot_anat(display_mode='z')
    z_slicer.savefig(filename)

    ortho_slicer = plot_anat(img, dim='auto')
    ortho_slicer.savefig(filename)

    # Save execution time and memory
    plt.close()


def test_plot_functions(testdata_3d, testdata_4d, tmpdir):
    img_3d = testdata_3d['img']
    img_4d = testdata_4d['img_4d']
    img_4d_mask = testdata_4d['img_mask']

    # smoke-test for 3D plotting functions with default arguments
    filename = str(tmpdir.join('temp.png'))
    for plot_func in [plot_anat, plot_img, plot_stat_map, plot_epi,
                      plot_glass_brain]:
        plot_func(img_3d, output_file=filename)

    # smoke-test for 4D plotting functions with default arguments
    for plot_func in [plot_carpet]:
        plot_func(img_4d, mask_img=img_4d_mask, output_file=filename)

    # test for bad input arguments (cf. #510)
    ax = plt.subplot(111, rasterized=True)
    plot_stat_map(img_3d, symmetric_cbar=True,
                  output_file=filename,
                  axes=ax, vmax=np.nan)
    plt.close()


def test_plot_functions_mosaic_mode(testdata_3d):
    img_3d = testdata_3d['img']

    # smoke-test for 3D plotting functions with display_mode='mosaic'
    for cut_coords in [None, 5, (5, 4, 3)]:
        for plot_func in [plot_anat, plot_img, plot_stat_map, plot_epi,
                          plot_roi]:
            plot_func(img_3d, display_mode='mosaic',
                      title='mosaic mode', cut_coords=cut_coords)

    plt.close()


def test_plot_glass_brain(testdata_3d, tmpdir):
    img = testdata_3d['img']

    # test plot_glass_brain with colorbar
    plot_glass_brain(img, colorbar=True, resampling_interpolation='nearest')

    # test plot_glass_brain with negative values
    plot_glass_brain(img, colorbar=True, plot_abs=False,
                     resampling_interpolation='nearest')

    # Save execution time and memory
    plt.close()
    # smoke-test for hemispheric glass brain
    filename = str(tmpdir.join('test.png'))
    plot_glass_brain(img, output_file=filename, display_mode='lzry')
    plt.close()


def test_plot_stat_map(testdata_3d):
    img = testdata_3d['img']

    plot_stat_map(img, cut_coords=(80, -120, -60))

    # Smoke test coordinate finder, with and without mask
    masked_img = nibabel.Nifti1Image(
        np.ma.masked_equal(get_data(img), 0),
        mni_affine)
    plot_stat_map(masked_img, display_mode='x')
    plot_stat_map(img, display_mode='y', cut_coords=2)

    # 'yx' display_mode
    plot_stat_map(img, display_mode='yx')

    # regression test #510
    data = np.zeros((91, 109, 91))
    aff = np.eye(4)
    new_img = nibabel.Nifti1Image(data, aff)
    plot_stat_map(new_img, threshold=1000, colorbar=True)

    rng = np.random.RandomState(42)
    data = rng.standard_normal(size=(91, 109, 91))
    new_img = nibabel.Nifti1Image(data, aff)
    plot_stat_map(new_img, threshold=1000, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_threshold_for_affine_with_rotation(testdata_3d):
    # threshold was not being applied when affine has a rotation
    # see https://github.com/nilearn/nilearn/issues/599 for more details
    rng = np.random.RandomState(42)
    data = rng.standard_normal(size=(10, 10, 10))
    # matrix with rotation
    affine = np.array([[-3., 1., 0., 1.],
                       [-1., -3., 0., -2.],
                       [0., 0., 3., 3.],
                       [0., 0., 0., 1.]])
    img = nibabel.Nifti1Image(data, affine)
    display = plot_stat_map(img, bg_img=None, threshold=1.,
                            display_mode='z', cut_coords=1)
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
    plotted_array = ax.images[0].get_array()
    # Given the high threshold the array should be partly masked
    assert plotted_array.mask.any()

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_threshold_for_uint8(testdata_3d):
    # mask was applied in [-threshold, threshold] which is problematic
    # for uint8 data. See https://github.com/nilearn/nilearn/issues/611
    # for more details
    data = 10 * np.ones((10, 10, 10), dtype='uint8')
    # Having a zero minimum value is important to reproduce
    # https://github.com/nilearn/nilearn/issues/762
    data[0, 0, 0] = 0
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    threshold = np.array(5, dtype='uint8')
    display = plot_stat_map(img, bg_img=None, threshold=threshold,
                            display_mode='z', cut_coords=[0])
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
    plotted_array = ax.images[0].get_array()
    # Make sure that there is one value masked
    assert plotted_array.mask.sum() == 1
    # Make sure that the value masked is in the corner. Note that the
    # axis orientation seem to be flipped, hence (0, 0) -> (-1, 0)
    assert plotted_array.mask[-1, 0]

    # Save execution time and memory
    plt.close()


def test_plot_glass_brain_threshold_for_uint8(testdata_3d):
    # mask was applied in [-threshold, threshold] which is problematic
    # for uint8 data. See https://github.com/nilearn/nilearn/issues/611
    # for more details
    data = 10 * np.ones((10, 10, 10), dtype='uint8')
    # Having a zero minimum value is important to reproduce
    # https://github.com/nilearn/nilearn/issues/762
    data[0, 0] = 0
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    threshold = np.array(5, dtype='uint8')
    display = plot_glass_brain(img, threshold=threshold,
                               display_mode='z', colorbar=True)
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
    plotted_array = ax.images[0].get_array()
    # Make sure that there is one value masked
    assert plotted_array.mask.sum() == 1
    # Make sure that the value masked is in the corner. Note that the
    # axis orientation seem to be flipped, hence (0, 0) -> (-1, 0)
    assert plotted_array.mask[-1, 0]

    # Save execution time and memory
    plt.close()


def test_plot_carpet(testdata_4d):
    """Check contents of plot_carpet figure against data in image."""
    img_4d = testdata_4d['img_4d']
    img_4d_long = testdata_4d['img_4d_long']
    mask_img = testdata_4d['img_mask']
    display = plot_carpet(img_4d, mask_img, detrend=False, title='TEST')
    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    plotted_array = ax.images[0].get_array()
    assert plotted_array.shape == (np.prod(img_4d.shape[:-1]), img_4d.shape[-1])
    # Make sure that the values in the figure match the values in the image
    np.testing.assert_almost_equal(
        plotted_array.sum(),
        img_4d.get_fdata().sum(),
        decimal=3
    )
    # Save execution time and memory
    plt.close(display)

    fig, ax = plt.subplots()
    display = plot_carpet(img_4d_long, mask_img, detrend=True, title='TEST',
                          figure=fig, axes=ax)
    # Next two lines retrieve the numpy array from the plot
    ax = display.axes[0]
    plotted_array = ax.images[0].get_array()
    # Check size
    n_items = (np.prod(img_4d_long.shape[:-1]) * np.ceil(img_4d_long.shape[-1] / 4))
    assert plotted_array.size == n_items
    plt.close(display)


def test_plot_carpet_with_atlas(testdata_4d):
    """Test plot_carpet when using an atlas."""
    img_4d = testdata_4d['img_4d']
    mask_img = testdata_4d['img_atlas']
    atlas_labels = testdata_4d['atlas_labels']

    # Test atlas - labels
    display = plot_carpet(img_4d, mask_img, detrend=False, title='TEST')

    # Check the output
    # Two axes: 1 for colorbar and 1 for imshow
    assert len(display.axes) == 2
    # The y-axis label of the imshow should be 'voxels' since atlas labels are
    # unknown
    ax = display.axes[1]
    assert ax.get_ylabel() == 'voxels'

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
        title='TEST',
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


def test_save_plot(testdata_3d, tmpdir):
    img = testdata_3d['img']

    kwargs_list = [{}, {'display_mode': 'x', 'cut_coords': 3}]

    filename = str(tmpdir.join('test.png'))
    for kwargs in kwargs_list:
        display = plot_stat_map(img, output_file=filename, **kwargs)
        assert display is None

        display = plot_stat_map(img, **kwargs)
        display.savefig(filename)

        # Save execution time and memory
        plt.close()


def test_display_methods(testdata_3d):
    img = testdata_3d['img']

    display = plot_img(img)
    display.add_overlay(img, threshold=0)
    display.add_edges(img, color='c')
    display.add_contours(img, contours=2, linewidth=4,
                         colors=['limegreen', 'yellow'])


def test_plot_with_axes_or_figure(testdata_3d):
    img = testdata_3d['img']
    figure = plt.figure()
    plot_img(img, figure=figure)

    ax = plt.subplot(111)
    plot_img(img, axes=ax)

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_colorbar_variations(testdata_3d):
    # This is only a smoke test
    img_positive = testdata_3d['img']
    data_positive = get_data(img_positive)
    rng = np.random.RandomState(42)
    data_negative = -data_positive
    data_heterogeneous = data_positive * rng.standard_normal(
        size=data_positive.shape
    )
    img_negative = nibabel.Nifti1Image(data_negative, mni_affine)
    img_heterogeneous = nibabel.Nifti1Image(data_heterogeneous, mni_affine)

    for img in [img_positive, img_negative, img_heterogeneous]:
        for func in [plot_stat_map,
                     partial(plot_stat_map, symmetric_cbar=True),
                     partial(plot_stat_map, symmetric_cbar=False),
                     partial(plot_stat_map, symmetric_cbar=False, vmax=10),
                     partial(plot_stat_map, symmetric_cbar=True, vmax=10),
                     partial(plot_stat_map, colorbar=False)]:
            func(img, cut_coords=(80, -120, -60))
            plt.close()


def test_plot_empty_slice(testdata_3d):
    # Test that things don't crash when we give a map with nothing above
    # threshold
    # This is only a smoke test
    data = np.zeros((20, 20, 20))
    img = nibabel.Nifti1Image(data, mni_affine)
    plot_img(img, display_mode='y', threshold=1)

    # Save execution time and memory
    plt.close()


def test_plot_img_invalid(testdata_3d):
    # Check that we get a meaningful error message when we give a wrong
    # display_mode argument
    pytest.raises(Exception, plot_anat, display_mode='zzz')


def test_plot_img_with_auto_cut_coords(testdata_3d):
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = nibabel.Nifti1Image(data, np.eye(4))

    for display_mode in 'xyz':
        plot_img(img, cut_coords=None, display_mode=display_mode,
                 black_bg=True)

        # Save execution time and memory
        plt.close()


def test_plot_img_with_resampling(testdata_3d):
    data = get_data(testdata_3d['img'])
    affine = np.array([[1., -1.,  0.,  0.],
                       [1.,  1.,  0.,  0.],
                       [0.,  0.,  1.,  0.],
                       [0.,  0.,  0.,  1.]])
    img = nibabel.Nifti1Image(data, affine)
    assert not _utils.niimg._is_binary_niimg(img)
    display = plot_img(img)
    display.add_overlay(img)
    display.add_contours(img, contours=2, linewidth=4,
                         colors=['limegreen', 'yellow'])
    display.add_edges(img, color='c')

    # Save execution time and memory
    plt.close()

def test_plot_binary_img_with_resampling(testdata_3d):
    data = get_data(testdata_3d['img'])
    data[data > 0] = 1
    data[data < 0] = 0
    affine = np.array([[1., -1.,  0.,  0.],
                       [1.,  1.,  0.,  0.],
                       [0.,  0.,  1.,  0.],
                       [0.,  0.,  0.,  1.]])
    img = nibabel.Nifti1Image(data, affine)
    assert _utils.niimg._is_binary_niimg(img)
    display = plot_img(img)
    display.add_overlay(img)
    display.add_contours(img)
    plt.close()


def test_plot_noncurrent_axes():
    """Regression test for Issue #450"""
    rng = np.random.RandomState(42)
    maps_img = nibabel.Nifti1Image(rng.random_sample((10, 10, 10)), np.eye(4))
    fh1 = plt.figure()
    fh2 = plt.figure()
    ax1 = fh1.add_subplot(1, 1, 1)

    assert plt.gcf() == fh2, "fh2  was the last plot created."

    # Since we gave ax1, the figure should be plotted in fh1.
    # Before #451, it was plotted in fh2.
    slicer = plot_glass_brain(maps_img, axes=ax1, title='test')
    for ax_name, niax in slicer.axes.items():
        ax_fh = niax.ax.get_figure()
        assert ax_fh == fh1, 'New axis %s should be in fh1.' % ax_name

    # Save execution time and memory
    plt.close()


def test_plot_connectome(tmpdir):
    node_color = ['green', 'blue', 'k', 'cyan']
    # symmetric up to 1e-3 relative tolerance
    adjacency_matrix = np.array([[1., -2., 0.3, 0.],
                                 [-2.002, 1, 0., 0.],
                                 [0.3, 0., 1., 0.],
                                 [0., 0., 0., 1.]])
    node_coords = np.arange(3 * 4).reshape(4, 3)

    args = adjacency_matrix, node_coords
    kwargs = dict(edge_threshold=0.38,
                  title='threshold=0.38',
                  node_size=10, node_color=node_color)
    plot_connectome(*args, **kwargs)
    plt.close()

    # Unique node color
    node_color = np.array(['red'])
    kwargs = dict(edge_threshold=0.38,
                  title='threshold=0.38',
                  node_size=10, node_color=node_color)
    plot_connectome(*args, **kwargs)
    plt.close()

    node_color = 'green'
    kwargs = dict(edge_threshold=0.38,
                  title='threshold=0.38',
                  node_size=10)
    plot_connectome(*args, node_color=node_color, **kwargs)
    plt.close()


    # used to speed-up tests for the next plots
    kwargs['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome(adjacency_matrix,
                    [tuple(each) for each in node_coords],
                    **kwargs)
    # saving to file
    filename = str(tmpdir.join('temp.png'))
    display = plot_connectome(*args, output_file=filename, **kwargs)
    assert display is None
    assert os.path.isfile(filename)
    assert os.path.getsize(filename) > 0
    plt.close()

    # with node_kwargs, edge_kwargs and edge_cmap arguments
    plot_connectome(*args,
                    edge_threshold='70%',
                    node_size=[10, 20, 30, 40],
                    node_color=np.zeros((4, 3)),
                    edge_cmap='RdBu',
                    colorbar=True,
                    node_kwargs={
                        'marker': 'v'},
                    edge_kwargs={
                        'linewidth': 4})
    plt.close()

    # masked array support
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, np.abs(adjacency_matrix) < 0.5)
    plot_connectome(masked_adjacency_matrix, node_coords,
                    **kwargs)
    plt.close()

    # sparse matrix support
    sparse_adjacency_matrix = sparse.coo_matrix(adjacency_matrix)
    plot_connectome(sparse_adjacency_matrix, node_coords,
                    **kwargs)
    plt.close()

    # NaN matrix support
    # Node colors specified as a numpy array rather than a list
    node_color = np.array(['green', 'blue', 'k'])
    # Overriding 'node_color' for 3  elements of size 3.
    kwargs['node_color'] = node_color
    nan_adjacency_matrix = np.array([[1., np.nan, 0.],
                                     [np.nan, 1., 2.],
                                     [np.nan, 2., 1.]])
    nan_node_coords = np.arange(3 * 3).reshape(3, 3)
    plot_connectome(nan_adjacency_matrix, nan_node_coords, **kwargs)
    plt.close()

    # smoke-test where there is no edge to draw, e.g. when
    # edge_threshold is too high
    plot_connectome(*args, edge_threshold=1e12)
    plt.close()

    # with colorbar=True
    plot_connectome(*args, colorbar=True)
    plt.close()

    # smoke-test with hemispheric saggital cuts
    plot_connectome(*args, display_mode='lzry')
    plt.close()

    # test node_color as a string with display_mode='lzry'
    plot_connectome(*args, node_color='red', display_mode='lzry')
    plt.close()
    plot_connectome(*args, node_color=['red'], display_mode='lzry')
    plt.close()

    # Non symmetric matrix
    adjacency_matrix = np.array([[1., -2., 0.3, 0.2],
                                 [0.1, 1, 1.1, 0.1],
                                 [0.01, 2.3, 1., 3.1],
                                 [0.6, 0.03, 1.2, 1.]])
    ax = plot_connectome(adjacency_matrix,
                         node_coords,
                         display_mode='ortho')
    # No thresholding was performed, we should get
    # as many arrows as we have edges
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.prod(adjacency_matrix.shape))

    # Set a few elements of adjacency matrix to zero
    adjacency_matrix[1, 0] = 0.0
    adjacency_matrix[2, 3] = 0.0
    # Plot with different display mode
    ax = plot_connectome(adjacency_matrix,
                         node_coords,
                         display_mode='lzry')
    # No edge in direction 'l' because of node coords
    assert(len([patch for patch in ax.axes['l'].ax.patches
             if isinstance(patch, FancyArrow)]) == 0)
    for direction in ['z', 'r', 'y']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.prod(adjacency_matrix.shape) - 2)

    # Edge thresholding
    # Case 1: Threshold is a number
    thresh = 1.1
    ax = plot_connectome(adjacency_matrix,
                         node_coords,
                         edge_threshold=thresh)
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.sum(np.abs(adjacency_matrix) >= thresh))
    # Case 2: Threshold is a percentage
    thresh = 80
    ax = plot_connectome(adjacency_matrix,
                         node_coords,
                         edge_threshold="{}%".format(thresh))
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
               np.sum(np.abs(adjacency_matrix) >=
                    np.percentile(np.abs(
                        adjacency_matrix.ravel()), thresh)))


def test_plot_connectome_exceptions():
    node_coords = np.arange(2 * 3).reshape((2, 3))

    # Used to speed-up tests because the glass brain is always plotted
    # before any error occurs
    kwargs = {'display_mode': 'x'}

    # adjacency_matrix is not symmetric
    non_symmetric_adjacency_matrix = np.array([[1., 2],
                                               [0.4, 1.]])
    with pytest.warns(UserWarning,
                      match=("'adjacency_matrix' is not symmetric. "
                             "A directed graph will be plotted.")):
        plot_connectome(non_symmetric_adjacency_matrix, node_coords, **kwargs)

    adjacency_matrix = np.array([[1., 2.],
                                 [2., 1.]])
    # adjacency_matrix mask is not symmetric
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, [[False, True], [False, False]])

    with pytest.warns(UserWarning,
                      match=("'adjacency_matrix' was masked with "
                             "a non symmetric mask. A directed "
                             "graph will be plotted.")):
        plot_connectome(masked_adjacency_matrix, node_coords, **kwargs)

    # edges threshold is neither a number nor a string
    with pytest.raises(TypeError,
                       match='should be either a number or a string'):
        plot_connectome(adjacency_matrix, node_coords,
                        edge_threshold=object(),
                        **kwargs)

    # wrong number of node colors
    with pytest.raises(ValueError,
                       match='Mismatch between the number of nodes'):
        plot_connectome(adjacency_matrix, node_coords,
                        node_color=['red', 'blue', 'yellow'],
                        **kwargs)

    with pytest.raises(ValueError,
                       match='Mismatch between the number of nodes'):
        plot_connectome(adjacency_matrix, node_coords,
                        node_color=np.array(['red', 'blue', 'yellow', 'cyan']),
                        **kwargs)

    # wrong shapes for node_coords or adjacency_matrix
    with pytest.raises(
            ValueError,
            match=r'supposed to have shape \(n, n\).+\(1L?, 2L?\)'):
        plot_connectome(adjacency_matrix[:1, :],
                        node_coords,
                        **kwargs)

    with pytest.raises(ValueError, match=r'shape \(2L?, 3L?\).+\(2L?,\)'):
        plot_connectome(adjacency_matrix, node_coords[:, 2], **kwargs)

    wrong_adjacency_matrix = np.zeros((3, 3))
    with pytest.raises(ValueError,
                       match=r'Shape mismatch.+\(3L?, 3L?\).+\(2L?, 3L?\)'
                       ):
        plot_connectome(wrong_adjacency_matrix, node_coords, **kwargs)

    # a few not correctly formatted strings for 'edge_threshold'
    wrong_edge_thresholds = ['0.1', '10', '10.2.3%', 'asdf%']
    for wrong_edge_threshold in wrong_edge_thresholds:
        with pytest.raises(
                ValueError,
                match='should be a number followed by the percent sign'):
            plot_connectome(adjacency_matrix, node_coords,
                            edge_threshold=wrong_edge_threshold, **kwargs)

    # specifying node sizes via node_kwargs
    with pytest.raises(ValueError,
                       match="Please use 'node_size' and not 'node_kwargs'"
                       ):
        plot_connectome(adjacency_matrix, node_coords,
                        node_kwargs={'s': 50},
                        **kwargs)

    # specifying node colors via node_kwargs
    with pytest.raises(
            ValueError,
            match="Please use 'node_color' and not 'node_kwargs'"):
        plot_connectome(adjacency_matrix, node_coords,
                        node_kwargs={'c': 'blue'},
                        **kwargs)


def test_singleton_ax_dim():
    for axis, direction in enumerate("xyz"):
        shape = [5, 6, 7]
        shape[axis] = 1
        img = nibabel.Nifti1Image(np.ones(shape), np.eye(4))
        plot_stat_map(img, None, display_mode=direction)
        plt.close()


def test_plot_prob_atlas():
    affine = np.eye(4)
    shape = (6, 8, 10, 5)
    rng = np.random.RandomState(42)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    # Testing the 4D plot prob atlas with contours
    plot_prob_atlas(img, view_type='contours')
    plt.close()
    # Testing the 4D plot prob atlas with contours
    plot_prob_atlas(img, view_type='filled_contours',
                    threshold=0.2)
    plt.close()
    # Testing the 4D plot prob atlas with contours
    plot_prob_atlas(img, view_type='continuous')
    plt.close()
    # Testing the 4D plot prob atlas with colormap
    plot_prob_atlas(img, view_type='filled_contours', colorbar=True)
    plt.close()
    # threshold=None
    plot_prob_atlas(img, threshold=None)
    plt.close()


def test_get_colorbar_and_data_ranges_with_vmin():
    data = np.array([[-.5, 1., np.nan],
                     [0., np.nan, -.2],
                     [1.5, 2.5, 3.]])

    with pytest.raises(ValueError,
                       match='does not accept a "vmin" argument'
                       ):
        _get_colorbar_and_data_ranges(data, vmax=None,
                                      symmetric_cbar=True,
                                      kwargs={'vmin': 1.}
                                      )


def test_get_colorbar_and_data_ranges_pos_neg():
    # data with positive and negative range
    data = np.array([[-.5, 1., np.nan],
                     [0., np.nan, -.2],
                     [1.5, 2.5, 3.]])

    # Reasonable additional arguments that would end up being passed
    # to imshow in a real plotting use case
    kwargs = {'aspect': 'auto', 'alpha': 0.9}

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == np.nanmin(data)
    assert cbar_vmax == np.nanmax(data)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == np.nanmin(data)
    assert cbar_vmax == np.nanmax(data)

    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None


def test_get_colorbar_and_data_ranges_pos():
    # data with positive range
    data_pos = np.array([[0, 1., np.nan],
                         [0., np.nan, 0],
                         [1.5, 2.5, 3.]])

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None,
        symmetric_cbar=True,
        kwargs={})
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar=True,
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None,
        symmetric_cbar=False,
        kwargs={})
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == 0
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar=False,
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == 0
    assert cbar_vmax == None

    # symmetric_cbar is set to 'auto', same behaviour as false in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None,
        symmetric_cbar='auto',
        kwargs={})
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == 0
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar='auto',
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == 0
    assert cbar_vmax == None


def test_get_colorbar_and_data_ranges_neg():
    # data with negative range
    data_neg = np.array([[-.5, 0, np.nan],
                         [0., np.nan, -.2],
                         [0, 0, 0]])

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None,
        symmetric_cbar=True,
        kwargs={})
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar=True,
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None,
        symmetric_cbar=False,
        kwargs={})
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == 0
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar=False,
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == 0

    # symmetric_cbar is set to 'auto', same behaviour as False in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None,
        symmetric_cbar='auto',
        kwargs={})
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == 0
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar='auto',
        kwargs={})
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == 0


def test_get_colorbar_and_data_ranges_masked_array():
    # data with positive and negative range
    data = np.array([[-.5, 1., np.nan],
                     [0., np.nan, -.2],
                     [1.5, 2.5, 3.]])
    masked_data = np.ma.masked_greater(data, 2.)
    # Easier to fill masked values with NaN to test against later on
    filled_data = masked_data.filled(np.nan)

    # Reasonable additional arguments that would end up being passed
    # to imshow in a real plotting use case
    kwargs = {'aspect': 'auto', 'alpha': 0.9}

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == np.nanmin(filled_data)
    assert cbar_vmax == np.nanmax(filled_data)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == np.nanmin(filled_data)
    assert cbar_vmax == np.nanmax(filled_data)

    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None


def test_invalid_in_display_mode_cut_coords_all_plots(testdata_3d):
    img = testdata_3d['img']
    for plot_func in [plot_img, plot_anat, plot_roi, plot_epi,
                      plot_stat_map, plot_prob_atlas, plot_glass_brain]:
        with pytest.raises(ValueError,
                           match="The input given for display_mode='ortho' "
                                 "needs to "
                                 "be a list of 3d world coordinates."
                           ):
            plot_func(img, display_mode='ortho', cut_coords=2)


def test_invalid_in_display_mode_tiled_cut_coords_single_all_plots(testdata_3d):
    img = testdata_3d['img']

    for plot_func in [plot_img, plot_anat, plot_roi, plot_epi,
                      plot_stat_map, plot_prob_atlas]:
        with pytest.raises(ValueError,
                           match="The input given for display_mode='tiled' "
                                 "needs to "
                                 "be a list of 3d world coordinates."
                           ):
            plot_func(img, display_mode='tiled', cut_coords=2)


def test_invalid_in_display_mode_tiled_cut_coords_all_plots(testdata_3d):
    img = testdata_3d['img']

    for plot_func in [plot_img, plot_anat, plot_roi, plot_epi,
                      plot_stat_map, plot_prob_atlas]:
        with pytest.raises(ValueError,
                           match="The number cut_coords passed does not "
                                 "match the display_mode"
                           ):
            plot_func(img, display_mode='tiled', cut_coords=(2, 2))


def test_invalid_in_display_mode_mosaic_cut_coords_all_plots(testdata_3d):
    img = testdata_3d['img']

    for plot_func in [plot_img, plot_anat, plot_roi, plot_epi,
                      plot_stat_map, plot_prob_atlas]:
        with pytest.raises(ValueError,
                           match="The number cut_coords passed does not "
                                 "match the display_mode"
                           ):
            plot_func(img, display_mode='mosaic', cut_coords=(2, 2))


def test_outlier_cut_coords():
    """ Test to plot a subset of a large set of cuts found for a small area."""
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
    img = nibabel.Nifti1Image(data, affine)
    cuts = find_cut_slices(img, n_cuts=20, direction='z')

    plot_stat_map(img, display_mode='z', cut_coords=cuts[-4:],
                  bg_img=bg_img)


def test_plot_stat_map_with_nans(testdata_3d):
    img = testdata_3d['img']
    data = get_data(img)

    data[6, 5, 1] = np.nan
    data[1, 5, 2] = np.nan
    data[1, 3, 2] = np.nan
    data[6, 5, 2] = np.inf

    img = nibabel.Nifti1Image(data, mni_affine)
    plot_epi(img)
    plot_stat_map(img)
    plot_glass_brain(img)


def test_plotting_functions_with_cmaps():
    img = load_mni152_template()
    cmaps = ['Paired', 'Set1', 'Set2', 'Set3']
    for cmap in cmaps:
        plot_roi(img, cmap=cmap, colorbar=True)
        plot_stat_map(img, cmap=cmap, colorbar=True)
        plot_glass_brain(img, cmap=cmap, colorbar=True)

    if LooseVersion(matplotlib.__version__) >= LooseVersion('2.0.0'):
        plot_stat_map(img, cmap='viridis', colorbar=True)

    plt.close()


def test_plotting_functions_with_nans_in_bg_img(testdata_3d):
    bg_img = testdata_3d['img']
    bg_data = get_data(bg_img)

    bg_data[6, 5, 1] = np.nan
    bg_data[1, 5, 2] = np.nan
    bg_data[1, 3, 2] = np.nan
    bg_data[6, 5, 2] = np.inf

    bg_img = nibabel.Nifti1Image(bg_data, mni_affine)
    plot_anat(bg_img)
    # test with plot_roi passing background image which contains nans values
    # in it
    roi_img = testdata_3d['img']
    plot_roi(roi_img=roi_img, bg_img=bg_img)
    stat_map_img = testdata_3d['img']
    plot_stat_map(stat_map_img=stat_map_img, bg_img=bg_img)

    plt.close()


def test_plotting_functions_with_dim_invalid_input(testdata_3d):
    # Test whether error raises with bad error to input
    img = testdata_3d['img']
    pytest.raises(ValueError, plot_stat_map, img, dim='-10')


def test_add_markers_using_plot_glass_brain():
    fig = plot_glass_brain(None)
    coords = [(-34, -39, -9)]
    fig.add_markers(coords)
    fig.close()

    # Add a single marker in right hemishpere such that no marker
    # should appear in the left hemisphere when plotting
    display = plotting.plot_glass_brain(None, display_mode='lyrz')
    display.add_markers([[20, 20, 20]])
    # Check that Axe 'l' has no marker
    assert display.axes['l'].ax.collections[0].get_offsets().data.shape == (0, 2)
    # Check that all other Axes have one marker
    for d in 'ryz':
        assert display.axes[d].ax.collections[0].get_offsets().data.shape == (1, 2)

    # Add two markers in left hemisphere such that no marker
    # should appear in the right hemisphere when plotting
    display = plotting.plot_glass_brain(None, display_mode='lyrz')
    display.add_markers([[-20, 20, 20], [-10, 10, 10]],
                        marker_color=['r', 'b'])
    # Check that Axe 'r' has no marker
    assert display.axes['r'].ax.collections[0].get_offsets().data.shape == (0, 2)
    # Check that all other Axes have two markers
    for d in 'lyz':
        assert display.axes[d].ax.collections[0].get_offsets().data.shape == (2, 2)


def test_plotting_functions_with_display_mode_tiled(testdata_3d):
    img = testdata_3d['img']
    plot_stat_map(img, display_mode='tiled')
    plot_anat(display_mode='tiled')
    plot_img(img, display_mode='tiled')
    plt.close()


def test_display_methods_with_display_mode_tiled(testdata_3d):
    img = testdata_3d['img']
    display = plot_img(img, display_mode='tiled')
    display.add_overlay(img, threshold=0)
    display.add_edges(img, color='c')
    display.add_contours(img, contours=2, linewidth=4,
                         colors=['limegreen', 'yellow'])


def test_plot_glass_brain_colorbar_having_nans(testdata_3d):
    img = testdata_3d['img']
    data = get_data(img)

    data[6, 5, 2] = np.inf
    img = nibabel.Nifti1Image(data, np.eye(4))
    plot_glass_brain(img, colorbar=True)
    plt.close()


def test_plot_glass_brain_display_modes_without_img():
    # Smoke test for work around from PR #1888
    fig = plot_glass_brain(None, display_mode='lr')
    fig = plot_glass_brain(None, display_mode='lzry')
    fig.close()


def test_plot_glass_brain_with_completely_masked_img():
    # Smoke test for PR #1888 with display modes having 'l'
    data = np.zeros((10, 20, 30))
    affine = np.eye(4)

    img = nibabel.Nifti1Image(data, affine)
    plot_glass_brain(img, display_mode='lzry')
    plot_glass_brain(img, display_mode='lr')
    plt.close()


def test_connectome_strength(tmpdir):
    # symmetric up to 1e-3 relative tolerance
    adjacency_matrix = np.array([[1., -2., 0.3, 0.],
                                 [-2.002, 1, 0., 0.],
                                 [0.3, 0., 1., 0.],
                                 [0., 0., 0., 1.]])
    node_coords = np.arange(3 * 4).reshape(4, 3)

    args = adjacency_matrix, node_coords
    kwargs = dict()
    plot_connectome_strength(*args, **kwargs)
    plt.close()

    # used to speed-up tests for the net plots
    kwargs['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome_strength(adjacency_matrix,
                             [tuple(each) for each in node_coords],
                             **kwargs)

    # saving to file
    filename = str(tmpdir.join('test.png'))
    display = plot_connectome_strength(
        *args, output_file=filename, **kwargs
    )
    assert display is None
    assert os.path.isfile(filename)
    assert os.path.getsize(filename) > 0
    plt.close()

    # passing node args
    plot_connectome_strength(*args, node_size=10, cmap='RdBu')
    plt.close()
    plot_connectome_strength(*args, node_size=10, cmap=plt.cm.RdBu)
    plt.close()

    # masked array support
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, np.abs(adjacency_matrix) < 0.5
    )
    plot_connectome_strength(
        masked_adjacency_matrix, node_coords, **kwargs
    )
    plt.close()

    # sparse matrix support
    sparse_adjacency_matrix = sparse.coo_matrix(adjacency_matrix)
    plot_connectome_strength(
        sparse_adjacency_matrix, node_coords, **kwargs
    )
    plt.close()

    # NaN matrix support
    nan_adjacency_matrix = np.array([[1., np.nan, 0.],
                                     [np.nan, 1., 2.],
                                     [np.nan, 2., 1.]])
    nan_node_coords = np.arange(3 * 3).reshape(3, 3)
    plot_connectome_strength(nan_adjacency_matrix, nan_node_coords, **kwargs)
    plt.close()

    # smoke-test with hemispheric sagital cuts
    plot_connectome_strength(*args, display_mode='lzry')
    plt.close()


def test_plot_connectome_strength_exceptions():
    node_coords = np.arange(2 * 3).reshape((2, 3))

    # Used to speed-up tests because the glass brain is always plotted
    # before any error occurs
    kwargs = {'display_mode': 'x'}

    # adjacency_matrix is not symmetric
    non_symmetric_adjacency_matrix = np.array([[1., 2],
                                               [0.4, 1.]])
    with pytest.raises(ValueError,
                       match='should be symmetric'
                       ):
        plot_connectome_strength(non_symmetric_adjacency_matrix,
                                 node_coords,
                                 **kwargs)

    adjacency_matrix = np.array([[1., 2.],
                                 [2., 1.]])
    # adjacency_matrix mask is not symmetric
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, [[False, True], [False, False]])

    with pytest.raises(ValueError, match='non symmetric mask'):
        plot_connectome_strength(masked_adjacency_matrix,
                                 node_coords,
                                 **kwargs)

    # wrong shapes for node_coords or adjacency_matrix
    with pytest.raises(ValueError,
                       match=r'supposed to have shape \(n, n\).+\(1L?, 2L?\)'
                       ):
        plot_connectome_strength(adjacency_matrix[:1, :],
                                 node_coords,
                                 **kwargs)

    with pytest.raises(ValueError, match=r'shape \(2L?, 3L?\).+\(2L?,\)'):
        plot_connectome_strength(adjacency_matrix,
                                 node_coords[:, 2], **kwargs)

    wrong_adjacency_matrix = np.zeros((3, 3))
    with pytest.raises(ValueError,
                       match=r'Shape mismatch.+\(3L?, 3L?\).+\(2L?, 3L?\)'
                       ):
        plot_connectome_strength(wrong_adjacency_matrix, node_coords,
                                 **kwargs)


def test_plot_markers(tmpdir):
    # Minimal usage
    node_values = [1, 2, 3, 4]
    node_coords = np.array([[39 ,   6, -32],
                            [29 ,  40,   1],
                            [-20, -74,  35],
                            [-29, -59, -37]])
    args = node_values, node_coords
    plot_markers(*args)
    plt.close()

    # Speed-up subsequent tests
    kwargs = {'display_mode': 'x'}

    # node_values is an array
    plot_markers(np.array(node_values), node_coords, **kwargs)
    plt.close()
    plot_markers(np.array(node_values)[:, np.newaxis], node_coords, **kwargs)
    plt.close()
    plot_markers(np.array(node_values)[np.newaxis, :], node_coords, **kwargs)
    plt.close()

    # all node_values are equal
    plot_markers((1, 1, 1, 1), node_coords, **kwargs)
    plt.close()

    # node_coords not an array but a list of tuples
    plot_markers(node_values, [tuple(coord) for coord in node_coords], **kwargs)
    plt.close()

    # Saving to file
    filename = str(tmpdir.join('test.png'))
    display = plot_markers(*args, output_file=filename, **kwargs)
    assert display is None
    assert (os.path.isfile(filename) and  # noqa: W504
                os.path.getsize(filename) > 0)
    plt.close()

    # Different options for node_size
    plot_markers(*args, node_size=10, **kwargs)
    plt.close()
    plot_markers(*args, node_size=[10, 20, 30, 40], **kwargs)
    plt.close()
    plot_markers(*args, node_size=np.array([10, 20, 30, 40]), **kwargs)
    plt.close()

    # Different options for cmap related arguments
    plot_markers(*args, node_cmap='RdBu', node_vmin=0, **kwargs)
    plt.close()
    plot_markers(*args, node_cmap=matplotlib.cm.get_cmap('jet'),
                 node_vmax=5, **kwargs)
    plt.close()
    plot_markers(*args, node_vmin=2, node_vmax=3, **kwargs)
    plt.close()

    # Node threshold support
    plot_markers(*args, node_threshold=-100, **kwargs)
    plt.close()
    plot_markers(*args, node_threshold=2.5, **kwargs)
    plt.close()

    # node_kwargs working and does not interfere with alpha
    node_kwargs = dict(marker='s')
    plot_markers(*args, alpha=.1, node_kwargs=node_kwargs, **kwargs)
    plt.close()


def test_plot_markers_exceptions():
    node_coords = np.array([[39 ,   6, -32],
                            [29 ,  40,   1],
                            [-20, -74,  35],
                            [-29, -59, -37]])

    # # Used to speed-up tests because the glass brain is always plotted
    kwargs = {'display_mode': 'x'}

    # node_values lenght mismatch with node_coords
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_markers([1, 2, 3, 4, 5], node_coords, **kwargs)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_markers([1, 2, 3], node_coords, **kwargs)

    # node_values incorrect shape
    adjacency_matrix = np.random.RandomState(42).random_sample((4, 4))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_markers(adjacency_matrix, node_coords, **kwargs)

    # node_values is wrong type
    with pytest.raises(TypeError):
        plot_markers(['1', '2', '3', '4'], node_coords, **kwargs)

    # incorrect vmin anord vmax bounds for node cmap
    with pytest.raises(ValueError):
        plot_markers([1, 2, 2, 4], node_coords, node_vmin=5, **kwargs)
    with pytest.raises(ValueError):
        plot_markers([1, 2, 2, 4], node_coords, node_vmax=0, **kwargs)

    # node_threshold higher than max node_value
    with pytest.raises(ValueError, match="Provided 'node_threshold' value"):
        plot_markers([1, 2, 2, 4], node_coords, node_threshold=5, **kwargs)

def test_plot_connectome_strength_deprecation_warning():
    with pytest.deprecated_call():
        adjacency_matrix = np.array([[1, -2, 0.3, 0.],
                                     [-2, 1, 0, 0],
                                     [0.3, 0, 1, 0],
                                     [0, 0, 0, 1]])
        node_coords = np.arange(3 * 4).reshape(4, 3)
        plot_connectome_strength(adjacency_matrix, node_coords)


def test_plot_img_comparison():
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    kwargs = {"shape": (3, 2, 4), "length": 5}
    query_images, mask_img = data_gen.generate_fake_fmri(
        rand_gen=np.random.RandomState(0), **kwargs)
    # plot_img_comparison doesn't handle 4d images ATM
    query_images = list(image.iter_img(query_images))
    target_images, _ = data_gen.generate_fake_fmri(
        rand_gen=np.random.RandomState(1), **kwargs)
    target_images = list(image.iter_img(target_images))
    target_images[0] = query_images[0]
    masker = NiftiMasker(mask_img).fit()
    correlations = plotting.plot_img_comparison(
        target_images, query_images, masker, axes=axes, src_label="query")
    assert len(correlations) == len(query_images)
    assert correlations[0] == pytest.approx(1.)
    ax_0, ax_1 = axes
    # 5 scatterplots
    assert len(ax_0.collections) == 5
    assert len(ax_0.collections[0].get_edgecolors() == masker.transform(
        target_images[0]).ravel().shape[0])
    assert ax_0.get_ylabel() == "query"
    assert ax_0.get_xlabel() == "image set 1"
    # 5 regression lines
    assert len(ax_0.lines) == 5
    assert ax_0.lines[0].get_linestyle() == "--"
    assert ax_1.get_title() == "Histogram of imgs values"
    assert len(ax_1.patches) == 5 * 2 * 128
    correlations_1 = plotting.plot_img_comparison(
        target_images, query_images, masker, plot_hist=False)
    assert np.allclose(correlations, correlations_1)
