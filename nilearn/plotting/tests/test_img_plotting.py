
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import tempfile
from functools import partial
from distutils.version import LooseVersion

import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
from nose.tools import assert_raises, assert_true, assert_equal
from scipy import sparse

from nilearn._utils.testing import assert_raises_regex
from nilearn.image.resampling import coord_transform
from nilearn.datasets import load_mni152_template
from nilearn.plotting.find_cuts import find_cut_slices
from nilearn.plotting.img_plotting import (MNI152TEMPLATE, plot_anat, plot_img,
                                           plot_roi, plot_stat_map, plot_epi,
                                           plot_glass_brain, plot_connectome,
                                           plot_prob_atlas,
                                           _get_colorbar_and_data_ranges)

mni_affine = np.array([[-2.,    0.,    0.,   90.],
                       [0.,    2.,    0., -126.],
                       [0.,    0.,    2.,  -72.],
                       [0.,    0.,    0.,    1.]])


def _generate_img():
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.rand(7, 7, 3)
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    return nibabel.Nifti1Image(data_positive, mni_affine)


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


def test_demo_plot_roi():
    # This is only a smoke test
    demo_plot_roi()
    # Test the black background code path
    demo_plot_roi(black_bg=True)
    # Test whether the function accepts a threshold argument
    demo_plot_roi(threshold=0.2)

    # Save execution time and memory
    plt.close()

    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        out = demo_plot_roi(output_file=fp)
    assert_true(out is None)


def test_plot_anat():
    img = _generate_img()

    # Test saving with empty plot
    z_slicer = plot_anat(anat_img=False, display_mode='z')
    filename = tempfile.mktemp(suffix='.png')
    try:
        z_slicer.savefig(filename)
    finally:
        os.remove(filename)

    z_slicer = plot_anat(display_mode='z')
    filename = tempfile.mktemp(suffix='.png')
    try:
        z_slicer.savefig(filename)
    finally:
        os.remove(filename)

    ortho_slicer = plot_anat(img, dim='auto')
    filename = tempfile.mktemp(suffix='.png')
    try:
        ortho_slicer.savefig(filename)
    finally:
        os.remove(filename)

    # Save execution time and memory
    plt.close()


def test_plot_functions():
    img = _generate_img()

    # smoke-test for each plotting function with default arguments
    for plot_func in [plot_anat, plot_img, plot_stat_map, plot_epi,
                      plot_glass_brain]:
        filename = tempfile.mktemp(suffix='.png')
        try:
            plot_func(img, output_file=filename)
        finally:
            os.remove(filename)

    # test for bad input arguments (cf. #510)
    ax = plt.subplot(111, rasterized=True)
    filename = tempfile.mktemp(suffix='.png')
    try:
        plot_stat_map(img, symmetric_cbar=True,
                      output_file=filename,
                      axes=ax, vmax=np.nan)
    finally:
        os.remove(filename)
    plt.close()


def test_plot_glass_brain():
    img = _generate_img()

    # test plot_glass_brain with colorbar
    plot_glass_brain(img, colorbar=True)

    # test plot_glass_brain with negative values
    plot_glass_brain(img, colorbar=True, plot_abs=False)

    # Save execution time and memory
    plt.close()
    # smoke-test for hemispheric glass brain
    filename = tempfile.mktemp(suffix='.png')
    plot_glass_brain(img, output_file=filename, display_mode='lzry')
    plt.close()


def test_plot_stat_map():
    img = _generate_img()

    plot_stat_map(img, cut_coords=(80, -120, -60))

    # Smoke test coordinate finder, with and without mask
    masked_img = nibabel.Nifti1Image(
        np.ma.masked_equal(img.get_data(), 0),
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
    data = rng.randn(91, 109, 91)
    new_img = nibabel.Nifti1Image(data, aff)
    plot_stat_map(new_img, threshold=1000, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_threshold_for_affine_with_rotation():
    # threshold was not being applied when affine has a rotation
    # see https://github.com/nilearn/nilearn/issues/599 for more details
    data = np.random.randn(10, 10, 10)
    # matrix with rotation
    affine = np.array([[-3., 1., 0., 1.],
                       [-1., -3., 0., -2.],
                       [0., 0., 3., 3.],
                       [0., 0., 0., 1.]])
    img = nibabel.Nifti1Image(data, affine)
    display = plot_stat_map(img, bg_img=None, threshold=1e6,
                            display_mode='z', cut_coords=1)
    # Next two lines retrieve the numpy array from the plot
    ax = list(display.axes.values())[0].ax
    plotted_array = ax.images[0].get_array()
    # Given the high threshold the array should be entirely masked
    assert_true(plotted_array.mask.all())

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_threshold_for_uint8():
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
    assert_equal(plotted_array.mask.sum(), 1)
    # Make sure that the value masked is in the corner. Note that the
    # axis orientation seem to be flipped, hence (0, 0) -> (-1, 0)
    assert_true(plotted_array.mask[-1, 0])

    # Save execution time and memory
    plt.close()


def test_plot_glass_brain_threshold_for_uint8():
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
    assert_equal(plotted_array.mask.sum(), 1)
    # Make sure that the value masked is in the corner. Note that the
    # axis orientation seem to be flipped, hence (0, 0) -> (-1, 0)
    assert_true(plotted_array.mask[-1, 0])

    # Save execution time and memory
    plt.close()


def test_save_plot():
    img = _generate_img()

    kwargs_list = [{}, {'display_mode': 'x', 'cut_coords': 3}]

    for kwargs in kwargs_list:
        filename = tempfile.mktemp(suffix='.png')
        try:
            display = plot_stat_map(img, output_file=filename, **kwargs)
        finally:
            os.remove(filename)
        assert_true(display is None)

        display = plot_stat_map(img, **kwargs)
        filename = tempfile.mktemp(suffix='.png')
        try:
            display.savefig(filename)
        finally:
            os.remove(filename)

        # Save execution time and memory
        plt.close()


def test_display_methods():
    img = _generate_img()

    display = plot_img(img)
    display.add_overlay(img, threshold=0)
    display.add_edges(img, color='c')
    display.add_contours(img, contours=2, linewidth=4,
                         colors=['limegreen', 'yellow'])


def test_plot_with_axes_or_figure():
    img = _generate_img()
    figure = plt.figure()
    plot_img(img, figure=figure)

    ax = plt.subplot(111)
    plot_img(img, axes=ax)

    # Save execution time and memory
    plt.close()


def test_plot_stat_map_colorbar_variations():
    # This is only a smoke test
    img_positive = _generate_img()
    data_positive = img_positive.get_data()
    rng = np.random.RandomState(42)
    data_negative = -data_positive
    data_heterogeneous = data_positive * rng.randn(*data_positive.shape)
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


def test_plot_empty_slice():
    # Test that things don't crash when we give a map with nothing above
    # threshold
    # This is only a smoke test
    data = np.zeros((20, 20, 20))
    img = nibabel.Nifti1Image(data, mni_affine)
    plot_img(img, display_mode='y', threshold=1)

    # Save execution time and memory
    plt.close()


def test_plot_img_invalid():
    # Check that we get a meaningful error message when we give a wrong
    # display_mode argument
    assert_raises(Exception, plot_anat, display_mode='zzz')


def test_plot_img_with_auto_cut_coords():
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = nibabel.Nifti1Image(data, np.eye(4))

    for display_mode in 'xyz':
        plot_img(img, cut_coords=None, display_mode=display_mode,
                 black_bg=True)

        # Save execution time and memory
        plt.close()


def test_plot_img_with_resampling():
    data = _generate_img().get_data()
    affine = np.array([[1., -1.,  0.,  0.],
                       [1.,  1.,  0.,  0.],
                       [0.,  0.,  1.,  0.],
                       [0.,  0.,  0.,  1.]])
    img = nibabel.Nifti1Image(data, affine)
    display = plot_img(img)
    display.add_overlay(img)
    display.add_contours(img, contours=2, linewidth=4,
                         colors=['limegreen', 'yellow'])
    display.add_edges(img, color='c')

    # Save execution time and memory
    plt.close()


def test_plot_noncurrent_axes():
    """Regression test for Issue #450"""

    maps_img = nibabel.Nifti1Image(np.random.random((10, 10, 10)), np.eye(4))
    fh1 = plt.figure()
    fh2 = plt.figure()
    ax1 = fh1.add_subplot(1, 1, 1)

    assert_equal(plt.gcf(), fh2, "fh2  was the last plot created.")

    # Since we gave ax1, the figure should be plotted in fh1.
    # Before #451, it was plotted in fh2.
    slicer = plot_glass_brain(maps_img, axes=ax1, title='test')
    for ax_name, niax in slicer.axes.items():
        ax_fh = niax.ax.get_figure()
        assert_equal(ax_fh, fh1, 'New axis %s should be in fh1.' % ax_name)

    # Save execution time and memory
    plt.close()


def test_plot_connectome():
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

    # used to speed-up tests for the next plots
    kwargs['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome(adjacency_matrix,
                    [tuple(each) for each in node_coords],
                    **kwargs)
    # saving to file
    filename = tempfile.mktemp(suffix='.png')
    try:
        display = plot_connectome(*args, output_file=filename, **kwargs)
        assert_true(display is None)
        assert_true(os.path.isfile(filename) and
                    os.path.getsize(filename) > 0)
    finally:
        os.remove(filename)
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


def test_plot_connectome_exceptions():
    node_coords = np.arange(2 * 3).reshape((2, 3))

    # Used to speed-up tests because the glass brain is always plotted
    # before any error occurs
    kwargs = {'display_mode': 'x'}

    # adjacency_matrix is not symmetric
    non_symmetric_adjacency_matrix = np.array([[1., 2],
                                               [0.4, 1.]])
    assert_raises_regex(ValueError,
                        'should be symmetric',
                        plot_connectome,
                        non_symmetric_adjacency_matrix, node_coords,
                        **kwargs)

    adjacency_matrix = np.array([[1., 2.],
                                 [2., 1.]])
    # adjacency_matrix mask is not symmetric
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, [[False, True], [False, False]])

    assert_raises_regex(ValueError,
                        'non symmetric mask',
                        plot_connectome,
                        masked_adjacency_matrix, node_coords,
                        **kwargs)

    # edges threshold is neither a number nor a string
    assert_raises_regex(TypeError,
                        'should be either a number or a string',
                        plot_connectome,
                        adjacency_matrix, node_coords,
                        edge_threshold=object(),
                        **kwargs)

    # wrong shapes for node_coords or adjacency_matrix
    assert_raises_regex(ValueError,
                        r'supposed to have shape \(n, n\).+\(1L?, 2L?\)',
                        plot_connectome, adjacency_matrix[:1, :],
                        node_coords,
                        **kwargs)

    assert_raises_regex(ValueError, r'shape \(2L?, 3L?\).+\(2L?,\)',
                        plot_connectome, adjacency_matrix, node_coords[:, 2],
                        **kwargs)

    wrong_adjacency_matrix = np.zeros((3, 3))
    assert_raises_regex(ValueError, r'Shape mismatch.+\(3L?, 3L?\).+\(2L?, 3L?\)',
                        plot_connectome,
                        wrong_adjacency_matrix, node_coords, **kwargs)

    # a few not correctly formatted strings for 'edge_threshold'
    wrong_edge_thresholds = ['0.1', '10', '10.2.3%', 'asdf%']
    for wrong_edge_threshold in wrong_edge_thresholds:
        assert_raises_regex(ValueError,
                            'should be a number followed by the percent sign',
                            plot_connectome,
                            adjacency_matrix, node_coords,
                            edge_threshold=wrong_edge_threshold, **kwargs)

    # specifying node sizes via node_kwargs
    assert_raises_regex(ValueError,
                        "Please use 'node_size' and not 'node_kwargs'",
                        plot_connectome,
                        adjacency_matrix, node_coords,
                        node_kwargs={'s': 50},
                        **kwargs)

    # specifying node colors via node_kwargs
    assert_raises_regex(ValueError,
                        "Please use 'node_color' and not 'node_kwargs'",
                        plot_connectome,
                        adjacency_matrix, node_coords,
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
    rng = np.random.RandomState(0)
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

    assert_raises_regex(ValueError,
                        'does not accept a "vmin" argument',
                        _get_colorbar_and_data_ranges,
                        data, vmax=None,
                        symmetric_cbar=True, kwargs={'vmin': 1.})


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
    assert_equal(vmin, -np.nanmax(data))
    assert_equal(vmax, np.nanmax(data))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert_equal(vmin, -np.nanmax(data))
    assert_equal(vmax, np.nanmax(data))
    assert_equal(cbar_vmin, np.nanmin(data))
    assert_equal(cbar_vmax, np.nanmax(data))
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, np.nanmin(data))
    assert_equal(cbar_vmax, np.nanmax(data))

    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert_equal(vmin, -np.nanmax(data))
    assert_equal(vmax, np.nanmax(data))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)


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
    assert_equal(vmin, -np.nanmax(data_pos))
    assert_equal(vmax, np.nanmax(data_pos))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar=True,
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None,
        symmetric_cbar=False,
        kwargs={})
    assert_equal(vmin, -np.nanmax(data_pos))
    assert_equal(vmax, np.nanmax(data_pos))
    assert_equal(cbar_vmin, 0)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar=False,
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, 0)
    assert_equal(cbar_vmax, None)

    # symmetric_cbar is set to 'auto', same behaviour as false in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None,
        symmetric_cbar='auto',
        kwargs={})
    assert_equal(vmin, -np.nanmax(data_pos))
    assert_equal(vmax, np.nanmax(data_pos))
    assert_equal(cbar_vmin, 0)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2,
        symmetric_cbar='auto',
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, 0)
    assert_equal(cbar_vmax, None)


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
    assert_equal(vmin, np.nanmin(data_neg))
    assert_equal(vmax, -np.nanmin(data_neg))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar=True,
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None,
        symmetric_cbar=False,
        kwargs={})
    assert_equal(vmin, np.nanmin(data_neg))
    assert_equal(vmax, -np.nanmin(data_neg))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, 0)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar=False,
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, 0)

    # symmetric_cbar is set to 'auto', same behaviour as False in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None,
        symmetric_cbar='auto',
        kwargs={})
    assert_equal(vmin, np.nanmin(data_neg))
    assert_equal(vmax, -np.nanmin(data_neg))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, 0)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2,
        symmetric_cbar='auto',
        kwargs={})
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, 0)


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
    assert_equal(vmin, -np.nanmax(filled_data))
    assert_equal(vmax, np.nanmax(filled_data))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar=True,
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)

    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert_equal(vmin, -np.nanmax(filled_data))
    assert_equal(vmax, np.nanmax(filled_data))
    assert_equal(cbar_vmin, np.nanmin(filled_data))
    assert_equal(cbar_vmax, np.nanmax(filled_data))
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar=False,
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, np.nanmin(filled_data))
    assert_equal(cbar_vmax, np.nanmax(filled_data))

    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert_equal(vmin, -np.nanmax(filled_data))
    assert_equal(vmax, np.nanmax(filled_data))
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2,
        symmetric_cbar='auto',
        kwargs=kwargs)
    assert_equal(vmin, -2)
    assert_equal(vmax, 2)
    assert_equal(cbar_vmin, None)
    assert_equal(cbar_vmax, None)


def test_invalid_in_display_mode_cut_coords_all_plots():
    img = _generate_img()

    for plot_func in [plot_img, plot_anat, plot_roi, plot_epi,
                      plot_stat_map, plot_prob_atlas, plot_glass_brain]:
        assert_raises_regex(ValueError,
                            "The input given for display_mode='ortho' needs to "
                            "be a list of 3d world coordinates.",
                            plot_func,
                            img, display_mode='ortho', cut_coords=2)


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
    x_map, y_map, z_map = coord_transform(x, y, z,
                                          np.linalg.inv(affine))

    data[int(x_map) - 1:int(x_map) + 1,
         int(y_map) - 1:int(y_map) + 1,
         int(z_map) - 1:int(z_map) + 1] = 1
    img = nibabel.Nifti1Image(data, affine)
    cuts = find_cut_slices(img, n_cuts=20, direction='z')

    p = plot_stat_map(img, display_mode='z', cut_coords=cuts[-4:],
                      bg_img=bg_img)


def test_plot_stat_map_with_nans():
    img = _generate_img()
    data = img.get_data()

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
    # some cmaps such as 'viridis' (the new default in 2.0), 'magma', 'plasma',
    # and 'inferno' are not supported for older matplotlib version from < 1.5
    cmaps = ['Paired', 'Set1', 'Set2', 'Set3']
    for cmap in cmaps:
        plot_roi(img, cmap=cmap, colorbar=True)
        plot_stat_map(img, cmap=cmap, colorbar=True)
        plot_glass_brain(img, cmap=cmap, colorbar=True)

    if LooseVersion(matplotlib.__version__) >= LooseVersion('2.0.0'):
        plot_stat_map(img, cmap='viridis', colorbar=True)

    plt.close()


def test_plotting_functions_with_nans_in_bg_img():
    bg_img = _generate_img()
    bg_data = bg_img.get_data()

    bg_data[6, 5, 1] = np.nan
    bg_data[1, 5, 2] = np.nan
    bg_data[1, 3, 2] = np.nan
    bg_data[6, 5, 2] = np.inf

    bg_img = nibabel.Nifti1Image(bg_data, mni_affine)
    plot_anat(bg_img)
    # test with plot_roi passing background image which contains nans values
    # in it
    roi_img = _generate_img()
    plot_roi(roi_img=roi_img, bg_img=bg_img)
    stat_map_img = _generate_img()
    plot_stat_map(stat_map_img=stat_map_img, bg_img=bg_img)

    plt.close()
