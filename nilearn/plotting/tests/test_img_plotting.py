
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile
import os
from functools import partial

import numpy as np
from scipy import sparse

from nose.tools import assert_raises, assert_true, assert_equal

import matplotlib.pyplot as plt

import nibabel

from nilearn.image.resampling import coord_transform

from nilearn.plotting.img_plotting import (MNI152TEMPLATE, plot_anat, plot_img,
                                           plot_roi, plot_stat_map, plot_epi,
                                           plot_glass_brain, plot_connectome)
from nilearn._utils.testing import assert_raises_regex

mni_affine = np.array([[  -2.,    0.,    0.,   90.],
                       [   0.,    2.,    0., -126.],
                       [   0.,    0.,    2.,  -72.],
                       [   0.,    0.,    0.,    1.]])


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

    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        out = demo_plot_roi(output_file=fp)
    assert_true(out is None)


def test_plot_anat():
    img = _generate_img()

    # Test saving with empty plot
    z_slicer = plot_anat(anat_img=False, display_mode='z')
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        z_slicer.savefig(fp.name)
    z_slicer = plot_anat(display_mode='z')
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        z_slicer.savefig(fp.name)

    ortho_slicer = plot_anat(img, dim=True)
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        ortho_slicer.savefig(fp.name)


def test_plot_functions():
    img = _generate_img()

    # smoke-test for each plotting function with default arguments
    for plot_func in [plot_anat, plot_img, plot_stat_map, plot_epi,
                      plot_glass_brain]:
        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            plot_func(img, output_file=fp.name)

    # test for bad input arguments (cf. #510)
    ax = plt.subplot(111, rasterized=True)
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        plot_stat_map(
            img, symmetric_cbar=True,
            output_file=fp.name,
            axes=ax, vmax=np.nan)
    plt.close()


def test_plot_glass_brain():
    img = _generate_img()

    # test plot_glass_brain with colorbar
    plot_glass_brain(img, colorbar=True)


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


def test_save_plot():
    img = _generate_img()

    kwargs_list = [{}, {'display_mode': 'x', 'cut_coords': 3}]

    for kwargs in kwargs_list:
        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            display = plot_stat_map(img, output_file=fp.name, **kwargs)
            assert_true(display is None)

        display = plot_stat_map(img, **kwargs)
        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            display.savefig(fp.name)


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


def test_plot_img_with_resampling():
    data = _generate_img().get_data()
    affine = np.array([[1., -1.,  0.,  0.],
                       [1.,  1.,  0.,  0.],
                       [0.,  0.,  1.,  0.],
                       [0.,  0.,  0.,  1.]])
    img = nibabel.Nifti1Image(data, affine)
    display = plot_img(img)
    display.add_overlay(img)


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

    # used to speed-up tests for the next plots
    kwargs['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome(adjacency_matrix,
                    [tuple(each) for each in node_coords],
                    **kwargs)
    # saving to file
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        display = plot_connectome(*args, output_file=fp.name,
                                  **kwargs)
        assert_true(display is None)
        assert_true(os.path.isfile(fp.name) and
                    os.path.getsize(fp.name) > 0)

    # with node_kwargs, edge_kwargs and edge_cmap arguments
    plot_connectome(*args,
                    edge_threshold='70%',
                    node_size=[10, 20, 30, 40],
                    node_color=np.zeros((4, 3)),
                    edge_cmap='RdBu',
                    node_kwargs={
                        'marker': 'v'},
                    edge_kwargs={
                        'linewidth': 4})

    # masked array support
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency_matrix, np.abs(adjacency_matrix) < 0.5)
    plot_connectome(masked_adjacency_matrix, node_coords,
                    **kwargs)

    # sparse matrix support
    sparse_adjacency_matrix = sparse.coo_matrix(adjacency_matrix)
    plot_connectome(sparse_adjacency_matrix, node_coords,
                    **kwargs)

    # NaN matrix support
    nan_adjacency_matrix = np.array([[1., np.nan, 0.],
                                     [np.nan, 1., 2.],
                                     [np.nan, 2., 1.]])
    nan_node_coords = np.arange(3 * 3).reshape(3, 3)
    plot_connectome(nan_adjacency_matrix, nan_node_coords, **kwargs)

    # smoke-test where there is no edge to draw, e.g. when
    # edge_threshold is too high
    plot_connectome(*args, edge_threshold=1e12)


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
                        r'supposed to have shape \(n, n\).+\(1, 2\)',
                        plot_connectome, adjacency_matrix[:1, :],
                        node_coords,
                        **kwargs)

    assert_raises_regex(ValueError, r'shape \(2, 3\).+\(2,\)',
                        plot_connectome, adjacency_matrix, node_coords[:, 2],
                        **kwargs)

    wrong_adjacency_matrix = np.zeros((3, 3))
    assert_raises_regex(ValueError, r'Shape mismatch.+\(3, 3\).+\(2, 3\)',
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
