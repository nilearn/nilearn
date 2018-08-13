# Tests for functions in surf_plotting.py

import tempfile

from distutils.version import LooseVersion
from nose import SkipTest
from nilearn._utils.testing import assert_raises_regex

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from nilearn.plotting.surf_plotting import (plot_surf, plot_surf_stat_map,
                                            plot_surf_roi)
from nilearn.surface.tests.test_surface import _generate_surf


def test_plot_surf():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )

    # Plot mesh only
    plot_surf(mesh)

    # Plot mesh with background
    plot_surf(mesh, bg_map=bg)
    plot_surf(mesh, bg_map=bg, darkness=0.5)
    plot_surf(mesh, bg_map=bg, alpha=0.5)

    # Plot different views
    plot_surf(mesh, bg_map=bg, hemi='right')
    plot_surf(mesh, bg_map=bg, view='medial')
    plot_surf(mesh, bg_map=bg, hemi='right', view='medial')

    # Plot with colorbar
    plot_surf(mesh, bg_map=bg, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_surf_error():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)

    # Wrong inputs for view or hemi
    assert_raises_regex(ValueError, 'view must be one of',
                        plot_surf, mesh, view='middle')
    assert_raises_regex(ValueError, 'hemi must be one of',
                        plot_surf, mesh, hemi='lft')

    # Wrong size of background image
    assert_raises_regex(ValueError,
                        'bg_map does not have the same number of vertices',
                        plot_surf, mesh,
                        bg_map=rng.randn(mesh[0].shape[0] - 1, ))

    # Wrong size of surface data
    assert_raises_regex(ValueError,
                        'surf_map does not have the same number of vertices',
                        plot_surf, mesh,
                        surf_map=rng.randn(mesh[0].shape[0] + 1, ))

    assert_raises_regex(ValueError,
                        'surf_map can only have one dimension', plot_surf,
                        mesh, surf_map=rng.randn(mesh[0].shape[0], 2))


def test_plot_surf_stat_map():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Plot mesh with stat map
    plot_surf_stat_map(mesh, stat_map=data)
    plot_surf_stat_map(mesh, stat_map=data, colorbar=True)
    plot_surf_stat_map(mesh, stat_map=data, alpha=1)

    # Plot mesh with background and stat map
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5)

    # Apply threshold
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3)

    # Change vmax
    plot_surf_stat_map(mesh, stat_map=data, vmax=5)
    plot_surf_stat_map(mesh, stat_map=data, vmax=5, colorbar=True)

    # Change colormap
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix')
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix', colorbar=True)

    # Plot to axes
    axes = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(mesh, stat_map=data, ax=ax)
    axes = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(mesh, stat_map=data, ax=ax, colorbar=True)

    fig = plot_surf_stat_map(mesh, stat_map=data, colorbar=False)
    assert len(fig.axes) == 1
    # symmetric_cbar
    fig = plot_surf_stat_map(
        mesh, stat_map=data, colorbar=True, symmetric_cbar=True)
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) == - float(last)
    # no symmetric_cbar
    fig = plot_surf_stat_map(
        mesh, stat_map=data, colorbar=True, symmetric_cbar=False)
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) != - float(last)
    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Try to input vmin
    assert_raises_regex(ValueError,
                        'this function does not accept a "vmin" argument',
                        plot_surf_stat_map, mesh, stat_map=data, vmin=0)

    # Wrong size of stat map data
    assert_raises_regex(ValueError,
                        'surf_map does not have the same number of vertices',
                        plot_surf_stat_map, mesh,
                        stat_map=np.hstack((data, data)))

    assert_raises_regex(ValueError,
                        'surf_map can only have one dimension',
                        plot_surf_stat_map, mesh,
                        stat_map=np.vstack((data, data)).T)


def test_plot_surf_roi():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    roi1 = rng.randint(0, mesh[0].shape[0], size=5)
    roi2 = rng.randint(0, mesh[0].shape[0], size=10)
    parcellation = rng.rand(mesh[0].shape[0])

    # plot roi
    plot_surf_roi(mesh, roi_map=roi1)
    plot_surf_roi(mesh, roi_map=roi1, colorbar=True)

    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation)
    plot_surf_roi(mesh, roi_map=parcellation, colorbar=True)

    # plot roi list
    plot_surf_roi(mesh, roi_map=[roi1, roi2])
    plot_surf_roi(mesh, roi_map=[roi1, roi2], colorbar=True)

    # plot to axes
    plot_surf_roi(mesh, roi_map=roi1, ax=None, figure=plt.gcf())

    # plot to axes
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi1, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi1, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_surf_roi_error():
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    roi1 = rng.randint(0, mesh[0].shape[0], size=5)
    roi2 = rng.randint(0, mesh[0].shape[0], size=10)

    # Wrong input
    assert_raises_regex(ValueError,
                        'Invalid input for roi_map',
                        plot_surf_roi, mesh,
                        roi_map={'roi1': roi1, 'roi2': roi2})
