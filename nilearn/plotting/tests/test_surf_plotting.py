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
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest

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

    # Save execution time and memory
    plt.close()


def test_plot_surf_error():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
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
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest

    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Plot mesh with stat map
    plot_surf_stat_map(mesh, stat_map=data)
    plot_surf_stat_map(mesh, stat_map=data, alpha=1)

    # Plot mesh with background and stat map
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_stat=True, darkness=0.5)

    # Apply threshold
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_stat=True, darkness=0.5,
                       threshold=0.3)

    # Change vmax
    plot_surf_stat_map(mesh, stat_map=data, vmax=5)

    # Change colormap
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix')

    # Plot to axes
    axes = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(mesh, stat_map=data, ax=ax)

    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
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
    # Axes3DSubplot has no attribute 'plot_trisurf' for older versions of
    # matplotlib
    if LooseVersion(matplotlib.__version__) <= LooseVersion('1.3.1'):
        raise SkipTest
    mesh = _generate_surf()
    rng = np.random.RandomState(0)
    roi1 = rng.randint(0, mesh[0].shape[0], size=5)
    roi2 = rng.randint(0, mesh[0].shape[0], size=10)
    parcellation = rng.rand(mesh[0].shape[0])

    # plot roi
    plot_surf_roi(mesh, roi_map=roi1)

    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation)

    # plot roi list
    plot_surf_roi(mesh, roi_map=[roi1, roi2])

    # plot to axes
    plot_surf_roi(mesh, roi_map=roi1, ax=None, figure=plt.gcf())

    # plot to axes
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi1, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name)

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
