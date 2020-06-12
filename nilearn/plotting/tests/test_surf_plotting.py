# Tests for functions in surf_plotting.py

import tempfile


import numpy as np
import pytest

from matplotlib import pyplot as plt

from nilearn.plotting.surf_plotting import (plot_surf, plot_surf_stat_map,
                                            plot_surf_roi, plot_surf_contours)
from nilearn.surface.testing_utils import generate_surf


def test_plot_surf():
    mesh = generate_surf()
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
    mesh = generate_surf()
    rng = np.random.RandomState(0)

    # Wrong inputs for view or hemi
    with pytest.raises(ValueError, match='view must be one of'):
        plot_surf(mesh, view='middle')
    with pytest.raises(ValueError, match='hemi must be one of'):
        plot_surf(mesh, hemi='lft')

    # Wrong size of background image
    with pytest.raises(
            ValueError,
            match='bg_map does not have the same number of vertices'):
        plot_surf(mesh, bg_map=rng.randn(mesh[0].shape[0] - 1, ))

    # Wrong size of surface data
    with pytest.raises(
            ValueError,
            match='surf_map does not have the same number of vertices'):
        plot_surf(mesh, surf_map=rng.randn(mesh[0].shape[0] + 1, ))

    with pytest.raises(
            ValueError,
            match='surf_map can only have one dimension'):
        plot_surf(mesh, surf_map=rng.randn(mesh[0].shape[0], 2))


def test_plot_surf_stat_map():
    mesh = generate_surf()
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
    fig.canvas.draw()
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) == - float(last)
    # no symmetric_cbar
    fig = plot_surf_stat_map(
        mesh, stat_map=data, colorbar=True, symmetric_cbar=False)
    fig.canvas.draw()
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) != - float(last)
    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Try to input vmin
    with pytest.raises(
            ValueError,
            match='this function does not accept a "vmin" argument'):
        plot_surf_stat_map(mesh, stat_map=data, vmin=0)

    # Wrong size of stat map data
    with pytest.raises(
            ValueError,
            match='surf_map does not have the same number of vertices'):
        plot_surf_stat_map(mesh, stat_map=np.hstack((data, data)))

    with pytest.raises(
            ValueError,
            match='surf_map can only have one dimension'):
        plot_surf_stat_map(mesh, stat_map=np.vstack((data, data)).T)


def test_plot_surf_roi():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=10)
    roi_map = np.zeros(mesh[0].shape[0])
    roi_map[roi_idx] = 1
    parcellation = rng.rand(mesh[0].shape[0])

    # plot roi
    plot_surf_roi(mesh, roi_map=roi_map)
    plot_surf_roi(mesh, roi_map=roi_map, colorbar=True)
    # change vmin, vmax
    img = plot_surf_roi(mesh, roi_map=roi_map,
						vmin=1.2, vmax=8.9, colorbar=True)
    img.canvas.draw()
    cbar = img.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())
    assert cbar_vmin == 1.2
    assert cbar_vmax == 8.9

    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation)
    plot_surf_roi(mesh, roi_map=parcellation, colorbar=True)

    # plot to axes
    plot_surf_roi(mesh, roi_map=roi_map, ax=None, figure=plt.gcf())

    # plot to axes
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_surf_roi_error():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=5)

    # Wrong input
    with pytest.raises(
            ValueError,
            match='roi_map does not have the same number of vertices'):
        plot_surf_roi(mesh, roi_map=roi_idx)

def test_plot_surf_contours():
    mesh = generate_surf()
    # we need a valid parcellation for testing
    parcellation = np.zeros((mesh[0].shape[0],))
    parcellation[mesh[1][3]] = 1
    parcellation[mesh[1][5]] = 2
    plot_surf_contours(mesh, parcellation)
    plot_surf_contours(mesh, parcellation, levels=[1, 2])
    plot_surf_contours(mesh, parcellation, levels=[1, 2], cmap='gist_ncar')
    plot_surf_contours(mesh, parcellation, levels=[1, 2],
                       colors=['r', 'g'])
    plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r', 'g'],
                       labels=['1', '2'])
    fig = plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r', 'g'],
                       labels=['1', '2'], legend=True)
    assert fig.legends is not None
    plot_surf_contours(mesh, parcellation, levels=[1, 2],
                       colors=[[0, 0, 0, 1], [1, 1, 1, 1]])
    fig, axes = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_surf_contours(mesh, parcellation, axes=axes)
    plot_surf_contours(mesh, parcellation, figure=fig)
    fig = plot_surf(mesh)
    plot_surf_contours(mesh, parcellation, figure=fig)
    plot_surf_contours(mesh, parcellation, title='title')
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_contours(mesh, parcellation, output_file=tmp_file.name)
    plt.close()

def test_plot_surf_contours_error():
    mesh = generate_surf()
    # we need an invalid parcellation for testing
    rng = np.random.RandomState(0)
    invalid_parcellation = rng.rand(mesh[0].shape[0])
    parcellation = np.zeros((mesh[0].shape[0],))
    parcellation[mesh[1][3]] = 1
    parcellation[mesh[1][5]] = 2
    with pytest.raises(
            ValueError,
            match='Vertices in parcellation do not form region.'):
        plot_surf_contours(mesh, invalid_parcellation)
    fig, axes = plt.subplots(1, 1)
    with pytest.raises(
            ValueError,
            match='Axes must be 3D.'):
        plot_surf_contours(mesh, parcellation, axes=axes)
    with pytest.raises(
            ValueError,
            match='All elements of colors need to be either a matplotlib color string or RGBA values.'):
        plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=[[1, 2], 3])
    with pytest.raises(
            ValueError,
            match='Levels, labels, and colors argument need to be either the same length or None.'):
        plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r'], labels=['1', '2'])
