"""
Tests for :func:`nilearn.plotting.plot_markers`.
"""

import os
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_markers


@pytest.fixture
def coords():
    return np.array([[39 ,   6, -32],
                     [29 ,  40,   1],
                     [-20, -74,  35],
                     [-29, -59, -37]])


@pytest.mark.parametrize("node_values",
                         [[1, 2, 3, 4],
                          np.array([1, 2, 3, 4]),
                          np.array([1, 2, 3, 4])[:, np.newaxis],
                          np.array([1, 2, 3, 4])[np.newaxis, :],
                          (1, 1, 1, 1)])
def test_plot_markers_node_values(node_values, coords):
    plot_markers(node_values, coords, display_mode='x')
    plt.close()


@pytest.mark.parametrize("node_size",
                         [10,
                          [10, 20, 30, 40],
                          np.array([10, 20, 30, 40])])
def test_plot_markers_node_sizes(node_size, coords):
    plot_markers([1, 2, 3, 4], coords, node_size=node_size, display_mode='x')
    plt.close()


@pytest.mark.parametrize("cmap,vmin,vmax",
                         [('RdBu', 0, None),
                          (matplotlib.cm.get_cmap('jet'), None, 5),
                          (plt.cm.viridis_r, 2, 3)])
def test_plot_markers_cmap(cmap, vmin, vmax, coords):
    plot_markers([1, 2, 3, 4], coords, node_cmap=cmap, node_vmin=vmin,
                 node_vmax=vmax, display_mode='x')
    plt.close()


@pytest.mark.parametrize("threshold", [-100, 2.5])
def test_plot_markers_threshold(threshold, coords):
    plot_markers([1, 2, 3, 4], coords, node_threshold=threshold,
                 display_mode='x')
    plt.close()


def test_plot_markers(coords, tmpdir):
    node_values = [1, 2, 3, 4]
    args = node_values, coords
    # node_coords not an array but a list of tuples
    plot_markers(node_values, [tuple(coord) for coord in coords],
                 display_mode='x')
    # Saving to file
    filename = str(tmpdir.join('test.png'))
    display = plot_markers(*args, output_file=filename, display_mode='x')
    assert display is None
    assert (os.path.isfile(filename) and  # noqa: W504
                os.path.getsize(filename) > 0)
    # node_kwargs working and does not interfere with alpha
    node_kwargs = dict(marker='s')
    plot_markers(*args, alpha=.1, node_kwargs=node_kwargs, display_mode='x')
    plt.close()


@pytest.mark.parametrize("matrix",
                         [[1, 2, 3, 4, 5],
                          [1, 2, 3],
                          np.random.RandomState(42).random_sample((4, 4))])
def test_plot_markers_dimension_mismatch(matrix, coords):
    # node_values length mismatch with node_coords
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_markers(matrix, coords, display_mode='x')


@pytest.mark.parametrize("vmin,vmax", [(5, None), (None, 0)])
def test_plot_markers_bound_error(vmin, vmax, coords):
    with pytest.raises(ValueError):
        plot_markers([1, 2, 2, 4], coords, node_vmin=vmin,
                     node_vmax=vmax, display_mode='x')


def test_plot_markers_errors(coords):
    # node_values is wrong type
    with pytest.raises(TypeError):
        plot_markers(['1', '2', '3', '4'], coords, display_mode='x')
    # node_threshold higher than max node_value
    with pytest.raises(ValueError,
                       match="Provided 'node_threshold' value"):
        plot_markers([1, 2, 2, 4], coords, node_threshold=5, display_mode='x')