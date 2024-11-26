"""Tests for :func:`nilearn.plotting.plot_markers`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nilearn.conftest import _rng
from nilearn.plotting import plot_markers


@pytest.fixture
def coords():
    """Node coordinates for testing."""
    return np.array(
        [[39, 6, -32], [29, 40, 1], [-20, -74, 35], [-29, -59, -37]]
    )


@pytest.mark.parametrize(
    "node_values",
    [
        [1, 2, 3, 4],
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 3, 4])[:, np.newaxis],
        np.array([1, 2, 3, 4])[np.newaxis, :],
        (1, 1, 1, 1),
    ],
)
def test_plot_markers_node_values(node_values, coords):
    """Smoke test for plot_markers with different node values."""
    plot_markers(node_values, coords, display_mode="x")
    plt.close()


@pytest.mark.parametrize(
    "node_size", [10, [10, 20, 30, 40], np.array([10, 20, 30, 40])]
)
def test_plot_markers_node_sizes(node_size, coords):
    """Smoke test for plot_markers with different node sizes."""
    plot_markers([1, 2, 3, 4], coords, node_size=node_size, display_mode="x")
    plt.close()


@pytest.mark.parametrize(
    "node_size", [[10] * 4, [10, 20, 30, 40], np.array([10, 20, 30, 40])]
)
def test_plot_markers_node_sizes_lyrz_display(node_size, coords):
    """Tests for plot_markers and 'lyrz' display mode.

    Tests that markers are plotted with the requested size
    with display_mode='lyrz'. (See issue #3012 and PR #3013).
    """
    display = plot_markers(
        [1, 2, 3, 4], coords, display_mode="lyrz", node_size=node_size
    )
    for d, axes in display.axes.items():
        display_sizes = axes.ax.collections[0].get_sizes()
        if d == "l":
            expected_sizes = node_size[-2:]
        elif d == "r":
            expected_sizes = node_size[:-2]
        else:
            expected_sizes = node_size
        assert np.all(display_sizes == expected_sizes)
    plt.close()


@pytest.mark.parametrize(
    "cmap,vmin,vmax",
    [
        ("RdBu", 0, None),
        (plt.get_cmap("jet"), None, 5),
        (plt.cm.viridis_r, 2, 3),
    ],
)
def test_plot_markers_cmap(cmap, vmin, vmax, coords):
    """Smoke test for plot_markers with different cmaps."""
    plot_markers(
        [1, 2, 3, 4],
        coords,
        node_cmap=cmap,
        node_vmin=vmin,
        node_vmax=vmax,
        display_mode="x",
    )
    plt.close()


@pytest.mark.parametrize("threshold", [-100, 2.5])
def test_plot_markers_threshold(threshold, coords):
    """Smoke test for plot_markers with different threshold values."""
    plot_markers(
        [1, 2, 3, 4], coords, node_threshold=threshold, display_mode="x"
    )
    plt.close()


def test_plot_markers_tuple_node_coords(coords):
    """Smoke test for plot_markers with node coordinates passed \
       as a list of tuples.
    """
    plot_markers(
        [1, 2, 3, 4], [tuple(coord) for coord in coords], display_mode="x"
    )
    plt.close()


def test_plot_markers_saving_to_file(coords, tmp_path):
    """Smoke test for plot_markers and file saving."""
    filename = tmp_path / "test.png"
    display = plot_markers(
        [1, 2, 3, 4], coords, output_file=filename, display_mode="x"
    )
    assert display is None
    assert filename.is_file() and filename.stat().st_size > 0

    plt.close()


def test_plot_markers_node_kwargs(coords):
    """Smoke test for plot_markers testing that node_kwargs is working \
       and does not interfere with alpha.
    """
    node_kwargs = {"marker": "s"}
    plot_markers(
        [1, 2, 3, 4],
        coords,
        alpha=0.1,
        node_kwargs=node_kwargs,
        display_mode="x",
    )
    plt.close()


@pytest.mark.parametrize(
    "matrix",
    [
        [1, 2, 3, 4, 5],
        [1, 2, 3],
        _rng().random((4, 4)),
    ],
)
def test_plot_markers_dimension_mismatch(matrix, coords):
    """Tests that an error is raised in plot_markers \
       when the length of node_values mismatches with node_coords.
    """
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_markers(matrix, coords, display_mode="x")


@pytest.mark.parametrize("vmin,vmax", [(5, None), (None, 0)])
def test_plot_markers_bound_error(vmin, vmax, coords):
    """Tests that a ValueError is raised when vmin and vmax \
       have inconsistent values.
    """
    with pytest.raises(ValueError):
        plot_markers(
            [1, 2, 2, 4],
            coords,
            node_vmin=vmin,
            node_vmax=vmax,
            display_mode="x",
        )


def test_plot_markers_node_values_errors(coords):
    """Tests that a TypeError is raised when node_values is wrong type."""
    with pytest.raises(TypeError):
        plot_markers(["1", "2", "3", "4"], coords, display_mode="x")


def test_plot_markers_threshold_errors(coords):
    """Tests that a ValueError is raised when node_threshold is \
       higher than the max node_value.
    """
    with pytest.raises(ValueError, match="Provided 'node_threshold' value"):
        plot_markers([1, 2, 2, 4], coords, node_threshold=5, display_mode="x")


def test_plot_markers_single_node_value():
    """Regression test for Issue #3253."""
    plot_markers([1], [[1, 1, 1]])
    plt.close()


def test_plot_markers_radiological_view():
    """Smoke test for radiological view."""
    result = plot_markers([1], [[1, 1, 1]], radiological=True)
    assert result.axes.get("y").radiological is True
    plt.close()
