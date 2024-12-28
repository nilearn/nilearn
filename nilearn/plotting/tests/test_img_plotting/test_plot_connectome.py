"""Tests for :func:`nilearn.plotting.plot_connectome`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyArrow
from scipy import sparse

from nilearn.plotting import plot_connectome


@pytest.fixture
def adjacency():
    """Adjacency matrix symmetric up to 1e-3 relative tolerance."""
    return np.array(
        [
            [1.0, -2.0, 0.3, 0.0],
            [-2.002, 1, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def non_symmetric_matrix():
    """Non symmetric adjacency matrix."""
    return np.array(
        [
            [1.0, -2.0, 0.3, 0.2],
            [0.1, 1, 1.1, 0.1],
            [0.01, 2.3, 1.0, 3.1],
            [0.6, 0.03, 1.2, 1.0],
        ]
    )


@pytest.fixture
def base_params():
    """Return basic set of parameters for testing."""
    return {"edge_threshold": 0.38, "title": "threshold=0.38", "node_size": 10}


@pytest.fixture
def node_coords():
    """Array of node coordinates for testing."""
    return np.arange(3 * 4).reshape(4, 3)


@pytest.mark.parametrize(
    "node_color",
    [["green", "blue", "k", "cyan"], np.array(["red"]), ["red"], "green"],
)
@pytest.mark.parametrize("display_mode", ["ortho", "lzry"])
def test_plot_connectome_node_colors(
    node_color, display_mode, node_coords, adjacency, base_params
):
    """Smoke test for plot_connectome with different values for node_color."""
    plot_connectome(
        adjacency,
        node_coords,
        node_color=node_color,
        display_mode=display_mode,
        **base_params,
    )
    plt.close()


@pytest.mark.parametrize(
    "display_mode",
    [
        "ortho",
        "x",
        "y",
        "z",
        "xz",
        "yx",
        "yz",
        "l",
        "r",
        "lr",
        "lzr",
        "lyr",
        "lzry",
        "lyrz",
    ],
)
def test_plot_connectome_display_mode(
    display_mode, node_coords, adjacency, base_params
):
    """Smoke test for plot_connectome with different values \
       for display_mode.
    """
    plot_connectome(
        adjacency, node_coords, display_mode=display_mode, **base_params
    )
    plt.close()


def test_plot_connectome_masked_array_sparse_matrix(
    node_coords, adjacency, base_params
):
    """Smoke tests for plot_connectome with masked arrays \
       and sparse matrices as inputs.
    """
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency, np.abs(adjacency) < 0.5
    )
    plot_connectome(masked_adjacency_matrix, node_coords, **base_params)
    sparse_adjacency_matrix = sparse.coo_matrix(adjacency)
    plot_connectome(sparse_adjacency_matrix, node_coords, **base_params)
    plt.close()


def test_plot_connectome_with_nans(adjacency, node_coords, base_params):
    """Smoke test for plot_connectome with nans in the adjacency matrix."""
    adjacency[0, 1] = np.nan
    adjacency[1, 0] = np.nan
    base_params["node_color"] = np.array(["green", "blue", "k", "yellow"])
    plot_connectome(adjacency, node_coords, **base_params)
    plt.close()


def test_plot_connectome_tuple_node_coords(
    adjacency, node_coords, base_params
):
    """Smoke test for plot_connectome where node_coords is not provided \
       as an array but as a list of tuples.
    """
    plot_connectome(
        adjacency,
        [tuple(each) for each in node_coords],
        display_mode="x",
        **base_params,
    )
    plt.close()


@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_connectome_colorbar(colorbar, adjacency, node_coords):
    """Smoke test for plot_connectome with default parameters \
       and with and without the colorbar.
    """
    plot_connectome(adjacency, node_coords, colorbar=colorbar)
    plt.close()


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
def test_plot_connectome_alpha(alpha, adjacency, node_coords):
    """Smoke test for plot_connectome with various alpha values."""
    plot_connectome(adjacency, node_coords, alpha=alpha)
    plt.close()


def test_plot_connectome_to_file(
    adjacency, node_coords, base_params, tmp_path
):
    """Smoke test for plot_connectome and saving to file."""
    base_params["display_mode"] = "x"
    filename = tmp_path / "temp.png"
    display = plot_connectome(
        adjacency, node_coords, output_file=filename, **base_params
    )
    assert display is None
    assert filename.is_file()
    assert filename.stat().st_size > 0

    plt.close()


def test_plot_connectome_node_and_edge_kwargs(adjacency, node_coords):
    """Smoke test for plot_connectome with node_kwargs, edge_kwargs, \
       and edge_cmap arguments.
    """
    plot_connectome(
        adjacency,
        node_coords,
        edge_threshold="70%",
        node_size=[10, 20, 30, 40],
        node_color=np.zeros((4, 3)),
        edge_cmap="RdBu",
        colorbar=True,
        node_kwargs={"marker": "v"},
        edge_kwargs={"linewidth": 4},
    )
    plt.close()


def test_plot_connectome_with_too_high_edge_threshold(adjacency, node_coords):
    """Smoke-test where there is no edge to draw, \
       e.g. when edge_threshold is too high.
    """
    plot_connectome(adjacency, node_coords, edge_threshold=1e12)
    plt.close()


def test_plot_connectome_non_symmetric(node_coords, non_symmetric_matrix):
    """Tests for plot_connectome with non symmetric adjacency matrices."""
    ax = plot_connectome(
        non_symmetric_matrix, node_coords, display_mode="ortho"
    )
    # No thresholding was performed, we should get
    # as many arrows as we have edges
    for direction in ["x", "y", "z"]:
        assert len(
            [
                patch
                for patch in ax.axes[direction].ax.patches
                if isinstance(patch, FancyArrow)
            ]
        ) == np.prod(non_symmetric_matrix.shape)

    # Set a few elements of adjacency matrix to zero
    non_symmetric_matrix[1, 0] = 0.0
    non_symmetric_matrix[2, 3] = 0.0
    # Plot with different display mode
    ax = plot_connectome(
        non_symmetric_matrix, node_coords, display_mode="lzry"
    )
    # No edge in direction 'l' because of node coords
    assert not [
        patch
        for patch in ax.axes["l"].ax.patches
        if isinstance(patch, FancyArrow)
    ]
    for direction in ["z", "r", "y"]:
        assert (
            len(
                [
                    patch
                    for patch in ax.axes[direction].ax.patches
                    if isinstance(patch, FancyArrow)
                ]
            )
            == np.prod(non_symmetric_matrix.shape) - 2
        )


def plot_connectome_edge_thresholding(node_coords, non_symmetric_matrix):
    """Test for plot_connectome with edge thresholding."""
    # Case 1: Threshold is a number
    thresh = 1.1
    ax = plot_connectome(
        non_symmetric_matrix, node_coords, edge_threshold=thresh
    )
    for direction in ["x", "y", "z"]:
        assert len(
            [
                patch
                for patch in ax.axes[direction].ax.patches
                if isinstance(patch, FancyArrow)
            ]
        ) == np.sum(np.abs(non_symmetric_matrix) >= thresh)
    # Case 2: Threshold is a percentage
    thresh = 80
    ax = plot_connectome(
        non_symmetric_matrix, node_coords, edge_threshold=f"{thresh}%"
    )
    for direction in ["x", "y", "z"]:
        assert len(
            [
                patch
                for patch in ax.axes[direction].ax.patches
                if isinstance(patch, FancyArrow)
            ]
        ) == np.sum(
            np.abs(non_symmetric_matrix)
            >= np.percentile(np.abs(non_symmetric_matrix.ravel()), thresh)
        )
    plt.close()


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, 2], [0.4, 1.0]]),
        np.ma.masked_array(
            np.array([[1.0, 2.0], [2.0, 1.0]]), [[False, True], [False, False]]
        ),
    ],
)
def test_plot_connectome_exceptions_non_symmetric_adjacency(matrix):
    """Tests that warning messages are given when the adjacency matrix \
       ends up being non symmetric.
    """
    node_coords = np.arange(2 * 3).reshape((2, 3))
    with pytest.warns(UserWarning, match="A directed graph will be plotted."):
        plot_connectome(matrix, node_coords, display_mode="x")
    plt.close()


@pytest.mark.parametrize(
    "node_color",
    [
        ["red", "blue"],
        ["red", "blue", "yellow", "cyan", "green"],
        np.array(["b", "y", "g", "c", "r"]),
    ],
)
def test_plot_connectome_exceptions_wrong_number_node_colors(
    node_color, adjacency, node_coords
):
    """Tests that a wrong number of node colors raises \
       a ValueError in plot_connectome.
    """
    with pytest.raises(
        ValueError, match="Mismatch between the number of nodes"
    ):
        plot_connectome(
            adjacency, node_coords, node_color=node_color, display_mode="x"
        )


def test_plot_connectome_exception_wrong_edge_threshold(
    adjacency, node_coords
):
    """Tests that a TypeError is raised in plot_connectome \
       when edge threshold is neither a number nor a string.
    """
    with pytest.raises(
        TypeError, match="should be either a number or a string"
    ):
        plot_connectome(
            adjacency, node_coords, edge_threshold=object(), display_mode="x"
        )


@pytest.mark.parametrize("threshold", ["0.1", "10", "10.2.3%", "asdf%"])
def test_plot_connectome_exception_wrong_edge_threshold_format(
    threshold, adjacency, node_coords
):
    """Tests that a ValueError is raised when edge_threshold is \
       an incorrectly formatted string.
    """
    with pytest.raises(
        ValueError,
        match=("should be a number followed by the percent sign"),
    ):
        plot_connectome(
            adjacency, node_coords, edge_threshold=threshold, display_mode="x"
        )


def test_plot_connectome_wrong_shapes():
    """Tests that ValueErrors are raised when wrong shapes for node_coords \
       or adjacency_matrix are given.
    """
    kwargs = {"display_mode": "x"}
    node_coords = np.arange(2 * 3).reshape((2, 3))
    adjacency_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(
        ValueError, match=r"supposed to have shape \(n, n\).+\(1L?, 2L?\)"
    ):
        plot_connectome(adjacency_matrix[:1, :], node_coords, **kwargs)

    with pytest.raises(ValueError, match=r"shape \(2L?, 3L?\).+\(2L?,\)"):
        plot_connectome(adjacency_matrix, node_coords[:, 2], **kwargs)

    wrong_adjacency_matrix = np.zeros((3, 3))
    with pytest.raises(
        ValueError, match=r"Shape mismatch.+\(3L?, 3L?\).+\(2L?, 3L?\)"
    ):
        plot_connectome(wrong_adjacency_matrix, node_coords, **kwargs)


@pytest.fixture
def expected_error_node_kwargs(node_kwargs):
    """Return the expected error message depending on node_kwargs."""
    if "s" in node_kwargs:
        return "Please use 'node_size' and not 'node_kwargs'"
    elif "c" in node_kwargs:
        return "Please use 'node_color' and not 'node_kwargs'"


@pytest.mark.parametrize("node_kwargs", [{"s": 50}, {"c": "blue"}])
def test_plot_connectome_exceptions_providing_node_info_with_kwargs(
    node_kwargs, adjacency, node_coords, expected_error_node_kwargs
):
    """Tests that an error is raised when specifying node parameters \
       via node_kwargs in plot_connectome.
    """
    with pytest.raises(ValueError, match=expected_error_node_kwargs):
        plot_connectome(
            adjacency, node_coords, node_kwargs=node_kwargs, display_mode="x"
        )
