"""
Tests for :func:`nilearn.plotting.plot_connectome` and
deprecated :func:`nilearn.plotting.plot_connectome_strength`.
"""

import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib.patches import FancyArrow
from nilearn.plotting import plot_connectome, plot_connectome_strength


PLOTTING_FUNCS = set([plot_connectome, plot_connectome_strength])


@pytest.fixture
def adjacency():
    # symmetric up to 1e-3 relative tolerance
    return np.array([[1., -2., 0.3, 0.],
                     [-2.002, 1, 0., 0.],
                     [0.3, 0., 1., 0.],
                     [0., 0., 0., 1.]])


@pytest.fixture
def non_symmetric_matrix():
    return np.array([[1., -2., 0.3, 0.2],
                     [0.1, 1, 1.1, 0.1],
                     [0.01, 2.3, 1., 3.1],
                     [0.6, 0.03, 1.2, 1.]])


@pytest.fixture
def base_params():
    return dict(edge_threshold=0.38,
                title='threshold=0.38',
                node_size=10)


@pytest.fixture
def node_coords():
    return np.arange(3 * 4).reshape(4, 3)


@pytest.mark.parametrize("node_color",
                         [['green', 'blue', 'k', 'cyan'],
                          np.array(['red']),
                          ['red'],
                          'green'])
@pytest.mark.parametrize("display_mode", ["ortho", "lzry"])
def test_plot_connectome_node_colors(node_color, display_mode, node_coords,
                                     adjacency, base_params, tmpdir):
    plot_connectome(adjacency, node_coords, node_color=node_color,
                    display_mode=display_mode, **base_params)
    plt.close()


@pytest.mark.parametrize("display_mode",
                         ['ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
                          'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'])
def test_plot_connectome_display_mode(display_mode, node_coords,
                                      adjacency, base_params):
    plot_connectome(adjacency, node_coords, display_mode=display_mode,
                    **base_params)
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS)
def test_plot_connectome_masked_array_sparse_matrix(plot_func, node_coords,
                                                    adjacency, base_params):
    if plot_func == plot_connectome_strength:
        base_params.pop("edge_threshold")
    masked_adjacency_matrix = np.ma.masked_array(
        adjacency, np.abs(adjacency) < 0.5)
    plot_func(masked_adjacency_matrix, node_coords, **base_params)
    sparse_adjacency_matrix = sparse.coo_matrix(adjacency)
    plot_func(sparse_adjacency_matrix, node_coords, **base_params)
    plt.close()


@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS)
def test_plot_connectome_with_nans(plot_func, adjacency,
                                   node_coords, base_params):
    adjacency[0, 1] = np.nan
    adjacency[1, 0] = np.nan
    if plot_func == plot_connectome_strength:
        base_params.pop("edge_threshold")
    if plot_func == plot_connectome:
        base_params["node_color"] = np.array(['green', 'blue', 'k', 'yellow'])
    plot_func(adjacency, node_coords, **base_params)
    plt.close()


def test_plot_connectome(adjacency, node_coords, base_params, tmpdir):
    args = adjacency, node_coords
    # used to speed-up tests for the next plots
    base_params['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome(adjacency, [tuple(each) for each in node_coords],
                    **base_params)
    # saving to file
    filename = str(tmpdir.join('temp.png'))
    display = plot_connectome(*args, output_file=filename, **base_params)
    assert display is None
    assert os.path.isfile(filename)
    assert os.path.getsize(filename) > 0
    # with node_kwargs, edge_kwargs and edge_cmap arguments
    plot_connectome(*args,
                    edge_threshold='70%',
                    node_size=[10, 20, 30, 40],
                    node_color=np.zeros((4, 3)),
                    edge_cmap='RdBu',
                    colorbar=True,
                    node_kwargs={'marker': 'v'},
                    edge_kwargs={'linewidth': 4})
    # smoke-test where there is no edge to draw, e.g. when
    # edge_threshold is too high
    plot_connectome(*args, edge_threshold=1e12)
    # with colorbar=True
    plot_connectome(*args, colorbar=True)
    plt.close()


def test_plot_connectome_non_symmetric(node_coords, non_symmetric_matrix):
    ax = plot_connectome(non_symmetric_matrix, node_coords,
                         display_mode='ortho')
    # No thresholding was performed, we should get
    # as many arrows as we have edges
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.prod(non_symmetric_matrix.shape))

    # Set a few elements of adjacency matrix to zero
    non_symmetric_matrix[1, 0] = 0.0
    non_symmetric_matrix[2, 3] = 0.0
    # Plot with different display mode
    ax = plot_connectome(non_symmetric_matrix,
                         node_coords,
                         display_mode='lzry')
    # No edge in direction 'l' because of node coords
    assert(len([patch for patch in ax.axes['l'].ax.patches
             if isinstance(patch, FancyArrow)]) == 0)
    for direction in ['z', 'r', 'y']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.prod(non_symmetric_matrix.shape) - 2)


def plot_connectome_edge_thresholding(node_coords, non_symmetric_matrix):
    # Case 1: Threshold is a number
    thresh = 1.1
    ax = plot_connectome(non_symmetric_matrix,
                         node_coords,
                         edge_threshold=thresh)
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
                    np.sum(np.abs(non_symmetric_matrix) >= thresh)
               )
    # Case 2: Threshold is a percentage
    thresh = 80
    ax = plot_connectome(non_symmetric_matrix,
                         node_coords,
                         edge_threshold="{}%".format(thresh))
    for direction in ['x', 'y', 'z']:
        assert(len([patch for patch in ax.axes[direction].ax.patches
             if isinstance(patch, FancyArrow)]) ==
               np.sum(np.abs(non_symmetric_matrix) >=
                    np.percentile(np.abs(non_symmetric_matrix.ravel()), thresh))
               )


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


def test_connectome_strength(node_coords, adjacency, tmpdir):
    args = adjacency, node_coords
    kwargs = dict()
    plot_connectome_strength(*args, **kwargs)
    plt.close()

    # used to speed-up tests for the net plots
    kwargs['display_mode'] = 'x'

    # node_coords not an array but a list of tuples
    plot_connectome_strength(adjacency,
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


def test_plot_connectome_strength_deprecation_warning(node_coords, adjacency):
    with pytest.deprecated_call():
        plot_connectome_strength(adjacency, node_coords)