
import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.stats import scoreatpercentile

import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm

from nilearn.plotting import cm
from nilearn._utils.param_validation import check_threshold
from nilearn.plotting.displays.slicers import(
    OrthoSlicer, _get_create_display_fun
)
from nilearn.plotting.displays.axes import GlassBrainAxes


class OrthoProjector(OrthoSlicer):
    """A class to create linked axes for plotting orthogonal projections
    of 3D maps.

    """
    _axes_class = GlassBrainAxes

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        return (None, ) * len(cls._cut_displayed)

    def draw_cross(self, cut_coords=None, **kwargs):
        # It does not make sense to draw crosses for the position of
        # the cuts since we are taking the max along one axis
        pass

    def add_graph(self, adjacency_matrix, node_coords,
                  node_color='auto', node_size=50,
                  edge_cmap=cm.bwr,
                  edge_vmin=None, edge_vmax=None,
                  edge_threshold=None,
                  edge_kwargs=None, node_kwargs=None, colorbar=False,
                  ):
        """Plot undirected graph on each of the axes

        Parameters
        ----------
        adjacency_matrix : numpy array of shape (n, n)
            Represents the edges strengths of the graph.
            The matrix can be symmetric which will result in
            an undirected graph, or not symmetric which will
            result in a directed graph.

        node_coords : numpy array_like of shape (n, 3)
            3d coordinates of the graph nodes in world space.

        node_color : color or sequence of colors, optional
            Color(s) of the nodes. Default='auto'.

        node_size : scalar or array_like, optional
            Size(s) of the nodes in points^2. Default=50.

        edge_cmap : colormap, optional
            Colormap used for representing the strength of the edges.
            Default=cm.bwr.

        edge_vmin, edge_vmax : float, optional
            If not None, either or both of these values will be used to
            as the minimum and maximum values to color edges. If None are
            supplied the maximum absolute value within the given threshold
            will be used as minimum (multiplied by -1) and maximum
            coloring levels.

        edge_threshold : str or number, optional
            If it is a number only the edges with a value greater than
            edge_threshold will be shown.
            If it is a string it must finish with a percent sign,
            e.g. "25.3%", and only the edges with a abs(value) above
            the given percentile will be shown.

        edge_kwargs : dict, optional
            Will be passed as kwargs for each edge matlotlib Line2D.

        node_kwargs : dict
            Will be passed as kwargs to the plt.scatter call that plots all
            the nodes in one go.

        """
        # set defaults
        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        if isinstance(node_color, str) and node_color == 'auto':
            nb_nodes = len(node_coords)
            node_color = mpl_cm.Set2(np.linspace(0, 1, nb_nodes))
        node_coords = np.asarray(node_coords)

        # decompress input matrix if sparse
        if issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.toarray()

        # make the lines below well-behaved
        adjacency_matrix = np.nan_to_num(adjacency_matrix)

        # safety checks
        if 's' in node_kwargs:
            raise ValueError("Please use 'node_size' and not 'node_kwargs' "
                             "to specify node sizes")
        if 'c' in node_kwargs:
            raise ValueError("Please use 'node_color' and not 'node_kwargs' "
                             "to specify node colors")

        adjacency_matrix_shape = adjacency_matrix.shape
        if (len(adjacency_matrix_shape) != 2 or
                adjacency_matrix_shape[0] != adjacency_matrix_shape[1]):
            raise ValueError(
                "'adjacency_matrix' is supposed to have shape (n, n)."
                ' Its shape was {0}'.format(adjacency_matrix_shape))

        node_coords_shape = node_coords.shape
        if len(node_coords_shape) != 2 or node_coords_shape[1] != 3:
            message = (
                "Invalid shape for 'node_coords'. You passed an "
                "'adjacency_matrix' of shape {0} therefore "
                "'node_coords' should be a array with shape ({0[0]}, 3) "
                'while its shape was {1}').format(adjacency_matrix_shape,
                                                  node_coords_shape)

            raise ValueError(message)

        if isinstance(node_color, (list, np.ndarray)) and len(node_color) != 1:
            if len(node_color) != node_coords_shape[0]:
                raise ValueError(
                    "Mismatch between the number of nodes ({0}) "
                    "and the number of node colors ({1})."
                    .format(node_coords_shape[0], len(node_color)))

        if node_coords_shape[0] != adjacency_matrix_shape[0]:
            raise ValueError(
                "Shape mismatch between 'adjacency_matrix' "
                "and 'node_coords'"
                "'adjacency_matrix' shape is {0}, 'node_coords' shape is {1}"
                .format(adjacency_matrix_shape, node_coords_shape))

        # If the adjacency matrix is not symmetric, give a warning
        symmetric = True
        if not np.allclose(adjacency_matrix, adjacency_matrix.T, rtol=1e-3):
            symmetric = False
            warnings.warn(("'adjacency_matrix' is not symmetric. "
                           "A directed graph will be plotted."))

        # For a masked array, masked values are replaced with zeros
        if hasattr(adjacency_matrix, 'mask'):
            if not (adjacency_matrix.mask == adjacency_matrix.mask.T).all():
                symmetric = False
                warnings.warn(("'adjacency_matrix' was masked with "
                               "a non symmetric mask. A directed "
                               "graph will be plotted."))
            adjacency_matrix = adjacency_matrix.filled(0)

        if edge_threshold is not None:
            if symmetric:
                # Keep a percentile of edges with the highest absolute
                # values, so only need to look at the covariance
                # coefficients below the diagonal
                lower_diagonal_indices = np.tril_indices_from(adjacency_matrix,
                                                              k=-1)
                lower_diagonal_values = adjacency_matrix[
                    lower_diagonal_indices]
                edge_threshold = check_threshold(
                    edge_threshold, np.abs(lower_diagonal_values),
                    scoreatpercentile, 'edge_threshold')
            else:
                edge_threshold = check_threshold(
                    edge_threshold, np.abs(adjacency_matrix.ravel()),
                    scoreatpercentile, 'edge_threshold')

            adjacency_matrix = adjacency_matrix.copy()
            threshold_mask = np.abs(adjacency_matrix) < edge_threshold
            adjacency_matrix[threshold_mask] = 0

        if symmetric:
            lower_triangular_adjacency_matrix = np.tril(adjacency_matrix, k=-1)
            non_zero_indices = lower_triangular_adjacency_matrix.nonzero()
        else:
            non_zero_indices = adjacency_matrix.nonzero()

        line_coords = [node_coords[list(index)]
                       for index in zip(*non_zero_indices)]

        adjacency_matrix_values = adjacency_matrix[non_zero_indices]
        for ax in self.axes.values():
            ax._add_markers(node_coords, node_color, node_size, **node_kwargs)
            if line_coords:
                ax._add_lines(line_coords, adjacency_matrix_values, edge_cmap,
                              vmin=edge_vmin, vmax=edge_vmax, directed=(not symmetric),
                              **edge_kwargs)
            # To obtain the brain left view, we simply invert the x axis
            if ax.direction == 'l' and not (ax.ax.get_xlim()[0] > ax.ax.get_xlim()[1]):
                ax.ax.invert_xaxis()

        if colorbar:
            self._colorbar = colorbar
            self._show_colorbar(ax.cmap, ax.norm, threshold=edge_threshold)

        plt.draw_if_interactive()


class XProjector(OrthoProjector):
    """XProjector class."""
    _cut_displayed = 'x'
    _default_figsize = [2.6, 2.3]


class YProjector(OrthoProjector):
    """YProjector class."""
    _cut_displayed = 'y'
    _default_figsize = [2.2, 2.3]


class ZProjector(OrthoProjector):
    """ZProjector class."""
    _cut_displayed = 'z'
    _default_figsize = [2.2, 2.3]


class XZProjector(OrthoProjector):
    """XZProjector class."""
    _cut_displayed = 'xz'


class YXProjector(OrthoProjector):
    """YXProjector class."""
    _cut_displayed = 'yx'


class YZProjector(OrthoProjector):
    """YZProjector class."""
    _cut_displayed = 'yz'


class LYRZProjector(OrthoProjector):
    """LYRZProjector class."""
    _cut_displayed = 'lyrz'


class LZRYProjector(OrthoProjector):
    """LZRYProjector class."""
    _cut_displayed = 'lzry'


class LZRProjector(OrthoProjector):
    """LZRProjector class."""
    _cut_displayed = 'lzr'


class LYRProjector(OrthoProjector):
    """LYRProjector class."""
    _cut_displayed = 'lyr'


class LRProjector(OrthoProjector):
    """LRProjector class."""
    _cut_displayed = 'lr'


class LProjector(OrthoProjector):
    """Lprojector class."""
    _cut_displayed = 'l'
    _default_figsize = [2.6, 2.3]


class RProjector(OrthoProjector):
    """RProjector class."""
    _cut_displayed = 'r'
    _default_figsize = [2.6, 2.3]


PROJECTORS = dict(ortho=OrthoProjector,
                  xz=XZProjector,
                  yz=YZProjector,
                  yx=YXProjector,
                  x=XProjector,
                  y=YProjector,
                  z=ZProjector,
                  lzry=LZRYProjector,
                  lyrz=LYRZProjector,
                  lyr=LYRProjector,
                  lzr=LZRProjector,
                  lr=LRProjector,
                  l=LProjector,
                  r=RProjector)


def get_projector(display_mode):
    "Internal function to retrieve a projector"
    return _get_create_display_fun(display_mode, PROJECTORS)