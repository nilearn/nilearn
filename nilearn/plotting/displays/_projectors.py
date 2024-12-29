import warnings
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mpl_cm
from scipy.sparse import issparse
from scipy.stats import scoreatpercentile

from nilearn._utils.param_validation import check_threshold
from nilearn.plotting import cm
from nilearn.plotting.displays._axes import GlassBrainAxes
from nilearn.plotting.displays._slicers import (
    OrthoSlicer,
    _get_create_display_fun,
)


class OrthoProjector(OrthoSlicer):
    """A class to create linked axes for plotting orthogonal projections \
    of 3D maps.

    This visualization mode can be activated from
    :func:`~nilearn.plotting.plot_glass_brain`, by setting
    ``display_mode='ortho'``:

      .. code-block:: python

          from nilearn.datasets import load_mni152_template
          from nilearn.plotting import plot_glass_brain

          img = load_mni152_template()
          # display is an instance of the OrthoProjector class
          display = plot_glass_brain(img, display_mode="ortho")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The 3 axes used to plot each view ('x', 'y', and 'z').

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    """

    _axes_class = GlassBrainAxes

    @classmethod
    def find_cut_coords(
        cls,
        img=None,  # noqa: ARG003
        threshold=None,  # noqa: ARG003
        cut_coords=None,  # noqa: ARG003
    ):
        """Find the coordinates of the cut."""
        return (None,) * len(cls._cut_displayed)

    def draw_cross(self, cut_coords=None, **kwargs):
        """Do nothing.

        It does not make sense to draw crosses for the position of
        the cuts since we are taking the max along one axis.
        """
        pass

    def _check_inputs_add_graph(
        self,
        adjacency_matrix,
        node_coords,
        node_color,
        node_kwargs,
    ):
        """Perform the input checks and raise different types of errors.

        ``_check_inputs_add_graph`` is called inside the method ``add_graph``.
        """
        # safety checks
        if "s" in node_kwargs:
            raise ValueError(
                "Please use 'node_size' and not 'node_kwargs' "
                "to specify node sizes."
            )
        if "c" in node_kwargs:
            raise ValueError(
                "Please use 'node_color' and not 'node_kwargs' "
                "to specify node colors."
            )

        adjacency_matrix_shape = adjacency_matrix.shape
        if (
            len(adjacency_matrix_shape) != 2
            or adjacency_matrix_shape[0] != adjacency_matrix_shape[1]
        ):
            raise ValueError(
                "'adjacency_matrix' is supposed to have shape (n, n)."
                f" Its shape was {adjacency_matrix_shape}."
            )

        node_coords_shape = node_coords.shape
        if len(node_coords_shape) != 2 or node_coords_shape[1] != 3:
            message = (
                "Invalid shape for 'node_coords'. "
                "You passed an 'adjacency_matrix' "
                f"of shape {adjacency_matrix_shape} "
                "therefore 'node_coords' should be a array "
                f"with shape ({adjacency_matrix_shape[0]}, 3) "
                f"while its shape was {node_coords_shape}."
            )

            raise ValueError(message)

        if (
            isinstance(node_color, (list, np.ndarray))
            and len(node_color) != 1
            and len(node_color) != node_coords_shape[0]
        ):
            raise ValueError(
                "Mismatch between the number of nodes "
                f"({node_coords_shape[0]}) "
                f"and the number of node colors ({len(node_color)})."
            )

        if node_coords_shape[0] != adjacency_matrix_shape[0]:
            raise ValueError(
                "Shape mismatch between 'adjacency_matrix' "
                "and 'node_coords'."
                f"'adjacency_matrix' shape is {adjacency_matrix_shape}, "
                f"'node_coords' shape is {node_coords_shape}."
            )

    def add_graph(
        self,
        adjacency_matrix,
        node_coords,
        node_color="auto",
        node_size=50,
        edge_cmap=cm.bwr,
        edge_vmin=None,
        edge_vmax=None,
        edge_threshold=None,
        edge_kwargs=None,
        node_kwargs=None,
        colorbar=False,
    ):
        """Plot undirected graph on each of the axes.

        Parameters
        ----------
        adjacency_matrix : :class:`numpy.ndarray` of shape ``(n, n)``
            Represents the edges strengths of the graph.
            The matrix can be symmetric which will result in
            an undirected graph, or not symmetric which will
            result in a directed graph.

        node_coords : :class:`numpy.ndarray` of shape ``(n, 3)``
            3D coordinates of the graph nodes in world space.

        node_color : color or sequence of colors, default='auto'
            Color(s) of the nodes.

        node_size : scalar or array_like, default=50
            Size(s) of the nodes in points^2.

        edge_cmap : :class:`~matplotlib.colors.Colormap`, default=cm.bwr
            Colormap used for representing the strength of the edges.


        edge_vmin, edge_vmax : :obj:`float`, optional
            - If not ``None``, either or both of these values will be used
              to as the minimum and maximum values to color edges.
            - If ``None`` are supplied, the maximum absolute value within the
              given threshold will be used as minimum (multiplied by -1) and
              maximum coloring levels.

        edge_threshold : :obj:`str` or :obj:`int` or :obj:`float`, optional
            - If it is a number only the edges with a value greater than
              ``edge_threshold`` will be shown.
            - If it is a string it must finish with a percent sign,
              e.g. "25.3%", and only the edges with a abs(value) above
              the given percentile will be shown.

        edge_kwargs : :obj:`dict`, optional
            Will be passed as kwargs for each edge
            :class:`~matplotlib.lines.Line2D`.

        node_kwargs : :obj:`dict`
            Will be passed as kwargs to the function
            :func:`~matplotlib.pyplot.scatter` which plots all the
            nodes at one.
        """
        # set defaults
        edge_kwargs = edge_kwargs or {}
        node_kwargs = node_kwargs or {}
        if isinstance(node_color, str) and node_color == "auto":
            node_color = mpl_cm.Set2(np.linspace(0, 1, len(node_coords)))
        node_coords = np.asarray(node_coords)

        # decompress input matrix if sparse
        if issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.toarray()

        # make the lines below well-behaved
        adjacency_matrix = np.nan_to_num(adjacency_matrix)

        self._check_inputs_add_graph(
            adjacency_matrix, node_coords, node_color, node_kwargs
        )

        # If the adjacency matrix is not symmetric, give a warning
        symmetric = True
        if not np.allclose(adjacency_matrix, adjacency_matrix.T, rtol=1e-3):
            symmetric = False
            warnings.warn(
                "'adjacency_matrix' is not symmetric.\n"
                "A directed graph will be plotted.",
                stacklevel=3,
            )

        # For a masked array, masked values are replaced with zeros
        if hasattr(adjacency_matrix, "mask"):
            if not (adjacency_matrix.mask == adjacency_matrix.mask.T).all():
                symmetric = False
                warnings.warn(
                    "'adjacency_matrix' was masked \
                    with a non symmetric mask.\n"
                    "A directed graph will be plotted.",
                    stacklevel=3,
                )
            adjacency_matrix = adjacency_matrix.filled(0)

        if edge_threshold is not None:
            if symmetric:
                # Keep a percentile of edges with the highest absolute
                # values, so only need to look at the covariance
                # coefficients below the diagonal
                lower_diagonal_indices = np.tril_indices_from(
                    adjacency_matrix, k=-1
                )
                lower_diagonal_values = adjacency_matrix[
                    lower_diagonal_indices
                ]
                edge_threshold = check_threshold(
                    edge_threshold,
                    np.abs(lower_diagonal_values),
                    scoreatpercentile,
                    "edge_threshold",
                )
            else:
                edge_threshold = check_threshold(
                    edge_threshold,
                    np.abs(adjacency_matrix.ravel()),
                    scoreatpercentile,
                    "edge_threshold",
                )

            adjacency_matrix = adjacency_matrix.copy()
            threshold_mask = np.abs(adjacency_matrix) < edge_threshold
            adjacency_matrix[threshold_mask] = 0

        if symmetric:
            lower_triangular_adjacency_matrix = np.tril(adjacency_matrix, k=-1)
            non_zero_indices = lower_triangular_adjacency_matrix.nonzero()
        else:
            non_zero_indices = adjacency_matrix.nonzero()

        line_coords = [
            node_coords[list(index)] for index in zip(*non_zero_indices)
        ]

        adjacency_matrix_values = adjacency_matrix[non_zero_indices]
        for ax in self.axes.values():
            ax._add_markers(node_coords, node_color, node_size, **node_kwargs)
            if line_coords:
                ax._add_lines(
                    line_coords,
                    adjacency_matrix_values,
                    edge_cmap,
                    vmin=edge_vmin,
                    vmax=edge_vmax,
                    directed=(not symmetric),
                    **edge_kwargs,
                )
            # To obtain the brain left view, we simply invert the x axis
            if ax.direction == "l" and not (
                ax.ax.get_xlim()[0] > ax.ax.get_xlim()[1]
            ):
                ax.ax.invert_xaxis()

        if colorbar:
            self._colorbar = colorbar
            self._show_colorbar(ax.cmap, ax.norm, threshold=edge_threshold)

        plt.draw_if_interactive()


class XProjector(OrthoProjector):
    """The ``XProjector`` class enables sagittal visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='x'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the XProjector class
        display = plot_glass_brain(img, display_mode="x")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
           The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
                 The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YProjector : Coronal view
    nilearn.plotting.displays.ZProjector : Axial view

    """

    _cut_displayed: ClassVar[str] = "x"
    _default_figsize: ClassVar[list[float, float]] = [2.6, 3.0]


class YProjector(OrthoProjector):
    """The ``YProjector`` class enables coronal visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='y'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YProjector class
        display = plot_glass_brain(img, display_mode="y")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XProjector : Sagittal view
    nilearn.plotting.displays.ZProjector : Axial view

    """

    _cut_displayed: ClassVar[str] = "y"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.0]


class ZProjector(OrthoProjector):
    """The ``ZProjector`` class enables axial visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='z'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the ZProjector class
        display = plot_glass_brain(img, display_mode="z")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XProjector : Sagittal view
    nilearn.plotting.displays.YProjector : Coronal view

    """

    _cut_displayed: ClassVar[str] = "z"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.4]


class XZProjector(OrthoProjector):
    """The ``XZProjector`` class enables to combine sagittal \
    and axial views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='xz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the XZProjector class
        display = plot_glass_brain(img, display_mode="xz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('x' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YXProjector : Coronal + Sagittal views
    nilearn.plotting.displays.YZProjector : Coronal + Axial views

    """

    _cut_displayed = "xz"


class YXProjector(OrthoProjector):
    """The ``YXProjector`` class enables to combine coronal \
    and sagittal views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='yx'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YXProjector class
        display = plot_glass_brain(img, display_mode="yx")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('x' and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZProjector : Sagittal + Axial views
    nilearn.plotting.displays.YZProjector : Coronal + Axial views

    """

    _cut_displayed = "yx"


class YZProjector(OrthoProjector):
    """The ``YZProjector`` class enables to combine coronal and axial views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='yz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YZProjector class
        display = plot_glass_brain(img, display_mode="yz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('y' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZProjector : Sagittal + Axial views
    nilearn.plotting.displays.YXProjector : Coronal + Sagittal views

    """

    _cut_displayed: ClassVar[str] = "yz"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.4]


class LYRZProjector(OrthoProjector):
    """The ``LYRZProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lyrz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LYRZProjector class
        display = plot_glass_brain(img, display_mode="lyrz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'y', 'r',
        and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LZRYProjector : ?? views

    """

    _cut_displayed = "lyrz"


class LZRYProjector(OrthoProjector):
    """The ``LZRYProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lzry'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LZRYProjector class
        display = plot_glass_brain(img, display_mode="lzry")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'z', 'r',
        and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LYRZProjector : ?? views

    """

    _cut_displayed = "lzry"


class LZRProjector(OrthoProjector):
    """The ``LZRProjector`` class enables hemispheric sagittal visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lzr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LZRProjector class
        display = plot_glass_brain(img, display_mode="lzr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'z' and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LYRProjector : ?? views

    """

    _cut_displayed = "lzr"


class LYRProjector(OrthoProjector):
    """The ``LYRProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lyr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LYRProjector class
        display = plot_glass_brain(img, display_mode="lyr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'y' and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LZRProjector : ?? views

    """

    _cut_displayed = "lyr"


class LRProjector(OrthoProjector):
    """The ``LRProjector`` class enables left-right visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LRProjector class
        display = plot_glass_brain(img, display_mode="lr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    """

    _cut_displayed = "lr"


class LProjector(OrthoProjector):
    """The ``LProjector`` class enables the visualization of left 2D \
    projection with :func:`~nilearn.plotting.plot_glass_brain`.

    This
    visualization mode can be activated by setting ``display_mode='l'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LProjector class
        display = plot_glass_brain(img, display_mode="l")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.RProjector : right projection view

    """

    _cut_displayed: ClassVar[str] = "l"
    _default_figsize: ClassVar[list[float, float]] = [2.6, 3.0]


class RProjector(OrthoProjector):
    """The ``RProjector`` class enables the visualization of right 2D \
    projection with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='r'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the RProjector class
        display = plot_glass_brain(img, display_mode="r")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LProjector : left projection view

    """

    _cut_displayed: ClassVar[str] = "r"
    _default_figsize: ClassVar[list[float, float]] = [2.6, 2.8]


PROJECTORS = {
    "ortho": OrthoProjector,
    "xz": XZProjector,
    "yz": YZProjector,
    "yx": YXProjector,
    "x": XProjector,
    "y": YProjector,
    "z": ZProjector,
    "lzry": LZRYProjector,
    "lyrz": LYRZProjector,
    "lyr": LYRProjector,
    "lzr": LZRProjector,
    "lr": LRProjector,
    "l": LProjector,
    "r": RProjector,
}


def get_projector(display_mode):
    """Retrieve a projector from a given display mode.

    Parameters
    ----------
    display_mode : {"ortho", "xz", "yz", "yx", "x", "y",\
    "z", "lzry", "lyrz", "lyr", "lzr", "lr", "l", "r"}
        The desired display mode.

    Returns
    -------
    projector : :class:`~nilearn.plotting.displays.OrthoProjector`\
    or instance of derived classes

        The projector corresponding to the requested display mode:

            - "ortho": Returns an
              :class:`~nilearn.plotting.displays.OrthoProjector`.
            - "xz": Returns a
              :class:`~nilearn.plotting.displays.XZProjector`.
            - "yz": Returns a
              :class:`~nilearn.plotting.displays.YZProjector`.
            - "yx": Returns a
              :class:`~nilearn.plotting.displays.YXProjector`.
            - "x": Returns a
              :class:`~nilearn.plotting.displays.XProjector`.
            - "y": Returns a
              :class:`~nilearn.plotting.displays.YProjector`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.ZProjector`.
            - "lzry": Returns a
              :class:`~nilearn.plotting.displays.LZRYProjector`.
            - "lyrz": Returns a
              :class:`~nilearn.plotting.displays.LYRZProjector`.
            - "lyr": Returns a
              :class:`~nilearn.plotting.displays.LYRProjector`.
            - "lzr": Returns a
              :class:`~nilearn.plotting.displays.LZRProjector`.
            - "lr": Returns a
              :class:`~nilearn.plotting.displays.LRProjector`.
            - "l": Returns a
              :class:`~nilearn.plotting.displays.LProjector`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.RProjector`.

    """
    return _get_create_display_fun(display_mode, PROJECTORS)
