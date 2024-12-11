"""Handle plotting of connectomes in html."""

import json

import numpy as np
from matplotlib import cm as mpl_cm
from scipy import sparse

from nilearn.plotting.html_document import HTMLDocument

from .. import datasets
from . import cm
from .js_plotting_utils import (
    add_js_lib,
    colorscale,
    encode,
    get_html_template,
    mesh_to_plotly,
    to_color_strings,
)


class ConnectomeView(HTMLDocument):  # noqa: D101
    pass


def _encode_coordinates(coords, prefix):
    """Transform a 2D-array of 3D data (x, y, z) into a dict of base64 values.

    Parameters
    ----------
    coords : :class:`numpy.ndarray` of shape=(n_nodes, 3)
        The coordinates of the nodes in MNI space.

    prefix : :obj:`str`
        Prefix for the key value in the returned dict.
        Schema is {prefix}{x|y|z}

    Returns
    -------
    coordinates : :obj:`dict`
        Dictionary containing base64 values for each axis
    """
    coordinates = {}

    coords = np.asarray(coords, dtype="<f4")
    marker_x, marker_y, marker_z = coords.T
    for coord, cname in [(marker_x, "x"), (marker_y, "y"), (marker_z, "z")]:
        coordinates[f"{prefix}{cname}"] = encode(
            np.asarray(coord, dtype="<f4")
        )

    return coordinates


def _prepare_line(edges, nodes):
    """Prepare a plotly scatter3d line plot \
    so that a set of disconnected edges \
    can be drawn as a single line.

    `edges` are values associated with each edge (that get mapped to colors
    through a colorscale). `nodes` are pairs of (source, target) node indices
    for each edge.

    the color of a line segment in plotly is a mixture of the colors associated
    with the points it connects. Moreover, segments that begin or end at a
    point whose value is `null` are not drawn.

    given edges = [eab, ecd, eef] and nodes = [(a, b), (c, d), (e, f)], this
    function returns:

        path_edges: eab eab   0 ecd ecd   0 eef eef   0
        path_nodes:   a   b   0   c   d   0   e   f   0

    moreover the javascript code replaces every third element (the '0' in the
    lists above) with `null`, so only the a-b, c-d, and e-f segments will get
    plotted, and their colors are correct because both their start and end
    points are associated with the same value.
    """
    path_edges = np.zeros(len(edges) * 3, dtype=int)
    path_edges[::3] = edges
    path_edges[1::3] = edges
    path_nodes = np.zeros(len(nodes) * 3, dtype=int)
    path_nodes[::3] = nodes[:, 0]
    path_nodes[1::3] = nodes[:, 1]
    return path_edges, path_nodes


def _prepare_colors_for_markers(marker_color, number_of_nodes):
    """Generate "color" and "colorscale" attributes \
    based on `marker_color` mode.

    Parameters
    ----------
    marker_color : color or sequence of colors, default='auto'
        Color(s) of the nodes.

    number_of_nodes : :obj:`int`
        Number of nodes in the view

    Returns
    -------
    markers_colors : :obj:`list`
        List of `number_of_nodes` colors as hexadecimal values
    """
    if isinstance(marker_color, str) and marker_color == "auto":
        colors = mpl_cm.viridis(np.linspace(0, 1, number_of_nodes))
    elif isinstance(marker_color, str):
        colors = [marker_color] * number_of_nodes
    else:
        colors = marker_color

    return to_color_strings(colors)


def _prepare_lines_metadata(
    adjacency_matrix, coords, threshold, cmap, symmetric_cmap
):
    """Generate metadata related to lines for _connectome_view plot.

    Parameters
    ----------
    adjacency_matrix : :class:`np.ndarray`, shape=(n_nodes, n_nodes)
        The weights of the edges.

    coords : :class:`np.ndarray`, shape=(n_nodes, 3)
        The coordinates of the nodes in MNI space.

    threshold : :obj:`str`, number or None, optional
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    cmap : :obj:`str` or matplotlib colormap, default=cm.bwr
        Colormap to use.

    symmetric_cmap : :obj:`bool`, default=True
        Make colormap symmetric (ranging from -vmax to vmax).

    Returns
    -------
    coordinates : :obj:`dict`
        Dictionary containing base64 values for each axis
    """
    adjacency_matrix = np.nan_to_num(adjacency_matrix, copy=True)
    colors = colorscale(
        cmap,
        adjacency_matrix.ravel(),
        threshold=threshold,
        symmetric_cmap=symmetric_cmap,
    )
    lines_metadata = {
        "line_colorscale": colors["colors"],
        "line_cmin": float(colors["vmin"]),
        "line_cmax": float(colors["vmax"]),
    }
    if threshold is not None:
        adjacency_matrix[
            np.abs(adjacency_matrix) <= colors["abs_threshold"]
        ] = 0
    s = sparse.coo_matrix(adjacency_matrix)
    nodes = np.asarray([s.row, s.col], dtype=int).T
    edges = np.arange(len(nodes))
    path_edges, path_nodes = _prepare_line(edges, nodes)
    lines_metadata["_con_w"] = encode(
        np.asarray(s.data, dtype="<f4")[path_edges]
    )

    line_coords = coords[path_nodes]

    lines_metadata = {
        **lines_metadata,
        **_encode_coordinates(line_coords, prefix="_con_"),
    }

    return lines_metadata


def _prepare_markers_metadata(coords, marker_size, marker_color, marker_only):
    markers_coordinates = _encode_coordinates(coords, prefix="_marker_")
    markers_metadata = {"markers_only": marker_only, **markers_coordinates}

    if np.ndim(marker_size) > 0:
        marker_size = np.asarray(marker_size)
    if hasattr(marker_size, "tolist"):
        marker_size = marker_size.tolist()
    markers_metadata["marker_size"] = marker_size
    markers_metadata["marker_color"] = _prepare_colors_for_markers(
        marker_color,
        len(coords),
    )

    return markers_metadata


def _get_connectome(
    adjacency_matrix,
    coords,
    threshold=None,
    marker_size=None,
    marker_color="auto",
    cmap=cm.cold_hot,
    symmetric_cmap=True,
):
    lines_metadata = _prepare_lines_metadata(
        adjacency_matrix,
        coords,
        threshold,
        cmap,
        symmetric_cmap,
    )

    markers_metadata = _prepare_markers_metadata(
        coords,
        marker_size,
        marker_color,
        marker_only=False,
    )

    return {
        **lines_metadata,
        **markers_metadata,
    }


def _make_connectome_html(connectome_info, embed_js=True):
    plot_info = {"connectome": connectome_info}
    mesh = datasets.fetch_surf_fsaverage()
    for hemi in ["pial_left", "pial_right"]:
        plot_info[hemi] = mesh_to_plotly(mesh[hemi])
    as_json = json.dumps(plot_info)
    as_html = get_html_template(
        "connectome_plot_template.html"
    ).safe_substitute(
        {
            "INSERT_CONNECTOME_JSON_HERE": as_json,
            "INSERT_PAGE_TITLE_HERE": (
                connectome_info["title"] or "Connectome plot"
            ),
        }
    )
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return ConnectomeView(as_html)


def view_connectome(
    adjacency_matrix,
    node_coords,
    edge_threshold=None,
    edge_cmap=cm.bwr,
    symmetric_cmap=True,
    linewidth=6.0,
    node_color="auto",
    node_size=3.0,
    colorbar=True,
    colorbar_height=0.5,
    colorbar_fontsize=25,
    title=None,
    title_fontsize=25,
):
    """Insert a 3d plot of a connectome into an HTML page.

    Parameters
    ----------
    adjacency_matrix : :class:`numpy.ndarray` of shape=(n_nodes, n_nodes)
        The weights of the edges.

    node_coords : :class:`numpy.ndarray` of shape=(n_nodes, 3)
        The coordinates of the nodes in :term:`MNI` space.

    node_color : color or sequence of colors, default='auto'
        Color(s) of the nodes.

    edge_threshold : :obj:`str`, number or None, default=None
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    edge_cmap : :obj:`str` or matplotlib colormap, default=cm.bwr
        Colormap to use.

    symmetric_cmap : :obj:`bool`, default=True
        Make colormap symmetric (ranging from -vmax to vmax).

    linewidth : :obj:`float`, default=6.0
        Width of the lines that show connections.

    node_size : :obj:`float`, default=3.0
        Size of the markers showing the seeds in pixels.

    colorbar : :obj:`bool`, default=True
        Add a colorbar.

    colorbar_height : :obj:`float`, default=0.5
        Height of the colorbar, relative to the figure height.

    colorbar_fontsize : :obj:`int`, default=25
        Fontsize of the colorbar tick labels.

    title : :obj:`str` or None, default=None
        Title for the plot.

    title_fontsize : :obj:`int`, default=25
        Fontsize of the title.

    Returns
    -------
    ConnectomeView : plot of the connectome.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_markers:
        interactive plot of colored markers

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    node_coords = np.asarray(node_coords)

    connectome_info = _get_connectome(
        adjacency_matrix,
        node_coords,
        threshold=edge_threshold,
        cmap=edge_cmap,
        symmetric_cmap=symmetric_cmap,
        marker_size=node_size,
        marker_color=node_color,
    )
    connectome_info["line_width"] = linewidth
    connectome_info["colorbar"] = colorbar
    connectome_info["cbar_height"] = colorbar_height
    connectome_info["cbar_fontsize"] = colorbar_fontsize
    connectome_info["title"] = title
    connectome_info["title_fontsize"] = title_fontsize
    return _make_connectome_html(connectome_info)


def view_markers(
    marker_coords,
    marker_color="auto",
    marker_size=5.0,
    marker_labels=None,
    title=None,
    title_fontsize=25,
):
    """Insert a 3d plot of markers in a brain into an HTML page.

    Parameters
    ----------
    marker_coords : :class:`numpy.ndarray` of shape=(n_nodes, 3)
        The coordinates of the nodes in :term:`MNI` space.

    marker_color : :class:`numpy.ndarray` of shape=(n_nodes,) or \
        'auto', default='auto'
        colors of the markers: list of strings, hex rgb or rgba strings, rgb
        triplets, or rgba triplets (see `formats accepted by matplotlib \
        <https://matplotlib.org/stable/users/explain/colors/colors.html>`)

    marker_size : :obj:`float` or array-like, default=5.0
        Size of the markers showing the seeds in pixels.

    marker_labels : :obj:`list` of :obj:`str` of shape=(n_nodes)\
                     or None, default=None
        Labels for the markers: list of strings

    title : :obj:`str` or None, default=None
        Title for the plot.

    title_fontsize : :obj:`int`, default=25
        Fontsize of the title.

    Returns
    -------
    ConnectomeView : plot of the markers.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_connectome:
        interactive plot of a connectome.

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    marker_coords = np.asarray(marker_coords)
    if marker_color is None:
        marker_color = ["red" for _ in range(len(marker_coords))]
    connectome_info = _prepare_markers_metadata(
        marker_coords,
        marker_size,
        marker_color,
        marker_only=True,
    )
    if marker_labels is None:
        marker_labels = ["" for _ in range(marker_coords.shape[0])]
    connectome_info["marker_labels"] = marker_labels
    connectome_info["title"] = title
    connectome_info["title_fontsize"] = title_fontsize
    return _make_connectome_html(connectome_info)
