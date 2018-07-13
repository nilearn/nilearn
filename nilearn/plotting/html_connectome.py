import json

import numpy as np
from scipy import sparse
from nilearn import datasets

from .html_surface import (add_js_lib, HTMLDocument, to_plotly,
                           _encode, colorscale, cm, _get_html_template)


class ConnectomeView(HTMLDocument):
    pass


def _prepare_line(edges, nodes):
    path_edges = np.zeros(len(edges) * 3, dtype=int)
    path_edges[::3] = edges
    path_edges[1::3] = edges
    path_nodes = np.zeros(len(nodes) * 3, dtype=int)
    path_nodes[::3] = nodes[:, 0]
    path_nodes[1::3] = nodes[:, 1]
    return path_edges, path_nodes


def _get_connectome(adjacency_matrix, coords, threshold=None,
                    cmap=cm.cold_hot, symmetric_cmap=True):
    connectome = {}
    coords = np.asarray(coords, dtype='<f4')
    adjacency_matrix = adjacency_matrix.copy()
    colors, vmin, vmax, cmap, norm, abs_threshold = colorscale(
        cmap, adjacency_matrix.ravel(), threshold=threshold,
        symmetric_cmap=symmetric_cmap)
    connectome['colorscale'] = colors
    connectome['cmin'], connectome['cmax'] = float(vmin), float(vmax)
    if threshold is not None:
        adjacency_matrix[np.abs(adjacency_matrix) <= abs_threshold] = 0
    s = sparse.coo_matrix(adjacency_matrix)
    nodes = np.asarray([s.row, s.col], dtype=int).T
    edges = np.arange(len(nodes))
    path_edges, path_nodes = _prepare_line(edges, nodes)
    connectome["_con_w"] = _encode(np.asarray(s.data, dtype='<f4')[path_edges])
    c = coords[path_nodes]
    x, y, z = c.T
    for coord, cname in [(x, "x"), (y, "y"), (z, "z")]:
        connectome["_con_{}".format(cname)] = _encode(
            np.asarray(coord, dtype='<f4'))
    return connectome


def view_connectome(adjacency_matrix, coords, threshold=None,
                    cmap=cm.cyan_orange, symmetric_cmap=True, embed_js=True):
    mesh = datasets.fetch_surf_fsaverage()
    mesh_info = {}
    mesh_info["connectome"] = _get_connectome(
        adjacency_matrix, coords, threshold=threshold, cmap=cmap,
        symmetric_cmap=symmetric_cmap)
    for hemi in ['pial_left', 'pial_right']:
        mesh_info[hemi] = to_plotly(mesh[hemi])
    as_json = json.dumps(mesh_info)
    as_html = _get_html_template('connectome_plot_template.html').replace(
        'INSERT_CONNECTOME_JSON_HERE', as_json)
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return ConnectomeView(as_html)
