import os
import json

import numpy as np
from scipy import sparse
from nilearn import datasets

from .html_surface import (add_js_lib, HTMLDocument, to_plotly,
                           _encode, colorscale, cm)


def _get_html_template():
    template_path = os.path.join(
        os.path.dirname(__file__),
        'data', 'html', 'connectome_plot_template.html')
    with open(template_path, 'rb') as f:
        return f.read().decode('utf-8')


def _get_test_connectome():
    connectome = {}
    a = np.asarray([[-50, 0, 0]])
    b = np.asarray([[50, 0, 0]])
    for points, name in [(a, "a"), (b, "b")]:
        x, y, z = map(_encode, np.asarray(points.T, dtype='<f4'))
        connectome["_{}_x".format(name)] = x
        connectome["_{}_y".format(name)] = y
        connectome["_{}_z".format(name)] = z
    return connectome


def _get_connectome(adjacency_matrix, coords, threshold=None,
                    cmap=cm.cold_hot):
    connectome = {}
    coords = np.asarray(coords, dtype='<f4')
    adjacency_matrix = adjacency_matrix.copy()
    colors, vmin, vmax, cmap, norm, abs_threshold = colorscale(
        cmap, adjacency_matrix.ravel(), threshold=threshold)
    colors = json.loads(colors)
    connectome['colorscale'] = colors
    connectome['cmin'], connectome['cmax'] = float(vmin), float(vmax)
    if threshold is not None:
        adjacency_matrix[np.abs(adjacency_matrix) <= abs_threshold] = 0
    s = sparse.coo_matrix(adjacency_matrix)
    idx = np.asarray([s.row, s.col], dtype=int).T.ravel()
    d = s.data
    padded = np.zeros(len(d) * 3, dtype='<f4')
    padded[::3] = d
    padded[1::3] = d
    connectome["_a_c"] = _encode(padded)
    c = coords[idx]
    x, y, z = c.T
    for coord, cname in [(x, "x"), (y, "y"), (z, "z")]:
        padded_coord = np.zeros(len(coord) * 3 // 2, dtype='<f4')
        padded_coord[::3] = coord[::2]
        padded_coord[1::3] = coord[1::2]
        coord = _encode(padded_coord)
        connectome["_a_{}".format(cname)] = coord
    return connectome


def view_connectome(adjacency_matrix, coords, threshold=None, embed_js=True,
                    cmap=cm.cold_hot):
    mesh = datasets.fetch_surf_fsaverage5()
    mesh_info = {}
    mesh_info["connectome"] = _get_connectome(
        adjacency_matrix, coords, threshold=threshold, cmap=cmap)
    for hemi in ['pial_left', 'pial_right']:
        mesh_info[hemi] = to_plotly(mesh[hemi])
    as_json = json.dumps(mesh_info)
    as_html = _get_html_template().replace(
        'INSERT_STAT_MAP_JSON_HERE', as_json)
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return HTMLDocument(as_html)
