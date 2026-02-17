import json

import numpy as np
import pytest

from nilearn.plotting import html_connectome
from nilearn.plotting.js_plotting_utils import decode
from nilearn.plotting.tests.test_js_plotting_utils import (
    check_html_surface_plots,
)


def test_prepare_line():
    e = np.asarray([0, 1, 2, 3], dtype=int)
    n = np.asarray([[0, 1], [0, 2], [2, 3], [8, 9]], dtype=int)
    pe, pn = html_connectome._prepare_line(e, n)
    assert (pn == [0, 1, 0, 0, 2, 0, 2, 3, 0, 8, 9, 0]).all()
    assert (pe == [0, 0, 0, 1, 1, 0, 2, 2, 0, 3, 3, 0]).all()


@pytest.mark.parametrize(
    "node_color,expected_marker_colors",
    [
        ("cyan", ["#00ffff", "#00ffff", "#00ffff"]),
        ("auto", ["#440154", "#20908c", "#fde724"]),
        (["cyan", "red", "blue"], ["#00ffff", "#ff0000", "#0000ff"]),
    ],
)
def test_prepare_colors_for_markers(node_color, expected_marker_colors):
    number_of_nodes = 3
    marker_colors = html_connectome._prepare_colors_for_markers(
        node_color,
        number_of_nodes,
    )

    assert marker_colors == expected_marker_colors


def _make_connectome():
    adj = np.diag([1.5, 0.3, 2.5], 2)
    adj += adj.T
    adj += np.eye(5)

    coord = np.arange(5)
    coord = np.asarray([coord * 10, -coord, coord[::-1]]).T
    return adj, coord


def test_get_connectome():
    adj, coord = _make_connectome()
    connectome = html_connectome._get_connectome(adj, coord)
    con_x = decode(connectome["_con_x"], "<f4")
    expected_x = np.asarray(
        [
            0,
            0,
            0,
            0,
            20,
            0,
            10,
            10,
            0,
            10,
            30,
            0,
            20,
            0,
            0,
            20,
            20,
            0,
            20,
            40,
            0,
            30,
            10,
            0,
            30,
            30,
            0,
            40,
            20,
            0,
            40,
            40,
            0,
        ],
        dtype="<f4",
    )
    assert (con_x == expected_x).all()
    assert {"_con_x", "_con_y", "_con_z", "_con_w"}.issubset(connectome.keys())
    assert (connectome["line_cmin"], connectome["line_cmax"]) == (-2.5, 2.5)
    adj[adj == 0] = np.nan
    connectome = html_connectome._get_connectome(adj, coord)
    con_x = decode(connectome["_con_x"], "<f4")
    assert (con_x == expected_x).all()
    assert (connectome["line_cmin"], connectome["line_cmax"]) == (-2.5, 2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"title": "SOME_TITLE"},
        {"linewidth": 8.5, "node_size": 4.2},
    ],
)
def test_view_connectome(tmp_path, kwargs):
    """Check output view_connectome."""
    adj, coord = _make_connectome()

    html = html_connectome.view_connectome(adj, coord, **kwargs)
    check_html_surface_plots(
        tmp_path,
        html,
        False,
        "connectome-plot",
        title=kwargs.get("title", None),
    )

    html = html_connectome.view_connectome(
        adj, coord, "85.3%", linewidth=8.5, node_size=np.arange(len(coord))
    )
    check_html_surface_plots(tmp_path, html, False, "connectome-plot")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"marker_color": ["r", "g", "black", "white"]},
        {"marker_size": 15},
        {
            "marker_size": 5.0,
            "marker_color": ["r", "g", "black", "white"],
            "marker_labels": [
                "red marker",
                "green marker",
                "black marker",
                "white marker",
            ],
        },
    ],
)
def test_view_markers(tmp_path, kwargs):
    """Test output view_markers."""
    coords = np.arange(12).reshape((4, 3))

    html = html_connectome.view_markers(coords, **kwargs)

    check_html_surface_plots(tmp_path, html, False, "connectome-plot")
    if kwargs.get("marker_labels", None) is not None:
        labels_dict = {"marker_labels": kwargs.get("marker_labels")}
        assert json.dumps(labels_dict)[1:-1] in html.html


def test_view_markers_coords(tmp_path):
    """Test output view_markers."""
    coords = np.arange(12).reshape((4, 3))

    html = html_connectome.view_markers(
        coords, marker_size=np.arange(len(coords))
    )
    check_html_surface_plots(tmp_path, html, False, "connectome-plot")

    html = html_connectome.view_markers(
        coords, marker_size=list(range(len(coords)))
    )
    check_html_surface_plots(tmp_path, html, False, "connectome-plot")
