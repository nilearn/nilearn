import numpy as np

from nilearn.plotting.js_plotting_utils import decode
from nilearn.plotting import html_connectome

from .test_js_plotting_utils import check_html


def test_prepare_line():
    e = np.asarray([0, 1, 2, 3], dtype=int)
    n = np.asarray([[0, 1], [0, 2], [2, 3], [8, 9]], dtype=int)
    pe, pn = html_connectome._prepare_line(e, n)
    assert (pn == [0, 1, 0, 0, 2, 0, 2, 3, 0, 8, 9, 0]).all()
    assert(pe == [0, 0, 0, 1, 1, 0, 2, 2, 0, 3, 3, 0]).all()


def _make_connectome():
    adj = np.diag([1.5, .3, 2.5], 2)
    adj += adj.T
    adj += np.eye(5)

    coord = np.arange(5)
    coord = np.asarray([coord * 10, -coord, coord[::-1]]).T
    return adj, coord


def test_get_connectome():
    adj, coord = _make_connectome()
    connectome = html_connectome._get_connectome(adj, coord)
    con_x = decode(connectome['_con_x'], '<f4')
    expected_x = np.asarray(
        [0, 0, 0,
         0, 20, 0,
         10, 10, 0,
         10, 30, 0,
         20, 0, 0,
         20, 20, 0,
         20, 40, 0,
         30, 10, 0,
         30, 30, 0,
         40, 20, 0,
         40, 40, 0], dtype='<f4')
    assert (con_x == expected_x).all()
    assert {'_con_x', '_con_y', '_con_z', '_con_w', 'colorscale'
            }.issubset(connectome.keys())
    assert (connectome['cmin'], connectome['cmax']) == (-2.5, 2.5)


def test_view_connectome():
    adj, coord = _make_connectome()
    html = html_connectome.view_connectome(adj, coord)
    check_html(html, False, 'connectome-plot')
    html = html_connectome.view_connectome(adj, coord, '85.3%')
    check_html(html, False, 'connectome-plot')
    html = html_connectome.view_connectome(adj, coord, '85.3%',
                                           linewidth=8.5, marker_size=4.2)
    check_html(html, False, 'connectome-plot')


def test_get_markers():
    coords = np.arange(12).reshape((4, 3))
    colors = ['r', 'g', 'black', 'white']
    markers = html_connectome._get_markers(coords, colors)
    assert markers["marker_color"] == [
        '#ff0000', '#007f00', '#000000', '#ffffff']
    assert markers['markers_only']
    con_x = decode(markers['_con_x'], '<f4')
    assert np.allclose(con_x, coords[:, 0])


def test_view_markers():
    coords = np.arange(12).reshape((4, 3))
    colors = ['r', 'g', 'black', 'white']
    html = html_connectome.view_markers(coords, colors)
    check_html(html, False, 'connectome-plot')
    html = html_connectome.view_markers(coords)
    check_html(html, False, 'connectome-plot')
    html = html_connectome.view_markers(coords, marker_size=15)
    check_html(html, False, 'connectome-plot')
