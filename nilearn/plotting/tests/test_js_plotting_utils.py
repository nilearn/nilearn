import base64

import numpy as np
import pytest

from nilearn._utils.helpers import is_gil_enabled
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting.js_plotting_utils import (
    decode,
    encode,
    mesh_to_plotly,
)
from nilearn.surface import load_surf_mesh


@pytest.mark.parametrize("dtype", ["<f4", "<i4", ">f4", ">i4"])
def test_encode(dtype):
    """Test base64 encoding/decoding for different dtypes."""
    a = np.arange(10, dtype=dtype)
    encoded = encode(a)
    decoded = base64.b64decode(encoded.encode("utf-8"))
    b = np.frombuffer(decoded, dtype=dtype)
    assert np.allclose(decode(encoded, dtype=dtype), b)
    assert np.allclose(a, b)


@pytest.mark.parametrize("hemi", ["left", "right"])
def test_mesh_to_plotly(hemi):
    """Tests for function mesh_to_plotly."""
    fsaverage = fetch_surf_fsaverage()
    coord, triangles = load_surf_mesh(fsaverage[f"pial_{hemi}"])
    plotly = mesh_to_plotly(fsaverage[f"pial_{hemi}"])
    for i, key in enumerate(["_x", "_y", "_z"]):
        assert np.allclose(decode(plotly[key], "<f4"), coord[:, i])
    for i, key in enumerate(["_i", "_j", "_k"]):
        assert np.allclose(decode(plotly[key], "<i4"), triangles[:, i])


def check_html_surface_plots(
    tmp_path,
    html,
    check_selects=True,
    plot_div_id="surface-plot",
    title=None,
    engine="plotly",
):
    """Perform several checks on raw HTML code.

    Used to check the output of
    - ``view_connectome``
    - ``view_surf``
    -  ``view_markers``

    """
    tmpfile = tmp_path / "test.html"

    assert "* plotly.js (gl3d - minified) v1." in html.html
    assert "jQuery v3.6.0" in html.html
    assert 'charset="UTF-8"' in html.html

    html.save_as_html(tmpfile)
    with tmpfile.open() as f:
        saved = f.read()

    # If present, replace Windows line-end '\r\n' with Unix's '\n'
    saved = saved.replace("\r\n", "\n")
    standalone = html.get_standalone().replace("\r\n", "\n")
    assert saved == standalone

    assert html.get_standalone() == html.html
    assert html._repr_html_() == html.get_iframe()
    assert str(html) == html.get_standalone()

    resized = html.resize(3, 17)
    assert resized is html
    assert (html.width, html.height) == (3, 17)
    assert 'width="3" height="17"' in html.get_iframe()
    assert 'width="33" height="37"' in html.get_iframe(33, 37)

    if title is not None:
        assert f"<title>Nilearn - {title}</title>" in str(html)

    # when testing without the GIL
    # we cannot import lxml as it requires the GIL
    if not is_gil_enabled():
        return

    _check_lxml(html, check_selects, plot_div_id, engine)


def _check_lxml(html, check_selects, plot_div_id, engine):
    from lxml import etree

    root = etree.HTML(
        html.html.encode("utf-8"), parser=etree.HTMLParser(huge_tree=True)
    )
    head = root.find("head")
    if engine == "plotly":
        assert len(head.findall("script")) == 5
    elif engine == "niivue":
        assert len(head.findall("script")) == 0

    main = root.find("body").find("main")
    div = main.find("div")
    assert ("id", plot_div_id) in div.items()

    if not check_selects:
        return

    selects = main.findall("select")
    assert len(selects) == 3

    for idx, selector, expected_n in zip(
        [0, 1, 2], ["hemisphere", "kind", "view"], [3, 2, 7], strict=False
    ):
        assert ("id", f"select-{selector}") in selects[idx].items()
        assert len(selects[idx].findall("option")) == expected_n
