import base64
import re

import numpy as np
import pytest

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting.js_plotting_utils import (
    add_js_lib,
    decode,
    encode,
    get_html_template,
    mesh_to_plotly,
)
from nilearn.surface import load_surf_mesh

try:
    from lxml import etree

    LXML_INSTALLED = True
except ImportError:
    LXML_INSTALLED = False


def _normalize_ws(text):
    return re.sub(r"\s+", " ", text)


def test_add_js_lib():
    """Tests for function add_js_lib.

    Checks that the html page contains the javascript code.
    """
    html = get_html_template("surface_plot_template.html")
    cdn = add_js_lib(html, embed_js=False)
    assert "decodeBase64" in cdn
    assert _normalize_ws(
        """<script
    src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js">
    </script>
    <script src="https://cdn.plot.ly/plotly-gl3d-latest.min.js"></script>
    """
    ) in _normalize_ws(cdn)
    inline = _normalize_ws(add_js_lib(html, embed_js=True))
    assert (
        _normalize_ws(
            """/*! jQuery v3.6.0 | (c) OpenJS Foundation and other
                            contributors | jquery.org/license */"""
        )
        in inline
    )
    assert (
        _normalize_ws(
            """**
                            * plotly.js (gl3d - minified)"""
        )
        in inline
    )
    assert "decodeBase64" in inline


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


def check_html(
    tmp_path, html, check_selects=True, plot_div_id="surface-plot", title=None
):
    """Perform several checks on raw HTML code."""
    tmpfile = tmp_path / "test.html"

    html.save_as_html(tmpfile)
    with tmpfile.open() as f:
        saved = f.read()
    # If present, replace Windows line-end '\r\n' with Unix's '\n'
    saved = saved.replace("\r\n", "\n")
    standalone = html.get_standalone().replace("\r\n", "\n")
    assert saved == standalone

    assert "INSERT" not in html.html
    assert html.get_standalone() == html.html
    assert html._repr_html_() == html.get_iframe()
    assert str(html) == html.get_standalone()
    assert '<meta charset="UTF-8" />' in str(html)
    resized = html.resize(3, 17)
    assert resized is html
    assert (html.width, html.height) == (3, 17)
    assert 'width="3" height="17"' in html.get_iframe()
    assert 'width="33" height="37"' in html.get_iframe(33, 37)
    if title is not None:
        assert f"<title>{title}</title>" in str(html)
    if not LXML_INSTALLED:
        return
    root = etree.HTML(
        html.html.encode("utf-8"), parser=etree.HTMLParser(huge_tree=True)
    )
    head = root.find("head")
    assert len(head.findall("script")) == 5
    body = root.find("body")
    div = body.find("div")
    assert ("id", plot_div_id) in div.items()
    if not check_selects:
        return
    selects = body.findall("select")
    assert len(selects) == 3
    hemi = selects[0]
    assert ("id", "select-hemisphere") in hemi.items()
    assert len(hemi.findall("option")) == 3
    kind = selects[1]
    assert ("id", "select-kind") in kind.items()
    assert len(kind.findall("option")) == 2
    view = selects[2]
    assert ("id", "select-view") in view.items()
    assert len(view.findall("option")) == 7


def test_import_html_document_from_js_plotting():
    """Smoke test importing HTMLDocument from js_plotting_utils."""
    from nilearn.plotting.js_plotting_utils import (  # noqa: F401
        HTMLDocument,
        set_max_img_views_before_warning,
    )
