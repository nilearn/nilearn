import time
import webbrowser
from unittest.mock import Mock

import pytest
import requests
from numpy.testing import assert_no_warnings

from nilearn.plotting import html_document

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


class Get:
    def __init__(self, delay=0.0):
        self.delay = delay

    def __call__(self, url):
        time.sleep(self.delay)
        self.url = url
        requests.get(url.replace("index.html", "favicon.ico"))
        self.content = requests.get(url).content


# disable request mocking for this test -- note we are accessing localhost only
@pytest.mark.parametrize("request_mocker", [None])
def test_open_in_browser(monkeypatch):
    opener = Get()
    monkeypatch.setattr(webbrowser, "open", opener)
    doc = html_document.HTMLDocument("hello")
    doc.open_in_browser()
    assert opener.content == b"hello"


def test_open_in_browser_timeout(monkeypatch):
    opener = Get(delay=1.0)
    monkeypatch.setattr(webbrowser, "open", opener)
    monkeypatch.setattr(html_document, "BROWSER_TIMEOUT_SECONDS", 0.01)
    doc = html_document.HTMLDocument("hello")
    with pytest.raises(RuntimeError, match="Failed to open"):
        doc.open_in_browser()


@pytest.mark.parametrize("request_mocker", [None])
def test_open_in_browser_deprecation_warning(monkeypatch):
    monkeypatch.setattr(webbrowser, "open", Get())
    doc = html_document.HTMLDocument("hello")
    with pytest.deprecated_call(match="temp_file_lifetime"):
        doc.open_in_browser(temp_file_lifetime=30.0)


def test_open_in_browser_file(tmp_path, monkeypatch):
    opener = Mock()
    monkeypatch.setattr(webbrowser, "open", opener)
    file_path = tmp_path / "doc.html"
    doc = html_document.HTMLDocument("hello")
    doc.open_in_browser(file_name=str(file_path))
    assert file_path.read_text("utf-8") == "hello"
    opener.assert_called_once_with(f"file://{file_path}")


def _open_views():
    return [html_document.HTMLDocument("") for _ in range(12)]


def _open_one_view():
    for _ in range(12):
        v = html_document.HTMLDocument("")
    return v


def test_open_view_warning():
    # opening many views (without deleting the SurfaceView objects)
    # should raise a warning about memory usage
    pytest.warns(UserWarning, _open_views)
    assert_no_warnings(_open_one_view)
    html_document.set_max_img_views_before_warning(15)
    assert_no_warnings(_open_views)
    html_document.set_max_img_views_before_warning(-1)
    assert_no_warnings(_open_views)
    html_document.set_max_img_views_before_warning(None)
    assert_no_warnings(_open_views)
    html_document.set_max_img_views_before_warning(6)
    pytest.warns(UserWarning, _open_views)


def test_repr():
    doc = html_document.HTMLDocument("hello")
    assert "hello" in doc._repr_html_()
    assert "hello" in doc._repr_mimebundle_()["text/html"]
