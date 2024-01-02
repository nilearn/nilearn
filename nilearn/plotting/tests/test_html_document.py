import time
import webbrowser

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


def _open_views():
    return [html_document.HTMLDocument("") for i in range(12)]


def _open_one_view():
    for i in range(12):
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
