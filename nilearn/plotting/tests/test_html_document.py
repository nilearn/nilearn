import os
import time
import pytest
import warnings
import tempfile
import webbrowser
from nilearn.plotting import html_document

from numpy.testing import assert_no_warnings

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _open_mock(f):
    print('opened {}'.format(f))


def test_temp_file_removing():
    html = html_document.HTMLDocument('hello')
    wb_open = webbrowser.open
    webbrowser.open = _open_mock
    fd, tmpfile = tempfile.mkstemp()
    try:
        os.close(fd)
        with warnings.catch_warnings(record=True) as record:
            html.open_in_browser(file_name=tmpfile, temp_file_lifetime=None)
        for warning in record:
            assert "Saved HTML in temporary file" not in str(warning.message)
        html.open_in_browser(temp_file_lifetime=0.5)
        assert os.path.isfile(html._temp_file)
        html._temp_file_removing_proc.wait()
        assert not os.path.isfile(html._temp_file)
        with pytest.warns(UserWarning, match="Saved HTML in temporary file"):
            html.open_in_browser(temp_file_lifetime=None)
        html.open_in_browser(temp_file_lifetime=None)
        assert os.path.isfile(html._temp_file)
        time.sleep(1.5)
        assert os.path.isfile(html._temp_file)
    finally:
        webbrowser.open = wb_open
        try:
            os.remove(html._temp_file)
        except Exception:
            pass
        try:
            os.remove(tmpfile)
        except Exception:
            pass


def _open_views():
    return [html_document.HTMLDocument('') for i in range(12)]


def _open_one_view():
    for i in range(12):
        v = html_document.HTMLDocument('')
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
