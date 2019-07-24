import os
import time
import pytest
import tempfile
import webbrowser
from nilearn import reporting

import numpy as np
from nibabel import Nifti1Image
from nilearn import input_data

try:
    from lxml import etree
    LXML_INSTALLED = True
except ImportError:
    LXML_INSTALLED = False

from numpy.testing import assert_warns, assert_no_warnings

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def check_html(html, check_selects=True, plot_div_id='surface-plot'):
    fd, tmpfile = tempfile.mkstemp()
    try:
        os.close(fd)
        html.save_as_html(tmpfile)
        with open(tmpfile) as f:
            saved = f.read()
        assert saved == html.get_standalone()
    finally:
        os.remove(tmpfile)
    assert "INSERT" not in html.html
    assert html.get_standalone() == html.html
    assert html._repr_html_() == html.get_iframe()
    assert str(html) == html.get_standalone()
    assert '<meta charset="UTF-8" />' in str(html)
    _check_open_in_browser(html)
    resized = html.resize(3, 17)
    assert resized is html
    assert (html.width, html.height) == (3, 17)
    assert 'width="3" height="17"' in html.get_iframe()
    assert 'width="33" height="37"' in html.get_iframe(33, 37)
    if not LXML_INSTALLED:
        return
    root = etree.HTML(html.html.encode('utf-8'),
                      parser=etree.HTMLParser(huge_tree=True))
    head = root.find('head')
    assert len(head.findall('script')) == 5
    body = root.find('body')
    div = body.find('div')
    assert ('id', plot_div_id) in div.items()
    if not check_selects:
        return
    selects = body.findall('select')
    assert len(selects) == 3
    hemi = selects[0]
    assert ('id', 'select-hemisphere') in hemi.items()
    assert len(hemi.findall('option')) == 2
    kind = selects[1]
    assert ('id', 'select-kind') in kind.items()
    assert len(kind.findall('option')) == 2
    view = selects[2]
    assert ('id', 'select-view') in view.items()
    assert len(view.findall('option')) == 7


def _open_mock(f):
    print('opened {}'.format(f))


def _check_open_in_browser(html):
    wb_open = webbrowser.open
    webbrowser.open = _open_mock
    try:
        html.open_in_browser(temp_file_lifetime=None)
        temp_file = html._temp_file
        assert html._temp_file is not None
        assert os.path.isfile(temp_file)
        html.remove_temp_file()
        assert html._temp_file is None
        assert not os.path.isfile(temp_file)
        html.remove_temp_file()
        html._temp_file = 'aaaaaaaaaaaaaaaaaaaaaa'
        html.remove_temp_file()
    finally:
        webbrowser.open = wb_open
        try:
            os.remove(temp_file)
        except Exception:
            pass


def test_temp_file_removing():
    html = reporting.HTMLDocument('hello')
    wb_open = webbrowser.open
    webbrowser.open = _open_mock
    fd, tmpfile = tempfile.mkstemp()
    try:
        os.close(fd)
        with pytest.warns(None) as record:
            html.open_in_browser(file_name=tmpfile, temp_file_lifetime=None)
        for warning in record:
            assert "Saved HTML in temporary file" not in str(warning.message)
        html.open_in_browser(temp_file_lifetime=.5)
        assert os.path.isfile(html._temp_file)
        time.sleep(1.5)
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
    return [reporting.HTMLDocument('') for i in range(12)]


def _open_one_view():
    for i in range(12):
        v = reporting.HTMLDocument('')
    return v


def test_open_view_warning():
    # opening many views (without deleting the SurfaceView objects)
    # should raise a warning about memory usage
    assert_warns(UserWarning, _open_views)
    assert_no_warnings(_open_one_view)
    reporting.set_max_img_views_before_warning(15)
    assert_no_warnings(_open_views)
    reporting.set_max_img_views_before_warning(-1)
    assert_no_warnings(_open_views)
    reporting.set_max_img_views_before_warning(None)
    assert_no_warnings(_open_views)
    reporting.set_max_img_views_before_warning(6)
    assert_warns(UserWarning, _open_views)


def _check_html(html_view):
    """ Check the presence of some expected code in the html viewer
    """
    assert "Parameters" in str(html_view)
    assert "data:image/png;base64," in str(html_view)


def test_3d_reports():
    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))
    # check 3d datasets
    mask = input_data.NiftiMasker()
    mask.fit(data_img_3d)
    html = mask.generate_report()
    _check_html(html)

    # Check providing mask to init
    mask = np.zeros((9, 9, 9))
    mask[4:-4, 4:-4, 4:-4] = True
    mask_img_3d = Nifti1Image(data, np.eye(4))

    masker = input_data.NiftiMasker(mask_img=mask_img_3d)
    masker.fit(data_img_3d)
    html = masker.generate_report()
    _check_html(html)


def test_4d_reports():
    # Dummy mask
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    # Dummy 4D data
    data = np.zeros((10, 10, 10, 3), dtype=int)
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, np.eye(4))

    # test .fit method
    mask = input_data.NiftiMasker(mask_strategy='epi')
    mask.fit(data_img_4d)
    html = mask.generate_report()
    _check_html(html)

    # test .fit_transform method
    masker = input_data.NiftiMasker(mask_img=mask_img, standardize=True)
    masker.fit_transform(data_img_4d)
    html = masker.generate_report()
    _check_html(html)
