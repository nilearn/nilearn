import warnings

import numpy as np

from nilearn import datasets, image
from nilearn.plotting import html_stat_map


def _check_html(html):
    """ Check the presence of some expected code in the html
    """
    assert isinstance(html, html_stat_map.StatMapView)
    assert "var brain =" in str(html)
    assert "overlayImg" in str(html)


def _assert_warnings_in(list1, list2):
    """ Check that warnings are inside a list
    """
    diff = list(list1 - list2)
    try:
        assert(len(diff) == 0)
    except AssertionError as e:
        e.args += ('The following warnings were not expected',
                   diff)
        raise


def test_view_stat_map():
    mni = datasets.load_mni152_template()
    with warnings.catch_warnings(record=True) as w:
        # Create a fake functional image by resample the template
        img = image.resample_img(mni, target_affine=3 * np.eye(3))
        html = html_stat_map.view_stat_map(img)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, threshold='95%')
        _check_html(html)
        html = html_stat_map.view_stat_map(img, bg_img=mni)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, bg_img=None)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, threshold=2., vmax=4.)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, symmetric_cmap=False)
        # img_4d = image.new_img_like(img, img.get_data()[:, :, :, np.newaxis])
        # assert len(img_4d.shape) == 4
        # html = html_stat_map.view_stat_map(img_4d, threshold=2., vmax=4.)
        # _check_html(html)
    warning_categories = set(warning_.category for warning_ in w)
    expected_categories = set([FutureWarning, UserWarning,
                               DeprecationWarning])
    _assert_warnings_in(warning_categories, expected_categories)
