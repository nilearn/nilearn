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


def _assert_warnings_in(set1, set2):
    """ Check that warnings are inside a list
    """
    assert set1.issubset(set2), ("the following warnings were not "
                                 "expected: {}").format(set1.difference(set2))

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
        img_4d = image.new_img_like(img, img.get_data()[:, :, :, np.newaxis])
        assert len(img_4d.shape) == 4
        html = html_stat_map.view_stat_map(img_4d, threshold=2., vmax=4.)
        _check_html(html)
    warning_categories = set(warning_.category for warning_ in w)
    expected_categories = set([FutureWarning, UserWarning,
                               DeprecationWarning])
    _assert_warnings_in(warning_categories, expected_categories)


def test_data2sprite():

    # Generate a simulated volume with a square inside
    data = np.zeros([8,8,8])
    data[2:6,2:6,2:6] = 1

    # turn that into a sprite and check it has the right shape
    sprite = html_stat_map._data2sprite(data)
    assert sprite.shape == (24, 24)

    # Generate ground truth for the sprite
    Z = np.zeros([8,8])
    Zr = np.zeros([2,8])
    Or = np.tile(np.array([[0, 0, 1, 1, 1, 1, 0, 0]]),[4,1])
    O = np.concatenate((Zr,Or,Zr),axis=0)
    gtruth = np.concatenate((np.concatenate((Z,Z,O), axis=1),
                             np.concatenate((O,O,O), axis=1),
                             np.concatenate((Z,Z,Z), axis=1)),
                             axis=0)

    # Check that the sprite matches ground truth
    assert (sprite == gtruth).all()


def test_get_vmin_vmax():

    # Generate a simulated volume with a square inside
    data = np.zeros([8,8,8])
    data[2:6,2:6,2:6] = 1

    # Check default vmin, vmax
    vmin, vmax = html_stat_map._get_vmin_vmax(data)
    assert (vmin == 0) and (vmax == 1)

    # Force vmin and vmax
    vmin, vmax = html_stat_map._get_vmin_vmax(data, vmin=.5, vmax=.7)
    assert (vmin == .5) and (vmax == .7)

    # Try to use NaN for vmax
    with warnings.catch_warnings(record=True) as w:
        vmin, vmax = html_stat_map._get_vmin_vmax(data, vmin=.5, vmax=np.nan)
    warning_categories = set(warning_.category for warning_ in w)
    expected_categories = set([UserWarning])
    assert(expected_categories.issubset(warning_categories))

    # Try to use NaN for vmin
    with warnings.catch_warnings(record=True) as w:
        vmin, vmax = html_stat_map._get_vmin_vmax(data, vmin=np.nan, vmax=0.7)
    warning_categories = set(warning_.category for warning_ in w)
    expected_categories = set([UserWarning])
    assert(expected_categories.issubset(warning_categories))

    # Try to feed vmax smaller than vmin
    try:
        vmin, vmax = html_stat_map._get_vmin_vmax(data, vmin=3, vmax=0.7)
    except ValueError:
        pass
    
