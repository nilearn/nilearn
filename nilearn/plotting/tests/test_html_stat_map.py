import warnings

from nose.tools import assert_equal
import numpy as np

from nilearn import datasets, image
from nilearn.plotting import html_stat_map


def test_encode_nii():
    mni = datasets.load_mni152_template()
    encoded = html_stat_map._encode_nii(mni)
    decoded = html_stat_map._decode_nii(encoded)
    assert np.allclose(mni.get_data(), decoded.get_data())

    mni = image.new_img_like(mni, np.asarray(mni.get_data(), dtype='>f8'))
    encoded = html_stat_map._encode_nii(mni)
    decoded = html_stat_map._decode_nii(encoded)
    assert np.allclose(mni.get_data(), decoded.get_data())

    mni = image.new_img_like(mni, np.asarray(mni.get_data(), dtype='<i4'))
    encoded = html_stat_map._encode_nii(mni)
    decoded = html_stat_map._decode_nii(encoded)
    assert np.allclose(mni.get_data(), decoded.get_data())


def _check_html(html):
    assert isinstance(html, html_stat_map.StatMapView)
    assert "var mni =" in str(html)
    assert "var stat_map =" in str(html)


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
    expected_categories = set([FutureWarning, UserWarning])
    assert_equal(warning_categories, expected_categories)
