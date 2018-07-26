import numpy as np

from nilearn import datasets
from nilearn.plotting import html_stat_map


def test_encode_nii():
    mni = datasets.load_mni152_template()
    encoded = html_stat_map._encode_nii(mni)
    decoded = html_stat_map._decode_nii(encoded)
    assert np.allclose(mni.get_data(), decoded.get_data())


def _check_html(html):
    assert isinstance(html, html_stat_map.StatMapView)
    assert "var mni =" in str(html)
    assert "var stat_map =" in str(html)


def test_view_stat_map():
    img = datasets.fetch_localizer_button_task()['tmaps'][0]
    mni = datasets.load_mni152_template()
    html = html_stat_map.view_stat_map(img)
    _check_html(html)
    html = html_stat_map.view_stat_map(img, threshold='95%')
    _check_html(html)
    html = html_stat_map.view_stat_map(img, bg_img=mni)
    _check_html(html)
    html = html_stat_map.view_stat_map(img, threshold=2., vmax=4.)
    _check_html(html)
