import warnings

import numpy as np
import pytest

from nibabel import Nifti1Image

from nilearn import datasets, image
from nilearn.plotting import html_stat_map
from nilearn.image import new_img_like
from nilearn.image import get_data
from nilearn.externals.brainsprite.brainsprite import StatMapView

def _check_html(html_view, title=None):
    """ Check the presence of some expected code in the html viewer
    """
    assert isinstance(html_view, StatMapView)
    assert "var brain =" in str(html_view)
    assert "overlayImg" in str(html_view)
    if title is not None:
        assert "<title>{}</title>".format(title) in str(html_view)


def test_view_img():
    mni = datasets.load_mni152_template()
    with warnings.catch_warnings(record=True) as w:
        # Create a fake functional image by resample the template
        img = image.resample_img(mni, target_affine=3 * np.eye(3))
        html_view = html_stat_map.view_img(img)
        _check_html(html_view, title="Slice viewer")
        html_view = html_stat_map.view_img(img, threshold='95%',
                                           title="SOME_TITLE")
        _check_html(html_view, title="SOME_TITLE")
        html_view = html_stat_map.view_img(img, bg_img=mni)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, bg_img=None)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, threshold=2., vmax=4.)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, symmetric_cmap=False)
        img_4d = image.new_img_like(img, get_data(img)[:, :, :, np.newaxis])
        assert len(img_4d.shape) == 4
        html_view = html_stat_map.view_img(img_4d, threshold=2., vmax=4.)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img_4d, threshold=1e6)
        _check_html(html_view)

    # Check that all warnings were expected
    warnings_set = set(warning_.category for warning_ in w)
    expected_set = set([FutureWarning, UserWarning,
                       DeprecationWarning])
    assert warnings_set.issubset(expected_set), (
        "the following warnings were not expected: {}").format(
        warnings_set.difference(expected_set))
