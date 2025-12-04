import pytest

from nilearn._utils.niimg_conversions import (
    check_niimg,
    check_niimg_3d,
    check_niimg_4d,
)


def test_deprecation(img_3d_ones_eye, img_4d_mni):
    with pytest.deprecated_call(match=r"0.14"):
        check_niimg(img_3d_ones_eye)
    with pytest.deprecated_call(match=r"0.14"):
        check_niimg_3d(img_3d_ones_eye)
    with pytest.deprecated_call(match=r"0.14"):
        check_niimg_4d(img_4d_mni)
