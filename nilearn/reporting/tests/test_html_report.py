import pytest
import numpy as np
from nibabel import Nifti1Image
from nilearn import input_data


# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _check_html(html_view):
    """ Check the presence of some expected code in the html viewer
    """
    assert "Parameters" in str(html_view)
    assert "data:image/svg+xml;base64," in str(html_view)


def test_3d_reports():
    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # Dummy mask
    mask = np.zeros((9, 9, 9))
    mask[4:-4, 4:-4, 4:-4] = True
    mask_img_3d = Nifti1Image(data, np.eye(4))

    # test .fit method
    mask = input_data.NiftiMasker()
    mask.fit(data_img_3d)
    html = mask.generate_report()
    _check_html(html)

    # check providing mask to init
    masker = input_data.NiftiMasker(mask_img=mask_img_3d)
    masker.fit(data_img_3d)
    html = masker.generate_report()
    _check_html(html)

    # check providing mask to init and no images to .fit
    masker = input_data.NiftiMasker(mask_img=mask_img_3d)
    masker.fit()
    html = masker.generate_report()
    _check_html(html)


def test_4d_reports():
    # Dummy 4D data
    data = np.zeros((10, 10, 10, 3), dtype=int)
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, np.eye(4))

    # Dummy mask
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

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


def _generate_empty_report():
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # turn off reporting
    mask = input_data.NiftiMasker(reports=False)
    mask.fit(data_img_3d)
    mask.generate_report()


def test_empty_report():
    pytest.warns(UserWarning, _generate_empty_report)


def test_overlaid_report():
    pytest.importorskip('matplotlib')

    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    mask = input_data.NiftiMasker(target_affine=np.eye(3) * 8)
    mask.fit(data_img_3d)
    html = mask.generate_report()
    assert '<div class="overlay">' in str(html)
