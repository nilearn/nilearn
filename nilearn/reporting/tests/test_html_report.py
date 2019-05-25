import numpy as np
from nibabel import Nifti1Image
from nilearn import input_data


def _check_html(html_view):
    """ Check the presence of some expected code in the html viewer
    """
    assert "Parameters" in str(html_view)
    assert "data:image/png;base64," in str(html_view)


def test_3d():
    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))
    # check 3d datasets
    mask = input_data.NiftiMasker()
    mask.fit(data_img_3d)
    html = mask.generate_report()
    _check_html(html)


def test_4d():
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


def test_empty():
    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # test when reporting not enabled
    mask = input_data.NiftiMasker(reports=False)
    mask.fit(data_img_3d)
    html = mask.generate_report()
    _check_html(html)
    assert 'undraw_blank_canvas.png' in str(html)
