"""Tests for nilearn.mass_univariate._utils."""
import nibabel as nib
import numpy as np

from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import _utils


def test__calculate_tfce():
    """Test _calculate_tfce."""
    test_arr4d = np.zeros((10, 10, 10, 1))

    # 10-voxel positive cluster, high intensity
    test_arr4d[:2, :2, :2, 0] = 10
    test_arr4d[0, 2, 0, 0] = 10
    test_arr4d[2, 0, 0, 0] = 10

    # 10-voxel negative cluster, higher intensity
    test_arr4d[3:5, 3:5, 3:5, 0] = -11
    test_arr4d[3, 5, 3, 0] = -11
    test_arr4d[5, 3, 3, 0] = -11

    mask_img = nib.Nifti1Image(np.ones(test_arr4d.shape[:3], int), np.eye(4))
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)
    data_img = nib.Nifti1Image(test_arr4d, mask_img.affine, mask_img.header)
    test_arr2d = masker.transform(data_img).T

    # One-sided test where positive cluster has the highest TFCE
    true_max_tfce = 5050
    test_tfce_arr2d = _utils._calculate_tfce(
        test_arr2d,
        masker=masker,
        E=1,
        H=1,
        dh='auto',
        two_sided=False,
    )
    assert test_tfce_arr2d.shape == (1000, 1)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce

    # Two-sided test where negative cluster has the highest TFCE
    true_max_tfce = 5555
    test_tfce_arr2d = _utils._calculate_tfce(
        test_arr2d,
        masker=masker,
        E=1,
        H=1,
        dh='auto',
        two_sided=True,
    )
    assert test_tfce_arr2d.shape == (1000, 1)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce

    # One-sided test with preset dh
    true_max_tfce = 550
    test_tfce_arr2d = _utils._calculate_tfce(
        test_arr2d,
        masker=masker,
        E=1,
        H=1,
        dh=1,
        two_sided=False,
    )
    assert test_tfce_arr2d.shape == (1000, 1)
    assert np.max(np.abs(test_tfce_arr2d)) == true_max_tfce
