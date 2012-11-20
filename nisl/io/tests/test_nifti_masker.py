"""
Test the nifti_masker module
"""
# Author: Gael Varoquaux
# License: simplified BSD

from nose.tools import assert_true, assert_false, assert_raises
import numpy as np

from nibabel import Nifti1Image

from ..nifti_masker import NiftiMasker


def test_auto_mask():
    # This mostly a smoke test
    data = np.ones((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    # Check that if we have not fit the masker we get a intelligible
    # error
    assert_raises(ValueError, masker.transform, [img, ])
    # Smoke test the fit
    masker.fit(img)
    # Smoke test the transform
    masker.transform([img, ])


def test_nan():
    data = np.ones((9, 9, 9))
    data[0] = np.nan
    data[:, 0] = np.nan
    data[:, :, 0] = np.nan
    data[-1] = np.nan
    data[:, -1] = np.nan
    data[:, :, -1] = np.nan
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    masker.fit(img)
    mask = masker.mask_img_.get_data()
    assert_true(mask[1:-1, 1:-1, 1:-1].all())
    assert_false(mask[0].any())
    assert_false(mask[:, 0].any())
    assert_false(mask[:, :, 0].any())
    assert_false(mask[-1].any())
    assert_false(mask[:, -1].any())
    assert_false(mask[:, :, -1].any())
