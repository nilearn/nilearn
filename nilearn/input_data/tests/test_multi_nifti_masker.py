"""
Test the multi_nifti_masker module
"""
# Author: Gael Varoquaux
# License: simplified BSD

from nose.tools import assert_true, assert_false, assert_raises
from nose import SkipTest
import numpy as np
from numpy.testing import assert_array_equal

from nibabel import Nifti1Image
import nibabel
from distutils.version import LooseVersion

from ..multi_nifti_masker import MultiNiftiMasker
from ..._utils import testing


def test_auto_mask():
    # This mostly a smoke test
    data = np.zeros((9, 9, 9))
    data[2:-2, 2:-2, 2:-2] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = MultiNiftiMasker(mask_opening=0)
    # Check that if we have not fit the masker we get a intelligible
    # error
    assert_raises(ValueError, masker.transform, [[img, ]])
    # Check error return due to bad data format
    assert_raises(ValueError, masker.fit, img)
    # Smoke test the fit
    masker.fit([[img]])

    # Test mask intersection
    data2 = np.zeros((9, 9, 9))
    data2[1:-3, 1:-3, 1:-3] = 10
    img2 = Nifti1Image(data2, np.eye(4))

    masker.fit([[img, img2]])
    assert_array_equal(masker.mask_img_.get_data(),
                       np.logical_or(data, data2))
    # Smoke test the transform
    masker.transform([[img, ]])
    # It should also work with a 3D image
    masker.transform(img)


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
    masker = MultiNiftiMasker(mask_opening=0)
    masker.fit([img])
    mask = masker.mask_img_.get_data()
    assert_true(mask[1:-1, 1:-1, 1:-1].all())
    assert_false(mask[0].any())
    assert_false(mask[:, 0].any())
    assert_false(mask[:, :, 0].any())
    assert_false(mask[-1].any())
    assert_false(mask[:, -1].any())
    assert_false(mask[:, :, -1].any())


def test_joblib_cache():
    if not LooseVersion(nibabel.__version__) > LooseVersion('1.1.0'):
        # Old nibabel do not pickle
        raise SkipTest
    from sklearn.externals.joblib import hash
    # Dummy mask
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True)\
                as filename:
        masker = MultiNiftiMasker(mask=filename)
        masker.fit()
        mask_hash = hash(masker.mask_img_)
        masker.mask_img_.get_data()
        assert_true(mask_hash == hash(masker.mask_img_))
