import os
import numpy as np

from nose.tools import assert_equal

import nibabel as nb
from nibabel import Nifti1Image
from nibabel.tmpdirs import InTemporaryDirectory
from nilearn._utils.compat import joblib

from nilearn.image import new_img_like
from nilearn._utils import niimg
from nilearn._utils.testing import assert_raises_regex


currdir = os.path.dirname(os.path.abspath(__file__))


def test_copy_img():
    assert_raises_regex(ValueError, "Input value is not an image",
                        niimg.copy_img, 3)


def test_copy_img_side_effect():
    img1 = Nifti1Image(np.ones((2, 2, 2, 2)), affine=np.eye(4))
    hash1 = joblib.hash(img1)
    niimg.copy_img(img1)
    hash2 = joblib.hash(img1)
    assert_equal(hash1, hash2)


def test_new_img_like_side_effect():
    img1 = Nifti1Image(np.ones((2, 2, 2, 2)), affine=np.eye(4))
    hash1 = joblib.hash(img1)
    new_img_like(img1, np.ones((2, 2, 2, 2)), img1.affine.copy(),
                 copy_header=True)
    hash2 = joblib.hash(img1)
    assert_equal(hash1, hash2)


def test_get_target_dtype():
    img = Nifti1Image(np.ones((2, 2, 2), dtype=np.float64), affine=np.eye(4))
    assert_equal(img.get_data().dtype.kind, 'f')
    dtype_kind_float = niimg._get_target_dtype(img.get_data().dtype,
                                               target_dtype='auto')
    assert_equal(dtype_kind_float, np.float32)

    img2 = Nifti1Image(np.ones((2, 2, 2), dtype=np.int64), affine=np.eye(4))
    assert_equal(img2.get_data().dtype.kind, 'i')
    dtype_kind_int = niimg._get_target_dtype(img2.get_data().dtype,
                                             target_dtype='auto')
    assert_equal(dtype_kind_int, np.int32)


def test_img_data_dtype():
    # Ignoring complex, binary, 128+ bit, RGBA
    nifti1_dtypes = (
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64)
    dtype_matches = []
    with InTemporaryDirectory():
        for logical_dtype in nifti1_dtypes:
            dataobj = np.random.uniform(0, 255,
                                        size=(2, 2, 2)).astype(logical_dtype)
            for on_disk_dtype in nifti1_dtypes:
                img = Nifti1Image(dataobj, np.eye(4))
                img.set_data_dtype(on_disk_dtype)
                img.to_filename('test.nii')
                loaded = nb.load('test.nii')
                # To verify later that sometimes these differ meaningfully
                dtype_matches.append(
                    loaded.get_data_dtype() == niimg.img_data_dtype(loaded))
                # Use np.array(dataobj) because get_data() is to be deprecated
                assert_equal(np.array(loaded.dataobj).dtype,
                             niimg.img_data_dtype(loaded))
    # Verify that the distinction is worth making
    assert any(dtype_matches)
    assert not all(dtype_matches)
