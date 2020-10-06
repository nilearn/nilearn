"""
Test the multi_nifti_masker module
"""
# Author: Gael Varoquaux
# License: simplified BSD
import shutil
from tempfile import mkdtemp

import numpy as np
import pytest
from nibabel import Nifti1Image

from numpy.testing import assert_array_equal
from joblib import Memory

from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import write_tmp_imgs
from nilearn.input_data.multi_nifti_masker import MultiNiftiMasker
from nilearn.image import get_data


def test_auto_mask():
    # This mostly a smoke test
    data = np.zeros((9, 9, 9))
    data[2:-2, 2:-2, 2:-2] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = MultiNiftiMasker(mask_args=dict(opening=0))
    # Check that if we have not fit the masker we get a intelligible
    # error
    pytest.raises(ValueError, masker.transform, [[img, ]])
    # Check error return due to bad data format
    pytest.raises(ValueError, masker.fit, img)
    # Smoke test the fit
    masker.fit([[img]])

    # Test mask intersection
    data2 = np.zeros((9, 9, 9))
    data2[1:-3, 1:-3, 1:-3] = 10
    img2 = Nifti1Image(data2, np.eye(4))

    masker.fit([[img, img2]])
    assert_array_equal(get_data(masker.mask_img_),
                       np.logical_or(data, data2))
    # Smoke test the transform
    masker.transform([[img, ]])
    # It should also work with a 3D image
    masker.transform(img)

    # check exception when transform() called without prior fit()
    masker2 = MultiNiftiMasker(mask_img=img)
    with pytest.raises(ValueError, match='has not been fitted. '):
        masker2.transform(img2)


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
    masker = MultiNiftiMasker(mask_args=dict(opening=0))
    masker.fit([img])
    mask = get_data(masker.mask_img_)
    assert mask[1:-1, 1:-1, 1:-1].all()
    assert not mask[0].any()
    assert not mask[:, 0].any()
    assert not mask[:, :, 0].any()
    assert not mask[-1].any()
    assert not mask[:, -1].any()
    assert not mask[:, :, -1].any()


def test_different_affines():
    # Mask and EIP files with different affines
    mask_img = Nifti1Image(np.ones((2, 2, 2), dtype=np.int8),
                           affine=np.diag((4, 4, 4, 1)))
    epi_img1 = Nifti1Image(np.ones((4, 4, 4, 3)),
                           affine=np.diag((2, 2, 2, 1)))
    epi_img2 = Nifti1Image(np.ones((3, 3, 3, 3)),
                           affine=np.diag((3, 3, 3, 1)))
    masker = MultiNiftiMasker(mask_img=mask_img)
    epis = masker.fit_transform([epi_img1, epi_img2])
    for this_epi in epis:
        masker.inverse_transform(this_epi)


def test_3d_images():
    # Test that the MultiNiftiMasker works with 3D images
    mask_img = Nifti1Image(np.ones((2, 2, 2), dtype=np.int8),
                           affine=np.diag((4, 4, 4, 1)))
    epi_img1 = Nifti1Image(np.ones((2, 2, 2)),
                           affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(np.ones((2, 2, 2)),
                           affine=np.diag((2, 2, 2, 1)))
    masker = MultiNiftiMasker(mask_img=mask_img)
    epis = masker.fit_transform([epi_img1, epi_img2])
    # This is mostly a smoke test
    assert len(epis) == 2

    # verify that 4D mask arguments are refused
    mask_img_4d = Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.int8),
                              affine=np.diag((4, 4, 4, 1)))
    masker2 = MultiNiftiMasker(mask_img=mask_img_4d)
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 3D and you provided "
                             "a 4D image."):
        masker2.fit()


def test_joblib_cache():
    from joblib import hash
    # Dummy mask
    mask = np.zeros((40, 40, 40))
    mask[20, 20, 20] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    with write_tmp_imgs(mask_img, create_files=True) as filename:
        masker = MultiNiftiMasker(mask_img=filename)
        masker.fit()
        mask_hash = hash(masker.mask_img_)
        get_data(masker.mask_img_)
        assert mask_hash == hash(masker.mask_img_)
        # enables to delete "filename" on windows
        del masker


def test_shelving():

    mask_img = Nifti1Image(np.ones((2, 2, 2), dtype=np.int8),
                           affine=np.diag((4, 4, 4, 1)))
    epi_img1 = Nifti1Image(np.ones((2, 2, 2)),
                           affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(np.ones((2, 2, 2)),
                           affine=np.diag((2, 2, 2, 1)))
    cachedir = mkdtemp()
    try:
        masker_shelved = MultiNiftiMasker(mask_img=mask_img,
                                          memory=Memory(location=cachedir,
                                                        mmap_mode='r',
                                                        verbose=0))
        masker_shelved._shelving = True
        masker = MultiNiftiMasker(mask_img=mask_img)
        epis_shelved = masker_shelved.fit_transform([epi_img1, epi_img2])
        epis = masker.fit_transform([epi_img1, epi_img2])
        for epi_shelved, epi in zip(epis_shelved, epis):
            epi_shelved = epi_shelved.get()
            assert_array_equal(epi_shelved, epi)

        epi = masker.fit_transform(epi_img1)
        epi_shelved = masker_shelved.fit_transform(epi_img1)
        epi_shelved = epi_shelved.get()
        assert_array_equal(epi_shelved, epi)
    finally:
        # enables to delete "filename" on windows
        del masker
        shutil.rmtree(cachedir, ignore_errors=True)


def test_compute_multi_gray_matter_mask():
    rng = np.random.RandomState(42)

    # Check mask is correctly is correctly calculated
    imgs = [Nifti1Image(rng.uniform(size=(9, 9, 5)), np.eye(4)),
            Nifti1Image(rng.uniform(size=(9, 9, 5)), np.eye(4))]

    masker = MultiNiftiMasker(mask_strategy='template')
    masker.fit(imgs)

    # Check that the order of the images does not change the output
    masker2 = MultiNiftiMasker(mask_strategy='template')
    masker2.fit(imgs[::-1])

    mask = masker.mask_img_
    mask2 = masker2.mask_img_

    mask_ref = np.zeros((9, 9, 5))
    mask_ref[2:7, 2:7, 2] = 1

    np.testing.assert_array_equal(get_data(mask), mask_ref)
    np.testing.assert_array_equal(get_data(mask2), mask_ref)


def test_dtype():
    data = np.zeros((9, 9, 9), dtype=np.float64)
    data[2:-2, 2:-2, 2:-2] = 10
    img = Nifti1Image(data, np.eye(4))

    masker = MultiNiftiMasker(dtype='auto')
    masker.fit([[img]])

    masked_img = masker.transform([[img]])
    assert(masked_img[0].dtype == np.float32)


def test_standardization():
    rng = np.random.RandomState(42)
    data_shape = (9, 9, 5)
    n_samples = 500

    signals = rng.standard_normal(size=(2, np.prod(data_shape), n_samples))
    means = rng.standard_normal(size=(2, np.prod(data_shape), 1)) * 50 + 1000
    signals += means

    img1 = Nifti1Image(signals[0].reshape(data_shape + (n_samples,)),
                       np.eye(4))
    img2 = Nifti1Image(signals[1].reshape(data_shape + (n_samples,)),
                       np.eye(4))

    mask = Nifti1Image(np.ones(data_shape), np.eye(4))

    # z-score
    masker = MultiNiftiMasker(mask, standardize='zscore')
    trans_signals = masker.fit_transform([img1, img2])

    for ts in trans_signals:
        np.testing.assert_almost_equal(ts.mean(0), 0)
        np.testing.assert_almost_equal(ts.std(0), 1)

    # psc
    masker = MultiNiftiMasker(mask, standardize='psc')
    trans_signals = masker.fit_transform([img1, img2])

    for ts, s in zip(trans_signals, signals):
        np.testing.assert_almost_equal(ts.mean(0), 0)
        np.testing.assert_almost_equal(ts,
                                       (s / s.mean(1)[:, np.newaxis] *
                                        100 - 100).T)
