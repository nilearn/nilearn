"""
Test the nifti_masker module

Functions in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""
# Author: Gael Varoquaux, Philippe Gervais
# License: simplified BSD
import os
import shutil
from tempfile import mkdtemp

import nibabel
import numpy as np
from nibabel import Nifti1Image
from nose import SkipTest
from nose.tools import assert_true, assert_false, assert_raises
from numpy.testing import assert_array_equal, assert_equal

from nilearn._utils import testing
from nilearn._utils.class_inspect import get_params
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import assert_raises_regex
from nilearn.image import index_img
from nilearn.input_data.nifti_masker import NiftiMasker, filter_and_mask


def test_auto_mask():
    # This mostly a smoke test
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    # Smoke test the fit
    masker.fit(img)
    # Smoke test the transform
    # With a 4D img
    masker.transform([img, ])
    # With a 3D img
    masker.transform(img)

    # check exception when transform() called without prior fit()
    masker2 = NiftiMasker(mask_img=img)
    testing.assert_raises_regex(
        ValueError,
        'has not been fitted. ', masker2.transform, img)


def test_detrend():
    # Check that detrending doesn't do something stupid with 3D images
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    mask = data.astype(np.int)
    mask_img = Nifti1Image(mask, np.eye(4))
    masker = NiftiMasker(mask_img=mask_img, detrend=True)
    # Smoke test the fit
    X = masker.fit_transform(img)
    assert_true(np.any(X != 0))


def test_with_files():
    # Standard masking
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img) as filename:
        masker = NiftiMasker()
        masker.fit(filename)
        masker.transform(filename)


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
    masker = NiftiMasker(mask_args=dict(opening=0))
    masker.fit(img)
    mask = masker.mask_img_.get_data()
    assert_true(mask[1:-1, 1:-1, 1:-1].all())
    assert_false(mask[0].any())
    assert_false(mask[:, 0].any())
    assert_false(mask[:, :, 0].any())
    assert_false(mask[-1].any())
    assert_false(mask[:, -1].any())
    assert_false(mask[:, :, -1].any())


def test_matrix_orientation():
    """Test if processing is performed along the correct axis."""

    # the "step" kind generate heavyside-like signals for each voxel.
    # all signals being identical, standardizing along the wrong axis
    # would leave a null signal. Along the correct axis, the step remains.
    fmri, mask = testing.generate_fake_fmri(shape=(40, 41, 42), kind="step")
    masker = NiftiMasker(mask_img=mask, standardize=True, detrend=True)
    timeseries = masker.fit_transform(fmri)
    assert(timeseries.shape[0] == fmri.shape[3])
    assert(timeseries.shape[1] == mask.get_data().sum())
    std = timeseries.std(axis=0)
    assert(std.shape[0] == timeseries.shape[1])  # paranoid
    assert(not np.any(std < 0.1))

    # Test inverse transform
    masker = NiftiMasker(mask_img=mask, standardize=False, detrend=False)
    masker.fit()
    timeseries = masker.transform(fmri)
    recovered = masker.inverse_transform(timeseries)
    np.testing.assert_array_almost_equal(recovered.get_data(), fmri.get_data())


def test_mask_3d():
    # Dummy mask
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True)\
            as filename:
        masker = NiftiMasker(mask_img=filename)
        assert_raises(TypeError, masker.fit)


def test_mask_4d():
    # Dummy mask
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1
    mask_bool = mask.astype(bool)
    mask_img = Nifti1Image(mask, np.eye(4))

    # Dummy data
    data = np.zeros((10, 10, 10, 3), dtype=int)
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, np.eye(4))
    data_imgs = [index_img(data_img_4d, 0), index_img(data_img_4d, 1),
                 index_img(data_img_4d, 2)]

    # check whether transform is indeed selecting niimgs subset
    sample_mask = np.array([0, 2])
    masker = NiftiMasker(mask_img=mask_img, sample_mask=sample_mask)
    masker.fit()
    data_trans = masker.transform(data_imgs)
    data_trans_img = index_img(data_img_4d, sample_mask)
    data_trans_direct = data_trans_img.get_data()[mask_bool, :]
    data_trans_direct = np.swapaxes(data_trans_direct, 0, 1)
    assert_array_equal(data_trans, data_trans_direct)

    masker = NiftiMasker(mask_img=mask_img, sample_mask=sample_mask)
    masker.fit()
    data_trans2 = masker.transform(data_img_4d)
    assert_array_equal(data_trans2, data_trans_direct)


def test_4d_single_scan():
    mask = np.zeros((10, 10, 10))
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    # Test that, in list of 4d images with last dimension=1, they are
    # considered as 3d

    data_5d = [np.random.random((10, 10, 10, 1)) for i in range(5)]
    data_4d = [d[..., 0] for d in data_5d]
    data_5d = [nibabel.Nifti1Image(d, np.eye(4)) for d in data_5d]
    data_4d = [nibabel.Nifti1Image(d, np.eye(4)) for d in data_4d]

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    data_trans_5d = masker.transform(data_5d)
    data_trans_4d = masker.transform(data_4d)

    assert_array_equal(data_trans_4d, data_trans_5d)


def test_5d():
    mask = np.zeros((10, 10, 10))
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    # Test that, in list of 4d images with last dimension=1, they are
    # considered as 3d

    data_5d = [np.random.random((10, 10, 10, 3)) for i in range(5)]
    data_5d = [nibabel.Nifti1Image(d, np.eye(4)) for d in data_5d]

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    testing.assert_raises_regex(
        DimensionError,
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a list of 4D images \(5D\).",
        masker.transform, data_5d)


def test_sessions():
    # Test the sessions vector
    data = np.ones((40, 40, 40, 4))
    # Create a border, so that the masking work well
    data[0] = 0
    data[-1] = 0
    data[:, -1] = 0
    data[:, 0] = 0
    data[..., -1] = 0
    data[..., 0] = 0
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker(sessions=np.ones(3, dtype=np.int))
    assert_raises(ValueError, masker.fit_transform, data_img)


def test_joblib_cache():
    from sklearn.externals.joblib import hash, Memory
    mask = np.zeros((40, 40, 40))
    mask[20, 20, 20] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    with testing.write_tmp_imgs(mask_img, create_files=True) as filename:
        masker = NiftiMasker(mask_img=filename)
        masker.fit()
        mask_hash = hash(masker.mask_img_)
        masker.mask_img_.get_data()
        assert_true(mask_hash == hash(masker.mask_img_))

        # Test a tricky issue with memmapped joblib.memory that makes
        # imgs return by inverse_transform impossible to save
        cachedir = mkdtemp()
        try:
            masker.memory = Memory(cachedir=cachedir, mmap_mode='r',
                                   verbose=0)
            X = masker.transform(mask_img)
            # inverse_transform a first time, so that the result is cached
            out_img = masker.inverse_transform(X)
            out_img = masker.inverse_transform(X)
            out_img.to_filename(os.path.join(cachedir, 'test.nii'))
        finally:
            # enables to delete "filename" on windows
            del masker
            shutil.rmtree(cachedir, ignore_errors=True)


def test_mask_init_errors():
    # Errors that are caught in init
    mask = NiftiMasker(mask_strategy='oops')
    testing.assert_raises_regex(
        ValueError, "Unknown value of mask_strategy 'oops'",
        mask.fit)


def test_compute_epi_mask():
    # Taken from test_masking.py, but used to test that the masker class
    #   is passing parameters appropriately.
    mean_image = np.ones((9, 9, 3))
    mean_image[3:-2, 3:-2, :] = 10
    mean_image[5, 5, :] = 11
    mean_image = Nifti1Image(mean_image.astype(float), np.eye(4))

    masker = NiftiMasker(mask_strategy='epi',
                         mask_args=dict(opening=False))
    masker.fit(mean_image)
    mask1 = masker.mask_img_

    masker2 = NiftiMasker(mask_strategy='epi',
                          mask_args=dict(opening=False, exclude_zeros=True))
    masker2.fit(mean_image)
    mask2 = masker2.mask_img_

    # With an array with no zeros, exclude_zeros should not make
    # any difference
    np.testing.assert_array_equal(mask1.get_data(), mask2.get_data())

    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = mean_image.get_data()
    mean_image2 = Nifti1Image(mean_image2, np.eye(4))

    masker3 = NiftiMasker(mask_strategy='epi',
                          mask_args=dict(opening=False, exclude_zeros=True))
    masker3.fit(mean_image2)
    mask3 = masker3.mask_img_
    np.testing.assert_array_equal(mask1.get_data(),
                                  mask3.get_data()[3:12, 3:12])

    # However, without exclude_zeros, it does
    masker4 = NiftiMasker(mask_strategy='epi', mask_args=dict(opening=False))
    masker4.fit(mean_image2)
    mask4 = masker4.mask_img_

    assert_false(np.allclose(mask1.get_data(),
                             mask4.get_data()[3:12, 3:12]))


def test_filter_and_mask_error():
    data = np.zeros([20, 30, 40, 5])
    mask = np.zeros([20, 30, 40, 2])
    mask[10, 15, 20, :] = 1

    data_img = nibabel.Nifti1Image(data, np.eye(4))
    mask_img = nibabel.Nifti1Image(mask, np.eye(4))

    masker = NiftiMasker()
    params = get_params(NiftiMasker, masker)

    assert_raises_regex(DimensionError,
                        "Input data has incompatible dimensionality: "
                        "Expected dimension is 3D and you provided "
                        "a 4D image.",
                        filter_and_mask,
                        data_img, mask_img, params)


def test_filter_and_mask():
    data = np.zeros([20, 30, 40, 5])
    mask = np.ones([20, 30, 40])

    data_img = nibabel.Nifti1Image(data, np.eye(4))
    mask_img = nibabel.Nifti1Image(mask, np.eye(4))

    masker = NiftiMasker()
    params = get_params(NiftiMasker, masker)

    # Test return_affine = False
    data = filter_and_mask(data_img, mask_img, params)
    assert_equal(data.shape, (5, 24000))
