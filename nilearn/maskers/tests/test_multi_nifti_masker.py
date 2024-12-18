"""Test the multi_nifti_masker module."""

# Author: Gael Varoquaux, Ana Luisa Pinho
import shutil
from tempfile import mkdtemp

import numpy as np
import pytest
from joblib import Memory
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import write_imgs_to_path
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiMasker

extra_valid_checks = [
    "check_estimators_unfitted",
    "check_get_params_invariance",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
    "check_parameters_default_constructible",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[MultiNiftiMasker()],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[MultiNiftiMasker()],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.fixture
def data_2(shape_3d_default):
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def img_1(data_1, affine_eye):
    return Nifti1Image(data_1, affine_eye)


@pytest.fixture
def img_2(data_2, affine_eye):
    return Nifti1Image(data_2, affine_eye)


def test_auto_mask(data_1, img_1, data_2, img_2):
    # This mostly a smoke test
    masker = MultiNiftiMasker(mask_args={"opening": 0})

    # Smoke test the fit
    masker.fit([[img_1]])

    # Test mask intersection
    masker.fit([[img_1, img_2]])
    assert_array_equal(
        get_data(masker.mask_img_), np.logical_or(data_1, data_2)
    )
    # Smoke test the transform
    masker.transform([[img_1]])
    # It should also work with a 3D image
    masker.transform(img_1)


def test_auto_mask_errors(img_3d_rand_eye, img_2):
    masker = MultiNiftiMasker(mask_args={"opening": 0})
    # Check that if we have not fit the masker we get a intelligible
    # error
    with pytest.raises(ValueError, match="has not been fitted. "):
        masker.transform(
            [[img_3d_rand_eye]],
        )
    # Check error return due to bad data format
    with pytest.raises(
        ValueError,
        match="For multiple processing, you should  provide a list of data",
    ):
        masker.fit(img_3d_rand_eye)

    # check exception when transform() called without prior fit()
    masker2 = MultiNiftiMasker(mask_img=img_3d_rand_eye)
    with pytest.raises(ValueError, match="has not been fitted. "):
        masker2.transform(img_2)


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

    masker = MultiNiftiMasker(mask_args={"opening": 0})
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
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )
    epi_img1 = Nifti1Image(np.ones((4, 4, 4, 3)), affine=np.diag((2, 2, 2, 1)))
    epi_img2 = Nifti1Image(np.ones((3, 3, 3, 3)), affine=np.diag((3, 3, 3, 1)))

    masker = MultiNiftiMasker(mask_img=mask_img)
    epis = masker.fit_transform([epi_img1, epi_img2])
    for this_epi in epis:
        masker.inverse_transform(this_epi)


def test_3d_images():
    # Test that the MultiNiftiMasker works with 3D images
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )
    epi_img1 = Nifti1Image(np.ones((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(np.ones((2, 2, 2)), affine=np.diag((2, 2, 2, 1)))
    masker = MultiNiftiMasker(mask_img=mask_img)

    # Check attributes defined at fit
    assert not hasattr(masker, "mask_img_")
    assert not hasattr(masker, "n_elements_")

    epis = masker.fit_transform([epi_img1, epi_img2])

    # This is mostly a smoke test
    assert len(epis) == 2

    # Check attributes defined at fit
    assert hasattr(masker, "mask_img_")
    assert hasattr(masker, "n_elements_")


def test_3d_images_error(img_4d_ones_eye):
    """Verify that 4D mask arguments are refused."""
    masker2 = MultiNiftiMasker(mask_img=img_4d_ones_eye)
    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided a 4D image.",
    ):
        masker2.fit()


def test_joblib_cache(mask_img_1, tmp_path):
    from joblib import hash

    filename = write_imgs_to_path(
        mask_img_1, file_path=tmp_path, create_files=True
    )
    masker = MultiNiftiMasker(mask_img=filename)
    masker.fit()
    mask_hash = hash(masker.mask_img_)
    get_data(masker.mask_img_)

    assert mask_hash == hash(masker.mask_img_)
    # enables to delete "filename" on windows
    del masker


def test_shelving():
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )
    epi_img1 = Nifti1Image(np.ones((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(np.ones((2, 2, 2)), affine=np.diag((2, 2, 2, 1)))
    cachedir = mkdtemp()
    try:
        masker_shelved = MultiNiftiMasker(
            mask_img=mask_img,
            memory=Memory(location=cachedir, mmap_mode="r", verbose=0),
        )
        masker_shelved._shelving = True
        epis_shelved = masker_shelved.fit_transform([epi_img1, epi_img2])
        masker = MultiNiftiMasker(mask_img=mask_img)
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


@pytest.fixture
def list_random_imgs(img_3d_rand_eye):
    return [img_3d_rand_eye] * 2


def test_mask_strategy_errors(list_random_imgs):
    # Error with unknown mask_strategy
    mask = MultiNiftiMasker(mask_strategy="foo")

    with pytest.raises(
        ValueError, match="Unknown value of mask_strategy 'foo'"
    ):
        mask.fit(list_random_imgs)

    # Warning with deprecated 'template' strategy,
    # plus an exception because there's no resulting mask
    mask = MultiNiftiMasker(mask_strategy="template")
    with pytest.warns(
        UserWarning, match="Masking strategy 'template' is deprecated."
    ):
        mask.fit(list_random_imgs)


@pytest.mark.parametrize(
    "strategy", [f"{p}-template" for p in ["whole-brain", "gm", "wm"]]
)
def test_compute_mask_strategy(strategy, shape_3d_default, list_random_imgs):
    masker = MultiNiftiMasker(mask_strategy=strategy, mask_args={"opening": 1})
    masker.fit(list_random_imgs)

    # Check that the order of the images does not change the output
    masker2 = MultiNiftiMasker(
        mask_strategy=strategy, mask_args={"opening": 1}
    )
    masker2.fit(list_random_imgs[::-1])
    mask_ref = np.zeros(shape_3d_default, dtype="int8")

    np.testing.assert_array_equal(get_data(masker.mask_img_), mask_ref)
    np.testing.assert_array_equal(get_data(masker2.mask_img_), mask_ref)


def test_dtype(affine_eye):
    """Check dtype returned by transform when using auto."""
    data = np.zeros((9, 10, 11), dtype=np.float64)
    data[2:-2, 2:-2, 2:-2] = 10
    img = Nifti1Image(data, affine_eye)

    masker = MultiNiftiMasker(dtype="auto")
    masker.fit([[img]])

    masked_img = masker.transform([[img]])
    assert masked_img[0].dtype == np.float32


def test_standardization(rng, shape_3d_default, affine_eye):
    n_samples = 500

    signals = rng.standard_normal(
        size=(2, np.prod(shape_3d_default), n_samples)
    )
    means = (
        rng.standard_normal(size=(2, np.prod(shape_3d_default), 1)) * 50 + 1000
    )
    signals += means

    img1 = Nifti1Image(
        signals[0].reshape((*shape_3d_default, n_samples)), affine_eye
    )
    img2 = Nifti1Image(
        signals[1].reshape((*shape_3d_default, n_samples)), affine_eye
    )

    mask = Nifti1Image(np.ones(shape_3d_default), affine_eye)

    # z-score
    masker = MultiNiftiMasker(mask, standardize="zscore_sample")
    trans_signals = masker.fit_transform([img1, img2])

    for ts in trans_signals:
        np.testing.assert_almost_equal(ts.mean(0), 0)
        np.testing.assert_almost_equal(ts.std(0), 1, decimal=3)

    # psc
    masker = MultiNiftiMasker(mask, standardize="psc")
    trans_signals = masker.fit_transform([img1, img2])

    for ts, s in zip(trans_signals, signals):
        np.testing.assert_almost_equal(ts.mean(0), 0)
        np.testing.assert_almost_equal(
            ts, (s / s.mean(1)[:, np.newaxis] * 100 - 100).T
        )
