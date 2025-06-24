"""Test the multi_nifti_masker module."""

import shutil
from tempfile import mkdtemp

import numpy as np
import pytest
from joblib import Memory, hash
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.image import get_data
from nilearn.maskers import MultiNiftiMasker

ESTIMATORS_TO_CHECK = [MultiNiftiMasker()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


# check_multi_masker_transformer_high_variance_confounds is slow
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.fixture
def data_2(shape_3d_default):
    """Return 3D zeros with a few 10 in the center."""
    data = np.zeros(shape_3d_default)
    data[1:-2, 1:-2, 1:-2] = 10
    return data


@pytest.fixture
def img_1(data_1, affine_eye):
    """Return Nifti image of 3D zeros with a few 10 in the center."""
    return Nifti1Image(data_1, affine_eye)


@pytest.fixture
def img_2(data_2, affine_eye):
    """Return Nifti image of 3D zeros with a few 10 in the center."""
    return Nifti1Image(data_2, affine_eye)


def test_auto_mask(data_1, img_1, data_2, img_2):
    """Test that a proper mask is generated from fitted image."""
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


def test_nan():
    """Check when fitted data contains nan."""
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
    """Check mask and EIP files with different affines."""
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )
    epi_img1 = Nifti1Image(np.ones((4, 4, 4, 3)), affine=np.diag((2, 2, 2, 1)))
    epi_img2 = Nifti1Image(np.ones((3, 3, 3, 3)), affine=np.diag((3, 3, 3, 1)))

    masker = MultiNiftiMasker(mask_img=mask_img)
    epis = masker.fit_transform([epi_img1, epi_img2])
    for this_epi in epis:
        masker.inverse_transform(this_epi)


def test_3d_images(rng):
    """Test that the MultiNiftiMasker works with 3D images.

    Note that fit() requires all images in list to have the same affine.
    """
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((2, 2, 2, 1))
    )
    epi_img1 = Nifti1Image(rng.random((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(rng.random((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
    masker = MultiNiftiMasker(mask_img=mask_img)

    masker.fit_transform([epi_img1, epi_img2])


def test_joblib_cache(mask_img_1, tmp_path):
    """Check cached data."""
    filename = write_imgs_to_path(
        mask_img_1, file_path=tmp_path, create_files=True
    )
    masker = MultiNiftiMasker(mask_img=filename)
    masker.fit()
    mask_hash = hash(masker.mask_img_)
    get_data(masker.mask_img_)

    assert mask_hash == hash(masker.mask_img_)


@pytest.mark.timeout(0)
def test_shelving(rng):
    """Check behavior when shelving masker."""
    mask_img = Nifti1Image(
        np.ones((2, 2, 2), dtype=np.int8), affine=np.diag((2, 2, 2, 1))
    )
    epi_img1 = Nifti1Image(rng.random((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
    epi_img2 = Nifti1Image(rng.random((2, 2, 2)), affine=np.diag((4, 4, 4, 1)))
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
    """Create a list of random 3D nifti images."""
    return [img_3d_rand_eye] * 2


def test_mask_strategy_errors(list_random_imgs):
    """Throw error with unknown mask_strategy."""
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
    """Check different strategies to compute masks."""
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


def test_standardization(rng, shape_3d_default, affine_eye):
    """Check output properly standardized with 'standardize' parameter."""
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
