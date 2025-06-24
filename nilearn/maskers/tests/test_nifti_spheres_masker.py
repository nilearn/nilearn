"""Test nilearn.maskers.nifti_spheres_masker."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import get_data, new_img_like
from nilearn.maskers import NiftiSpheresMasker

ESTIMATORS_TO_CHECK = [NiftiSpheresMasker(seeds=[(1, 1, 1)])]

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


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def test_seed_extraction(rng, affine_eye):
    """Test seed extraction."""
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, affine_eye)
    masker = NiftiSpheresMasker([(1, 1, 1)])

    # Test the fit
    masker.fit()

    # Test the transform
    s = masker.transform(img)

    assert_array_equal(s[:, 0], data[1, 1, 1])


def test_sphere_extraction(rng, affine_eye):
    """Test sphere extraction."""
    seed = (1, 1, 1)

    data = rng.random((3, 3, 3, 5))

    img = Nifti1Image(data, affine_eye)

    masker = NiftiSpheresMasker([seed], radius=1)

    masker.fit()

    # Check attributes defined at fit
    assert masker.n_elements_ == 1

    # Test the transform
    s = masker.transform(img)

    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 1, 1] = True
    mask[1, :, 1] = True
    mask[1, 1, :] = True
    assert_array_equal(s[:, 0], np.mean(data[mask], axis=0))

    # Now with a mask
    mask_img = np.zeros((3, 3, 3))
    mask_img[1, :, :] = 1
    mask_img = Nifti1Image(mask_img, affine_eye)

    masker = NiftiSpheresMasker([seed], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)

    assert_array_equal(
        s[:, 0],
        np.mean(data[np.logical_and(mask, get_data(mask_img))], axis=0),
    )


def test_anisotropic_sphere_extraction(rng, affine_eye):
    """Test non anisotropic sphere extraction."""
    seed = (2, 1, 2)

    data = rng.random((3, 3, 3, 5))

    affine = affine_eye
    affine[0, 0] = 2
    affine[2, 2] = 2

    img = Nifti1Image(data, affine_eye)

    masker = NiftiSpheresMasker([seed], radius=1)

    # Test the fit
    masker.fit()

    # Test the transform
    s = masker.transform(img)

    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, :, 1] = True
    assert_array_equal(s[:, 0], np.mean(data[mask], axis=0))

    # Now with a mask
    mask_img = np.zeros((3, 2, 3))
    mask_img[1, 0, 1] = 1

    affine_2 = affine_eye.copy()
    affine_2[0, 0] = 4

    mask_img = Nifti1Image(mask_img, affine=affine_2)

    masker = NiftiSpheresMasker([seed], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)

    assert_array_equal(s[:, 0], data[1, 0, 1])


def test_errors():
    """Check seed input."""
    masker = NiftiSpheresMasker(([1, 2]), radius=0.2)
    with pytest.raises(ValueError, match="Seeds must be a list .+"):
        masker.fit()


def test_nifti_spheres_masker_overlap(rng, affine_eye):
    """Throw error when allow_overlap=False and some spheres overlap."""
    shape = (5, 5, 5)

    data = rng.random((*shape, 5))
    fmri_img = Nifti1Image(data, affine_eye)

    seeds = [(0, 0, 0), (2, 2, 2)]

    overlapping_masker = NiftiSpheresMasker(
        seeds, radius=1, allow_overlap=True
    )
    overlapping_masker.fit_transform(fmri_img)

    overlapping_masker = NiftiSpheresMasker(
        seeds, radius=2, allow_overlap=True
    )
    overlapping_masker.fit_transform(fmri_img)

    noverlapping_masker = NiftiSpheresMasker(
        seeds, radius=1, allow_overlap=False
    )
    noverlapping_masker.fit_transform(fmri_img)

    noverlapping_masker = NiftiSpheresMasker(
        seeds, radius=2, allow_overlap=False
    )

    with pytest.raises(ValueError, match="Overlap detected"):
        noverlapping_masker.fit_transform(fmri_img)


def test_small_radius(rng):
    """Check behavior when radius smaller than voxel size."""
    shape = (3, 3, 3)

    data = rng.random(shape)

    mask = np.zeros(shape)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1

    affine = np.eye(4) * 1.2

    seed = (1.4, 1.4, 1.4)

    masker = NiftiSpheresMasker(
        [seed], radius=0.1, mask_img=Nifti1Image(mask, affine)
    )
    spheres_data = masker.fit_transform(Nifti1Image(data, affine))
    masker.inverse_transform(spheres_data)

    # Test if masking is taken into account
    mask[1, 1, 1] = 0
    mask[1, 1, 0] = 1

    masker = NiftiSpheresMasker(
        [seed], radius=0.1, mask_img=Nifti1Image(mask, affine)
    )

    with pytest.raises(ValueError, match="These spheres are empty"):
        masker.fit_transform(Nifti1Image(data, affine))

    masker.fit(Nifti1Image(data, affine))

    with pytest.raises(ValueError, match="These spheres are empty"):
        masker.inverse_transform(spheres_data)

    # Inverse transform should still work with a masker larger radius
    masker = NiftiSpheresMasker(
        [seed], radius=1.6, mask_img=Nifti1Image(mask, affine)
    )
    masker.fit(Nifti1Image(data, affine))
    masker.inverse_transform(spheres_data)


def test_is_nifti_spheres_masker_give_nans(rng, affine_eye):
    """Check behavior when data to fit_transform contains nan."""
    data_with_nans = np.zeros((10, 10, 10), dtype=np.float32)
    data_with_nans[:, :, :] = np.nan

    data_without_nans = rng.random((9, 9, 9))
    indices = np.nonzero(data_without_nans)

    # Leaving nans outside of some data
    data_with_nans[indices] = data_without_nans[indices]
    img = Nifti1Image(data_with_nans, affine_eye)

    # Interaction of seed with nans
    seed = [(7, 7, 7)]
    masker = NiftiSpheresMasker(seeds=seed, radius=2.0)

    assert not np.isnan(np.sum(masker.fit_transform(img)))

    # When mask_img is provided, the seed interacts within the brain, so no nan
    mask = np.ones((9, 9, 9))
    mask_img = Nifti1Image(mask, affine_eye)
    masker = NiftiSpheresMasker(seeds=seed, radius=2.0, mask_img=mask_img)

    assert not np.isnan(np.sum(masker.fit_transform(img)))


def test_standardization(rng, affine_eye):
    """Check output properly standardized with 'standardize' parameter."""
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, affine_eye)

    # test zscore
    masker = NiftiSpheresMasker([(1, 1, 1)], standardize="zscore_sample")
    # Test the fit
    s = masker.fit_transform(img)

    np.testing.assert_almost_equal(s.mean(), 0)
    np.testing.assert_almost_equal(s.std(), 1, decimal=1)

    # test psc
    masker = NiftiSpheresMasker([(1, 1, 1)], standardize="psc")
    # Test the fit
    s = masker.fit_transform(img)

    np.testing.assert_almost_equal(s.mean(), 0)
    np.testing.assert_almost_equal(
        s.ravel(),
        data[1, 1, 1] / data[1, 1, 1].mean() * 100 - 100,
    )


def test_nifti_spheres_masker_inverse_transform(rng, affine_eye):
    """Applying the sphere_extraction example from above backwards."""
    data = rng.random((3, 3, 3, 5))

    img = Nifti1Image(data, affine_eye)

    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)

    # Test the fit
    masker.fit()

    # Transform data
    signal = masker.transform(img)
    with pytest.raises(ValueError, match="Please provide mask_img"):
        masker.inverse_transform(signal)

    # Now with a mask
    mask_img = np.zeros((3, 3, 3))
    mask_img[1, :, :] = 1
    mask_img = Nifti1Image(mask_img, affine_eye)

    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)

    # Mask describes the extend of the masker's sphere
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 1, 1] = True
    mask[1, :, 1] = True
    mask[1, 1, :] = True

    # Create an array mask
    array_mask = np.logical_and(mask, get_data(mask_img))

    inverse_map = masker.inverse_transform(s)

    # Testing whether mask is applied to inverse transform
    assert_array_equal(
        np.mean(get_data(inverse_map), axis=-1) != 0, array_mask
    )
    # Test whether values are preserved
    assert_array_equal(get_data(inverse_map)[array_mask].mean(0), s[:, 0])

    # Test whether the mask's shape is applied
    assert_array_equal(inverse_map.shape[:3], mask_img.shape)


def test_nifti_spheres_masker_inverse_overlap(rng, affine_eye):
    """Throw error when data to inverse_transform has overlapping data and \
        allow_overlap=False.
    """
    shape = (5, 5, 5)

    data = rng.random((*shape, 5))
    fmri_img = Nifti1Image(data, affine_eye)

    # Apply mask image - to allow inversion
    mask_img = new_img_like(fmri_img, np.ones(shape))
    seeds = [(0, 0, 0), (2, 2, 2)]
    # Inverse data
    inv_data = rng.random(len(seeds))

    overlapping_masker = NiftiSpheresMasker(
        seeds, radius=1, allow_overlap=True, mask_img=mask_img
    ).fit()
    overlapping_masker.inverse_transform(inv_data)

    overlapping_masker = NiftiSpheresMasker(
        seeds, radius=2, allow_overlap=True, mask_img=mask_img
    ).fit()

    overlap = overlapping_masker.inverse_transform(inv_data)

    # Test whether overlapping data is averaged
    assert_array_almost_equal(get_data(overlap)[1, 1, 1], np.mean(inv_data))

    noverlapping_masker = NiftiSpheresMasker(
        seeds, radius=1, allow_overlap=False, mask_img=mask_img
    ).fit()

    noverlapping_masker.inverse_transform(inv_data)
    noverlapping_masker = NiftiSpheresMasker(
        seeds, radius=2, allow_overlap=False, mask_img=mask_img
    ).fit()

    with pytest.raises(ValueError, match="Overlap detected"):
        noverlapping_masker.inverse_transform(inv_data)
