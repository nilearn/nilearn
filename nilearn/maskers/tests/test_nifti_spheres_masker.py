"""Test nilearn.maskers.nifti_spheres_masker."""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nilearn._utils import data_gen
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.image import get_data, new_img_like
from nilearn.maskers import NiftiSpheresMasker

extra_valid_checks = [
    "check_estimators_unfitted",
    "check_get_params_invariance",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[NiftiSpheresMasker([(1, 1, 1)])],
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
        estimator=[NiftiSpheresMasker([(1, 1, 1)])],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_seed_extraction(rng):
    """Test seed extraction."""
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)])
    # Test the fit
    masker.fit()
    # Test the transform
    s = masker.transform(img)
    assert_array_equal(s[:, 0], data[1, 1, 1])


def test_sphere_extraction(rng):
    """Test sphere extraction."""
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)

    # Check attributes defined at fit
    assert not hasattr(masker, "seeds_")
    assert not hasattr(masker, "n_elements_")

    masker.fit()

    # Check attributes defined at fit
    assert hasattr(masker, "seeds_")
    assert hasattr(masker, "n_elements_")
    assert masker.n_elements_ == 1

    # Test the fit
    masker.fit()

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
    mask_img = Nifti1Image(mask_img, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)
    assert_array_equal(
        s[:, 0],
        np.mean(data[np.logical_and(mask, get_data(mask_img))], axis=0),
    )


def test_anisotropic_sphere_extraction(rng):
    data = rng.random((3, 3, 3, 5))
    affine = np.eye(4)
    affine[0, 0] = 2
    affine[2, 2] = 2
    img = Nifti1Image(data, affine)
    masker = NiftiSpheresMasker([(2, 1, 2)], radius=1)
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
    affine_2 = affine.copy()
    affine_2[0, 0] = 4
    mask_img = Nifti1Image(mask_img, affine=affine_2)
    masker = NiftiSpheresMasker([(2, 1, 2)], radius=1, mask_img=mask_img)

    masker.fit()
    s = masker.transform(img)
    assert_array_equal(s[:, 0], data[1, 0, 1])


def test_errors():
    masker = NiftiSpheresMasker(([1, 2]), radius=0.2)
    with pytest.raises(ValueError, match="Seeds must be a list .+"):
        masker.fit()


def test_nifti_spheres_masker_overlap(rng):
    # Test resampling in NiftiMapsMasker
    affine = np.eye(4)
    shape = (5, 5, 5)

    data = rng.random((*shape, 5))
    fmri_img = Nifti1Image(data, affine)

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
    affine = np.eye(4)
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
    masker.fit_transform(Nifti1Image(data, affine))

    # Test if masking is taken into account
    mask[1, 1, 1] = 0
    mask[1, 1, 0] = 1

    masker = NiftiSpheresMasker(
        [seed], radius=0.1, mask_img=Nifti1Image(mask, affine)
    )
    with pytest.raises(ValueError, match="These spheres are empty"):
        masker.fit_transform(Nifti1Image(data, affine))

    masker = NiftiSpheresMasker(
        [seed], radius=1.6, mask_img=Nifti1Image(mask, affine)
    )
    masker.fit_transform(Nifti1Image(data, affine))


def test_is_nifti_spheres_masker_give_nans(rng):
    affine = np.eye(4)

    data_with_nans = np.zeros((10, 10, 10), dtype=np.float32)
    data_with_nans[:, :, :] = np.nan

    data_without_nans = rng.random((9, 9, 9))
    indices = np.nonzero(data_without_nans)

    # Leaving nans outside of some data
    data_with_nans[indices] = data_without_nans[indices]
    img = Nifti1Image(data_with_nans, affine)
    seed = [(7, 7, 7)]

    # Interaction of seed with nans
    masker = NiftiSpheresMasker(seeds=seed, radius=2.0)
    assert not np.isnan(np.sum(masker.fit_transform(img)))

    mask = np.ones((9, 9, 9))
    mask_img = Nifti1Image(mask, affine)
    # When mask_img is provided, the seed interacts within the brain, so no nan
    masker = NiftiSpheresMasker(seeds=seed, radius=2.0, mask_img=mask_img)
    assert not np.isnan(np.sum(masker.fit_transform(img)))


def test_standardization(rng):
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, np.eye(4))

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


def test_nifti_spheres_masker_inverse_transform(rng):
    # Applying the sphere_extraction example from above backwards
    data = rng.random((3, 3, 3, 5))
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)
    # Test the fit
    masker.fit()
    # Transform data
    with pytest.raises(ValueError, match="Please provide mask_img"):
        masker.inverse_transform(data[0, 0, 0, :])

    # Mask describes the extend of the masker's sphere
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 1, 1] = True
    mask[1, :, 1] = True
    mask[1, 1, :] = True

    # Now with a mask
    mask_img = np.zeros((3, 3, 3))
    mask_img[1, :, :] = 1
    mask_img = Nifti1Image(mask_img, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)
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


def test_nifti_spheres_masker_inverse_overlap(rng):
    # Test overlapping data in inverse_transform
    affine = np.eye(4)
    shape = (5, 5, 5)

    data = rng.random((*shape, 5))
    fmri_img = Nifti1Image(data, affine)

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


def test_small_radius_inverse(rng):
    affine = np.eye(4)
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
    masker.fit(Nifti1Image(data, affine))

    with pytest.raises(ValueError, match="These spheres are empty"):
        masker.inverse_transform(spheres_data)

    masker = NiftiSpheresMasker(
        [seed], radius=1.6, mask_img=Nifti1Image(mask, affine)
    )
    masker.fit(Nifti1Image(data, affine))
    masker.inverse_transform(spheres_data)


def test_nifti_spheres_masker_io_shapes(rng, shape_3d_default, affine_eye):
    """Ensure that NiftiSpheresMasker handles 1D/2D/3D/4D data appropriately.

    transform(4D image) --> 2D output, no warning
    transform(3D image) --> 2D output, DeprecationWarning
    inverse_transform(2D array) --> 4D image, no warning
    inverse_transform(1D array) --> 3D image, no warning
    inverse_transform(2D array with wrong shape) --> ValueError
    """
    n_regions, n_volumes = 2, 5
    shape_4d = (*shape_3d_default, n_volumes)

    img_4d, mask_img = data_gen.generate_random_img(
        shape_4d,
        affine=affine_eye,
    )
    img_3d, _ = data_gen.generate_random_img(
        shape_3d_default, affine=affine_eye
    )

    masker = NiftiSpheresMasker(
        [(1, 1, 1), (4, 4, 4)],  # number of tuples equal to n_regions
        radius=1,
        mask_img=mask_img,
    )
    masker.fit()

    # DeprecationWarning *should* be raised for 3D inputs
    with pytest.deprecated_call(match="Starting in version 0.12"):
        test_data = masker.transform(img_3d)
        assert test_data.shape == (1, n_regions)

    # DeprecationWarning should *not* be raised for 4D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_data = masker.transform(img_4d)
        assert test_data.shape == (n_volumes, n_regions)

    data_1d = rng.random(n_regions)
    data_2d = rng.random((n_volumes, n_regions))
    # DeprecationWarning should *not* be raised for 1D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_1d)
        assert test_img.shape == shape_3d_default

    # DeprecationWarning should *not* be raised for 2D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_2d)
        assert test_img.shape == shape_4d

    with pytest.raises(ValueError):
        masker.inverse_transform(data_2d.T)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_nifti_spheres_masker_reporting_mpl_warning():
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        result = NiftiSpheresMasker([(1, 1, 1)]).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
    assert result == [None]
