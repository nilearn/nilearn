import nibabel
import numpy as np
import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nilearn.input_data import NiftiSpheresMasker
from nilearn.image import get_data, new_img_like


def test_seed_extraction():
    data = np.random.RandomState(42).random_sample((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)])
    # Test the fit
    masker.fit()
    # Test the transform
    s = masker.transform(img)
    assert_array_equal(s[:, 0], data[1, 1, 1])


def test_sphere_extraction():
    data = np.random.RandomState(42).random_sample((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)
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
    mask_img = nibabel.Nifti1Image(mask_img, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)
    assert_array_equal(s[:, 0],
                       np.mean(data[np.logical_and(mask, get_data(mask_img))],
                               axis=0))


def test_anisotropic_sphere_extraction():
    data = np.random.RandomState(42).random_sample((3, 3, 3, 5))
    affine = np.eye(4)
    affine[0, 0] = 2
    affine[2, 2] = 2
    img = nibabel.Nifti1Image(data, affine)
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
    mask_img = nibabel.Nifti1Image(mask_img, affine=affine_2)
    masker = NiftiSpheresMasker([(2, 1, 2)], radius=1, mask_img=mask_img)

    masker.fit()
    s = masker.transform(img)
    assert_array_equal(s[:, 0], data[1, 0, 1])


def test_errors():
    masker = NiftiSpheresMasker(([1, 2]), radius=.2)
    with pytest.raises(ValueError, match='Seeds must be a list .+'):
        masker.fit()


def test_nifti_spheres_masker_overlap():
    # Test resampling in NiftiMapsMasker
    affine = np.eye(4)
    shape = (5, 5, 5)

    data = np.random.RandomState(42).random_sample(shape + (5,))
    fmri_img = nibabel.Nifti1Image(data, affine)

    seeds = [(0, 0, 0), (2, 2, 2)]

    overlapping_masker = NiftiSpheresMasker(seeds, radius=1,
                                            allow_overlap=True)
    overlapping_masker.fit_transform(fmri_img)
    overlapping_masker = NiftiSpheresMasker(seeds, radius=2,
                                            allow_overlap=True)
    overlapping_masker.fit_transform(fmri_img)

    noverlapping_masker = NiftiSpheresMasker(seeds, radius=1,
                                             allow_overlap=False)
    noverlapping_masker.fit_transform(fmri_img)
    noverlapping_masker = NiftiSpheresMasker(seeds, radius=2,
                                             allow_overlap=False)
    with pytest.raises(ValueError, match='Overlap detected'):
        noverlapping_masker.fit_transform(fmri_img)


def test_small_radius():
    affine = np.eye(4)
    shape = (3, 3, 3)

    data = np.random.RandomState(42).random_sample(shape)
    mask = np.zeros(shape)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    affine = np.eye(4) * 1.2
    seed = (1.4, 1.4, 1.4)

    masker = NiftiSpheresMasker([seed], radius=0.1,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    masker.fit_transform(nibabel.Nifti1Image(data, affine))

    # Test if masking is taken into account
    mask[1, 1, 1] = 0
    mask[1, 1, 0] = 1

    masker = NiftiSpheresMasker([seed], radius=0.1,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    with pytest.raises(ValueError, match='These spheres are empty'):
        masker.fit_transform(nibabel.Nifti1Image(data, affine))

    masker = NiftiSpheresMasker([seed], radius=1.6,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    masker.fit_transform(nibabel.Nifti1Image(data, affine))


def test_is_nifti_spheres_masker_give_nans():
    affine = np.eye(4)

    data_with_nans = np.zeros((10, 10, 10), dtype=np.float32)
    data_with_nans[:, :, :] = np.nan

    data_without_nans = np.random.RandomState(42).random_sample((9, 9, 9))
    indices = np.nonzero(data_without_nans)

    # Leaving nans outside of some data
    data_with_nans[indices] = data_without_nans[indices]
    img = nibabel.Nifti1Image(data_with_nans, affine)
    seed = [(7, 7, 7)]

    # Interaction of seed with nans
    masker = NiftiSpheresMasker(seeds=seed, radius=2.)
    assert not np.isnan(np.sum(masker.fit_transform(img)))

    mask = np.ones((9, 9, 9))
    mask_img = nibabel.Nifti1Image(mask, affine)
    # When mask_img is provided, the seed interacts within the brain, so no nan
    masker = NiftiSpheresMasker(seeds=seed, radius=2., mask_img=mask_img)
    assert not np.isnan(np.sum(masker.fit_transform(img)))


def test_standardization():
    data = np.random.RandomState(42).random_sample((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))

    # test zscore
    masker = NiftiSpheresMasker([(1, 1, 1)], standardize='zscore')
    # Test the fit
    s = masker.fit_transform(img)

    np.testing.assert_almost_equal(s.mean(), 0)
    np.testing.assert_almost_equal(s.std(), 1)

    # test psc
    masker = NiftiSpheresMasker([(1, 1, 1)], standardize='psc')
    # Test the fit
    s = masker.fit_transform(img)

    np.testing.assert_almost_equal(s.mean(), 0)
    np.testing.assert_almost_equal(s.ravel(), data[1, 1, 1] /
                                   data[1, 1, 1].mean() * 100 - 100,
                                   )


def test_nifti_spheres_masker_inverse_transform():
    # Applying the sphere_extraction example from above backwards
    data = np.random.RandomState(42).random_sample((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)
    # Test the fit
    masker.fit()
    # Transform data
    with pytest.raises(ValueError, match='Please provide mask_img'):
        masker.inverse_transform(data[0, 0, 0, :])

    # Mask describes the extend of the masker's sphere
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 1, 1] = True
    mask[1, :, 1] = True
    mask[1, 1, :] = True

    # Now with a mask
    mask_img = np.zeros((3, 3, 3))
    mask_img[1, :, :] = 1
    mask_img = nibabel.Nifti1Image(mask_img, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1, mask_img=mask_img)
    masker.fit()
    s = masker.transform(img)
    # Create an array mask
    array_mask = np.logical_and(mask, get_data(mask_img))

    inverse_map = masker.inverse_transform(s)

    # Testing whether mask is applied to inverse transform
    assert_array_equal(np.mean(get_data(inverse_map), axis=-1) != 0,
                       array_mask)
    # Test whether values are preserved
    assert_array_equal(get_data(inverse_map)[array_mask].mean(0), s[:, 0])

    # Test whether the mask's shape is applied
    assert_array_equal(inverse_map.shape[:3], mask_img.shape)


def test_nifti_spheres_masker_inverse_overlap():
    rng = np.random.RandomState(42)

    # Test overlapping data in inverse_transform
    affine = np.eye(4)
    shape = (5, 5, 5)

    data = rng.random_sample(shape + (5,))
    fmri_img = nibabel.Nifti1Image(data, affine)

    # Apply mask image - to allow inversion
    mask_img = new_img_like(fmri_img, np.ones(shape))
    seeds = [(0, 0, 0), (2, 2, 2)]
    # Inverse data
    inv_data = rng.random_sample(len(seeds))

    overlapping_masker = NiftiSpheresMasker(seeds, radius=1,
                                            allow_overlap=True,
                                            mask_img=mask_img).fit()
    overlapping_masker.inverse_transform(inv_data)

    overlapping_masker = NiftiSpheresMasker(seeds, radius=2,
                                            allow_overlap=True,
                                            mask_img=mask_img).fit()

    overlap = overlapping_masker.inverse_transform(inv_data)

    # Test whether overlapping data is averaged
    assert_array_almost_equal(get_data(overlap)[1, 1, 1], np.mean(inv_data))

    noverlapping_masker = NiftiSpheresMasker(seeds, radius=1,
                                             allow_overlap=False,
                                             mask_img=mask_img).fit()

    noverlapping_masker.inverse_transform(inv_data)
    noverlapping_masker = NiftiSpheresMasker(seeds, radius=2,
                                             allow_overlap=False,
                                             mask_img=mask_img).fit()

    with pytest.raises(ValueError, match='Overlap detected'):
        noverlapping_masker.inverse_transform(inv_data)


def test_small_radius_inverse():
    affine = np.eye(4)
    shape = (3, 3, 3)

    data = np.random.RandomState(42).random_sample(shape)
    mask = np.zeros(shape)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    affine = np.eye(4) * 1.2
    seed = (1.4, 1.4, 1.4)

    masker = NiftiSpheresMasker([seed], radius=0.1,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    spheres_data = masker.fit_transform(nibabel.Nifti1Image(data, affine))
    masker.inverse_transform(spheres_data)
    # Test if masking is taken into account
    mask[1, 1, 1] = 0
    mask[1, 1, 0] = 1

    masker = NiftiSpheresMasker([seed], radius=0.1,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    masker.fit(nibabel.Nifti1Image(data, affine))

    with pytest.raises(ValueError, match='These spheres are empty'):
        masker.inverse_transform(spheres_data)

    masker = NiftiSpheresMasker([seed], radius=1.6,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    masker.fit(nibabel.Nifti1Image(data, affine))
    masker.inverse_transform(spheres_data)
