import nibabel
import numpy as np
from numpy.testing import assert_array_equal
from nilearn.input_data import NiftiSpheresMasker
from nilearn._utils.testing import assert_raises_regex


def test_seed_extraction():
    data = np.random.random((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)])
    # Test the fit
    masker.fit()
    # Test the transform
    s = masker.transform(img)
    assert_array_equal(s[:, 0], data[1, 1, 1])


def test_sphere_extraction():
    data = np.random.random((3, 3, 3, 5))
    img = nibabel.Nifti1Image(data, np.eye(4))
    masker = NiftiSpheresMasker([(1, 1, 1)], radius=1)
    # Test the fit
    masker.fit()
    # Test the transform
    s = masker.transform(img)
    mask = np.zeros((3, 3, 3), dtype=np.bool)
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
                       np.mean(data[np.logical_and(mask, mask_img.get_data())],
                               axis=0))


def test_anisotropic_sphere_extraction():
    data = np.random.random((3, 3, 3, 5))
    affine = np.eye(4)
    affine[0, 0] = 2
    affine[2, 2] = 2
    img = nibabel.Nifti1Image(data, affine)
    masker = NiftiSpheresMasker([(2, 1, 2)], radius=1)
    # Test the fit
    masker.fit()
    # Test the transform
    s = masker.transform(img)
    mask = np.zeros((3, 3, 3), dtype=np.bool)
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
    assert_raises_regex(ValueError, 'Seeds must be a list .+', masker.fit)


def test_nifti_spheres_masker_overlap():
    # Test resampling in NiftiMapsMasker
    affine = np.eye(4)
    shape = (5, 5, 5)

    data = np.random.random(shape + (5,))
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
    assert_raises_regex(ValueError, 'Overlap detected',
                        noverlapping_masker.fit_transform, fmri_img)


def test_small_radius():
    affine = np.eye(4)
    shape = (3, 3, 3)

    data = np.random.random(shape)
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
    assert_raises_regex(ValueError, 'Sphere around seed #0 is empty',
                        masker.fit_transform,
                        nibabel.Nifti1Image(data, affine))

    masker = NiftiSpheresMasker([seed], radius=1.6,
                                mask_img=nibabel.Nifti1Image(mask, affine))
    masker.fit_transform(nibabel.Nifti1Image(data, affine))
