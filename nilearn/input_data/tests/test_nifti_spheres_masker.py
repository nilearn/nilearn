import nibabel
import numpy as np
from numpy.testing import assert_array_equal
from nilearn.input_data import NiftiSpheresMasker
from nilearn._utils.testing import assert_raises_regexp


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
    mask = np.asarray(
        [
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]
        ], dtype=bool
    )
    assert_array_equal(s[:, 0], np.mean(data[mask], axis=0))

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
    mask = np.asarray(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ], dtype=bool
    )
    assert_array_equal(s[:, 0], np.mean(data[mask], axis=0))


def test_errors():
    masker = NiftiSpheresMasker(([1, 2]), radius=.2)
    assert_raises_regexp(ValueError, 'Seeds must be a list .+', masker.fit)
