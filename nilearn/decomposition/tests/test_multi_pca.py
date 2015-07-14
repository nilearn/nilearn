"""
Test the multi-PCA module
"""

import numpy as np
from nose.tools import assert_raises, assert_true
import nibabel

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn.input_data import MultiNiftiMasker


def test_multi_pca():
    # Smoke test the MultiPCA
    # XXX: this is mostly a smoke test
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    # Create a "multi-subject" dataset
    data = []
    for i in range(8):
        this_data = rng.normal(size=shape)
        # Create fake activation to get non empty mask
        this_data[2:4, 2:4, 2:4, :] += 10
        data.append(nibabel.Nifti1Image(this_data, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=3)

    # Test that the components are the same if we put twice the same data
    components1 = multi_pca.fit(data).components_
    components2 = multi_pca.fit(2 * data).components_
    np.testing.assert_array_almost_equal(components1, components2)

    # Smoke test fit with 'confounds' argument
    confounds = [np.arange(10).reshape(5, 2)] * 8
    multi_pca.fit(data, confounds=confounds)

    # Smoke test that multi_pca also works with single subject data
    multi_pca.fit(data[0])

    # Check that asking for too little components raises a ValueError
    multi_pca = MultiPCA()
    assert_raises(ValueError, multi_pca.fit, data[:2])

    # Smoke test the use of a masker and without CCA
    multi_pca = MultiPCA(mask=MultiNiftiMasker(mask_args=dict(opening=0)),
                         do_cca=False, n_components=3)
    multi_pca.fit(data[:2])

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(data[-2:]))

    # Smoke test to fit with no img
    assert_raises(TypeError, multi_pca.fit)

def test_multi_pca_score():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    # Create a "multi-subject" dataset
    data = []
    for i in range(8):
        this_data = rng.normal(size=shape)
        data.append(nibabel.Nifti1Image(this_data, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)

    # Assert that score is between zero and one
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=3)
    multi_pca.fit(data)
    s = multi_pca.score(data, per_component=False)
    assert_true(np.all(s <= 1))
    assert_true(np.all(0 <= s))

    # Assert that score does not fail with single subject data
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=3)
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0], per_component=False)
    assert_true(isinstance(s, float))
    assert(0. <= s <= 1.)

    # Assert that score is one for n_components == n_sample in single subject configuration
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=5)
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0], per_component=False)
    assert(s == 1.)

    # Per component score
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=5)
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0], per_component=True)
    assert_true(np.all(s <= 1))
    assert_true(np.all(0 <= s))

    # Per component score, multisubject
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0, n_components=5)
    multi_pca.fit(data)
    s = multi_pca.score(data, per_component=True)
    assert_true(s.shape == (8, 5))
    assert_true(np.all(s <= 1))
    assert_true(np.all(0 <= s))





