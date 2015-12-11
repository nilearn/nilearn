"""
Test the multi-PCA module
"""

import numpy as np
from nose.tools import assert_raises, assert_true
import nibabel
from numpy.testing import assert_almost_equal, assert_equal

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn.input_data import MultiNiftiMasker, NiftiMasker
from nilearn._utils.testing import assert_raises_regex


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
    multi_pca = MultiPCA(mask=mask_img, n_components=3,
                         random_state=0)

    # Test that the components are the same if we put twice the same data, and
    # that fit output is deterministic
    components1 = multi_pca.fit(data).components_
    components2 = multi_pca.fit(data).components_
    components3 = multi_pca.fit(2 * data).components_
    np.testing.assert_array_equal(components1, components2)
    np.testing.assert_array_almost_equal(components1, components3)

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

    multi_pca = MultiPCA(mask=mask_img, n_components=3)
    assert_raises_regex(ValueError,
                        "Object has no components_ attribute. "
                        "This is probably because fit has not been called",
                        multi_pca.transform, data)



def test_multi_pca_score():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    # Create a "multi-subject" dataset
    imgs = []
    for i in range(8):
        this_img = rng.normal(size=shape)
        imgs.append(nibabel.Nifti1Image(this_img, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)

    # Assert that score is between zero and one
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0,
                         n_components=3)
    multi_pca.fit(imgs)
    s = multi_pca.score(imgs)
    assert_true(np.all(s <= 1))
    assert_true(np.all(0 <= s))

    # Assert that score does not fail with single subject data
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0,
                         n_components=3)
    multi_pca.fit(imgs[0])
    s = multi_pca.score(imgs[0])
    assert_true(isinstance(s, float))
    assert(0. <= s <= 1.)

    # Assert that score is one for n_components == n_sample
    # in single subject configuration
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0,
                         n_components=5)
    multi_pca.fit(imgs[0])
    s = multi_pca.score(imgs[0])
    assert_almost_equal(s, 1., 1)

    # Per component score
    multi_pca = MultiPCA(mask=mask_img, random_state=0, memory_level=0,
                         n_components=5)
    multi_pca.fit(imgs[0])
    masker = NiftiMasker(mask_img).fit()
    s = multi_pca._raw_score(masker.transform(imgs[0]), per_component=True)
    assert_equal(s.shape, (5,))
    assert_true(np.all(s <= 1))
    assert_true(np.all(0 <= s))
