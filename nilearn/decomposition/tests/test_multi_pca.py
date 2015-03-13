"""
Test the multi-PCA module
"""
import nibabel
import numpy as np

import nose

from nilearn.input_data import MultiNiftiMasker
from nilearn.decomposition.multi_pca import MultiPCA


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
    multi_pca = MultiPCA(mask=mask_img, n_components=3)

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
    nose.tools.assert_raises(ValueError, multi_pca.fit, data[:2])

    # Smoke test the use of a masker and without CCA
    multi_pca = MultiPCA(mask=MultiNiftiMasker(mask_args=dict(opening=0)),
                         do_cca=False, n_components=3)
    multi_pca.fit(data[:2])

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(data[-2:]))
