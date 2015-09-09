import os
import numpy as np
from nose.tools import assert_true, assert_false, assert_is
import nibabel
from numpy.testing import assert_equal, assert_array_almost_equal

from nilearn.datasets.utils import _get_dataset_dir
from nilearn._utils.testing import assert_raises_regex
from nilearn.input_data import MultiNiftiMasker
from nilearn.decomposition.base import DecompositionEstimator, mask_and_reduce


def test_mask_and_reduce():
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
    masker = MultiNiftiMasker(mask_img=mask_img).fit()

    # Test fit on multiple image
    with mask_and_reduce(masker, data) as components:
        assert_equal(components.shape, (8 * 5, 6 * 8 * 10))

    with mask_and_reduce(masker, data, n_components=3) as components:
        assert_equal(components.shape, (8 * 3, 6 * 8 * 10))

    with mask_and_reduce(masker, data, reduction_ratio=0.4) as components:
        assert_equal(components.shape, (8 * 2, 6 * 8 * 10))

    # Test on single image
    with mask_and_reduce(masker, data[0]) as components:
        assert_true(components.shape == (5, 6 * 8 * 10))

    # Test that reduced data is orthogonal
    with mask_and_reduce(masker, data[0], n_components=3) as components:
        assert_true(components.shape == (3, 6 * 8 * 10))
    cov = components.dot(components.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]
    assert_array_almost_equal(cov, cov_diag)

    # Test memorymap
    with mask_and_reduce(masker, data[0], n_components=3,
                         max_nbytes=0) as components:
        assert_equal(components.shape, (3, 6 * 8 * 10))
        temp_file = components.filename
        assert_true(os.path.exists(os.path.join(temp_file)))
    # Assert that temp file removal has worked
    assert_false(os.path.exists(temp_file))

    # Test mock
    with mask_and_reduce(masker, data[0], n_components=3,
                         max_nbytes=0,
                         n_jobs=1,
                         mock=True) as components:
        assert_is(components, None)
        # Should test cache

    # Test n_jobs > 1 with memory map
    with mask_and_reduce(masker, data[0], n_components=3,
                         max_nbytes=0,
                         n_jobs=2) as components:
        assert_equal(components.shape, (3, 6 * 8 * 10))
        temp_file = components.filename
        assert_true(os.path.exists(os.path.join(temp_file)))
    # Assert that temp file removal has worked
    assert_false(os.path.exists(temp_file))
    assert_false(os.path.exists(os.path.join(temp_file, os.path.pardir)))

    # Test n_jobs > 1 with array
    with mask_and_reduce(masker, data[0], n_components=3,
                         max_nbytes=None,
                         n_jobs=2) as components:
        assert_equal(components.shape, (3, 6 * 8 * 10))
        assert_true(isinstance(components, np.ndarray))
        # Should assert that temp file removal has worked



def test_decomposition_estimator():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data = []
    for i in range(8):
        this_data = rng.normal(size=shape)
        # Create fake activation to get non empty mask
        this_data[2:4, 2:4, 2:4, :] += 10
        data.append(nibabel.Nifti1Image(this_data, affine))
    mask = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    masker = MultiNiftiMasker(mask_img=mask)
    decomposition_estimator = DecompositionEstimator(mask=masker,
                                                     n_components=3)
    decomposition_estimator.fit(data)
    assert_true(decomposition_estimator.mask_img_ == mask)
    assert_true(decomposition_estimator.mask_img_ ==
           decomposition_estimator.masker_.mask_img_)

    # Testing fit on data
    masker = MultiNiftiMasker()
    decomposition_estimator = DecompositionEstimator(mask=masker,
                                                     n_components=3)
    decomposition_estimator.fit(data)
    assert_true(decomposition_estimator.mask_img_ ==
           decomposition_estimator.masker_.mask_img_)

    assert_raises_regex(ValueError,
                        "Object has no components_ attribute. "
                        "This may be because "
                        "DecompositionEstimator is direclty "
                        "being used.",
                        decomposition_estimator.transform, data)
    assert_raises_regex(ValueError,
                        'Need one or more Niimg-like objects as input, '
                        'an empty list was given.',
                        decomposition_estimator.fit, [])