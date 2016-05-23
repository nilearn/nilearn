import numpy as np
from nose.tools import assert_true
import nibabel
from numpy.testing import assert_equal, assert_array_almost_equal
from nilearn._utils.testing import assert_raises_regex
from nilearn.input_data import MultiNiftiMasker
from nilearn.decomposition.base import BaseDecomposition, mask_and_reduce

def test_mask_reducer():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    # Create a "multi-subject" dataset
    imgs = []
    for i in range(8):
        this_img = rng.normal(size=shape)
        # Create fake activation to get non empty mask
        this_img[2:4, 2:4, 2:4, :] += 10
        imgs.append(nibabel.Nifti1Image(this_img, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    masker = MultiNiftiMasker(mask_img=mask_img).fit()

    # Test fit on multiple image
    data = mask_and_reduce(masker, imgs)
    assert_equal(data.shape, (8 * 5, 6 * 8 * 10))

    data = mask_and_reduce(masker, imgs, n_components=3)
    assert_equal(data.shape, (8 * 3, 6 * 8 * 10))

    data = mask_and_reduce(masker, imgs, reduction_ratio=0.4)
    assert_equal(data.shape, (8 * 2, 6 * 8 * 10))

    # Test on single image
    data_single = mask_and_reduce(masker, imgs[0], n_components=3)
    assert_true(data_single.shape == (3, 6 * 8 * 10))

    # Test n_jobs > 1
    data = mask_and_reduce(masker, imgs[0], n_components=3,
                           n_jobs=2, random_state=0)
    assert_equal(data.shape, (3, 6 * 8 * 10))
    assert_array_almost_equal(data_single, data)

    # Test that reduced data is orthogonal
    data = mask_and_reduce(masker, imgs[0], n_components=3,
                           random_state=0)
    assert_true(data.shape == (3, 6 * 8 * 10))
    cov = data.dot(data.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]
    assert_array_almost_equal(cov, cov_diag)

    # Test reproducibility
    data1 = mask_and_reduce(masker, imgs[0], n_components=3,
                                  random_state=0)
    data2 = mask_and_reduce(masker, [imgs[0]] * 2, n_components=3,
                                  random_state=0)
    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)


def test_base_decomposition():
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
    base_decomposition = BaseDecomposition(mask=masker, n_components=3)
    base_decomposition.fit(data)
    assert_true(base_decomposition.mask_img_ == mask)
    assert_true(base_decomposition.mask_img_ ==
                base_decomposition.masker_.mask_img_)

    # Testing fit on data
    masker = MultiNiftiMasker()
    base_decomposition = BaseDecomposition(mask=masker, n_components=3)
    base_decomposition.fit(data)
    assert_true(base_decomposition.mask_img_ ==
                base_decomposition.masker_.mask_img_)

    assert_raises_regex(ValueError,
                        "Object has no components_ attribute. "
                        "This may be because "
                        "BaseDecomposition is directly "
                        "being used.",
                        base_decomposition.transform, data)
    assert_raises_regex(ValueError,
                        'Need one or more Niimg-like objects as input, '
                        'an empty list was given.',
                        base_decomposition.fit, [])

    # Test passing masker arguments to estimator
    base_decomposition = BaseDecomposition(target_affine=affine,
                                           target_shape=shape[:3],
                                           n_components=3, 
                                           mask_strategy='background')
    base_decomposition.fit(data)

# Score is tested in multi_pca
