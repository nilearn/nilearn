import nibabel
import numpy as np
import pytest
from nilearn.decomposition._base import _fast_svd, _mask_and_reduce
from nilearn.maskers import MultiNiftiMasker
from numpy.testing import assert_array_almost_equal
from scipy import linalg


def test_fast_svd():
    n_samples = 100
    k = 10

    rng = np.random.RandomState(42)

    # We need to use n_features > 500 to trigger the randomized_svd
    for n_features in (30, 100, 550):
        # generate a matrix X of approximate effective rank `rank` and no noise
        # component (very structured signal):
        U = rng.normal(size=(n_samples, k))
        V = rng.normal(size=(k, n_features))
        X = np.dot(U, V)
        assert X.shape == (n_samples, n_features)

        # compute the singular values of X using the slow exact method
        U_, s_, V_ = linalg.svd(X, full_matrices=False)

        Ur, Sr, Vr = _fast_svd(X, k, random_state=0)
        assert Vr.shape == (k, n_features)
        assert Ur.shape == (n_samples, k)

        # check the singular vectors too (while not checking the sign)
        assert_array_almost_equal(
            np.abs(np.diag(np.corrcoef(V_[:k], Vr)))[:k], np.ones(k)
        )


def _make_mask_reduce_test_data():
    """Create "multi-subject" dataset with fake activation \
    to get non empty mask"""
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    imgs = []
    for _ in range(8):
        this_img = rng.normal(size=shape)

        # Add activation
        this_img[2:4, 2:4, 2:4, :] += 10

        imgs.append(nibabel.Nifti1Image(this_img, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    masker = MultiNiftiMasker(mask_img=mask_img).fit()

    return masker, imgs


@pytest.mark.parametrize(
    "n_components,reduction_ratio,expected_shape_0",
    [
        (None, "auto", 8 * 5),
        (3, "auto", 8 * 3),
        (None, 0.4, 8 * 2),
    ],
)
def test_mask_reducer_multiple_image(
    n_components, reduction_ratio, expected_shape_0
):
    """Mask and reduce 4D images with several values of input arguments."""
    masker, imgs = _make_mask_reduce_test_data()

    data = _mask_and_reduce(
        masker=masker,
        imgs=imgs,
        n_components=n_components,
        reduction_ratio=reduction_ratio,
    )

    expected_shape = (expected_shape_0, 6 * 8 * 10)
    assert data.shape == expected_shape


def test_mask_reducer_single_image_same_with_multiple_jobs():
    """Mask and reduce a 3D image and check results is the same \
    when split over several CPUs."""
    masker, imgs = _make_mask_reduce_test_data()

    data_single = _mask_and_reduce(masker, imgs[0], n_components=3)

    assert data_single.shape == (3, 6 * 8 * 10)

    # Test n_jobs > 1
    data = _mask_and_reduce(
        masker, imgs[0], n_components=3, n_jobs=2, random_state=0
    )

    assert data.shape == (3, 6 * 8 * 10)

    assert_array_almost_equal(data_single, data)


def test_mask_reducer_reduced_data_is_orthogonal():
    """# Test that the reduced data is orthogonal."""
    masker, imgs = _make_mask_reduce_test_data()

    data = _mask_and_reduce(masker, imgs[0], n_components=3, random_state=0)

    assert data.shape == (3, 6 * 8 * 10)

    cov = data.dot(data.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]

    assert_array_almost_equal(cov, cov_diag)


def test_mask_reducer_reduced_reproducible():
    masker, imgs = _make_mask_reduce_test_data()

    data1 = _mask_and_reduce(masker, imgs[0], n_components=3, random_state=0)
    data2 = _mask_and_reduce(
        masker, [imgs[0]] * 2, n_components=3, random_state=0
    )

    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)
