import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn.decomposition._base import _fast_svd, _mask_and_reduce
from nilearn.maskers import MultiNiftiMasker
from numpy.testing import assert_array_almost_equal
from scipy import linalg

AFFINE_EYE = np.eye(4)

SHAPE = (6, 8, 10)


@pytest.fixture
def data_for_mask_and_reduce():
    """Create "multi-subject" dataset with fake activation."""
    shape = (6, 8, 10, 5)
    rng = np.random.RandomState(0)

    imgs = []
    for _ in range(8):
        this_img = rng.normal(size=shape)

        # Add activation
        this_img[2:4, 2:4, 2:4, :] += 10

        imgs.append(Nifti1Image(this_img, AFFINE_EYE))

    return imgs


@pytest.fixture
def masker():
    mask_img = Nifti1Image(np.ones(SHAPE, dtype=np.int8), AFFINE_EYE)
    return MultiNiftiMasker(mask_img=mask_img).fit()


# We need to use n_features > 500 to trigger the randomized_svd
@pytest.mark.parametrize("n_features", [30, 100, 550])
def test_fast_svd(n_features):
    n_samples = 100
    k = 10

    rng = np.random.RandomState(42)

    # generate a matrix X of approximate effective rank `rank` and no noise
    # component (very structured signal):
    U = rng.normal(size=(n_samples, k))
    V = rng.normal(size=(k, n_features))
    X = np.dot(U, V)

    assert X.shape == (n_samples, n_features)

    # compute the singular values of X using the slow exact method
    _, _, V_ = linalg.svd(X, full_matrices=False)

    Ur, _, Vr = _fast_svd(X, k, random_state=0)

    assert Vr.shape == (k, n_features)
    assert Ur.shape == (n_samples, k)
    # check the singular vectors too (while not checking the sign)
    assert_array_almost_equal(
        np.abs(np.diag(np.corrcoef(V_[:k], Vr)))[:k], np.ones(k)
    )


@pytest.mark.parametrize(
    "n_components,reduction_ratio,expected_shape_0",
    [
        (None, "auto", 8 * 5),
        (3, "auto", 8 * 3),
        (None, 0.4, 8 * 2),
    ],
)
def test_mask_reducer_multiple_image(
    data_for_mask_and_reduce,
    masker,
    n_components,
    reduction_ratio,
    expected_shape_0,
):
    """Mask and reduce 4D images with several values of input arguments."""
    data = _mask_and_reduce(
        masker=masker,
        imgs=data_for_mask_and_reduce,
        n_components=n_components,
        reduction_ratio=reduction_ratio,
    )

    expected_shape = (expected_shape_0, 6 * 8 * 10)

    assert data.shape == expected_shape


def test_mask_reducer_single_image_same_with_multiple_jobs(
    data_for_mask_and_reduce, masker
):
    """Mask and reduce a 3D image and check results is the same \
    when split over several CPUs."""
    data_single = _mask_and_reduce(
        masker, data_for_mask_and_reduce[0], n_components=3
    )

    assert data_single.shape == (3, 6 * 8 * 10)

    # Test n_jobs > 1
    data = _mask_and_reduce(
        masker,
        data_for_mask_and_reduce[0],
        n_components=3,
        n_jobs=2,
        random_state=0,
    )

    assert data.shape == (3, 6 * 8 * 10)
    assert_array_almost_equal(data_single, data)


def test_mask_reducer_reduced_data_is_orthogonal(
    data_for_mask_and_reduce, masker
):
    """Test that the reduced data is orthogonal."""
    data = _mask_and_reduce(
        masker, data_for_mask_and_reduce[0], n_components=3, random_state=0
    )

    assert data.shape == (3, 6 * 8 * 10)

    cov = data.dot(data.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]

    assert_array_almost_equal(cov, cov_diag)


def test_mask_reducer_reduced_reproducible(data_for_mask_and_reduce, masker):
    data1 = _mask_and_reduce(
        masker, data_for_mask_and_reduce[0], n_components=3, random_state=0
    )
    data2 = _mask_and_reduce(
        masker,
        [data_for_mask_and_reduce[0]] * 2,
        n_components=3,
        random_state=0,
    )

    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)
