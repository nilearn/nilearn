import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal
from scipy import linalg
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _affine_eye, _img_3d_ones, _rng
from nilearn.decomposition._base import (
    _BaseDecomposition,
    _fast_svd,
    _mask_and_reduce,
)
from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
from nilearn.surface import SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

ESTIMATORS_TO_CHECK = [_BaseDecomposition()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def data_for_mask_and_reduce(rng, data_type="nifti"):
    """Create "multi-subject" dataset with fake activation."""
    n_samples = 5
    n_subjects = 8
    imgs = []

    if data_type == "nifti":
        shape = (6, 8, 10, n_samples)

        for _ in range(n_subjects):
            this_img = _rng().normal(size=shape)

            # Add activation
            this_img[2:4, 2:4, 2:4, :] += 10

            imgs.append(Nifti1Image(this_img, _affine_eye()))
    elif data_type == "surface":
        mesh = {
            "left": flat_mesh(10, 8),
            "right": flat_mesh(9, 7),
        }
        for _ in range(n_subjects):
            data = {
                "left": rng.standard_normal(
                    size=(mesh["left"].coordinates.shape[0], n_samples)
                ),
                "right": rng.standard_normal(
                    size=(mesh["right"].coordinates.shape[0], n_samples)
                ),
            }
            data["left"][2:4, :] += 10
            data["right"][2:4, :] += 10
            imgs.append(SurfaceImage(mesh=mesh, data=data))
    return imgs


def create_masker(data_type="nifti"):
    if data_type == "nifti":
        return MultiNiftiMasker(mask_img=_img_3d_ones()).fit()
    elif data_type == "surface":
        mesh = {
            "left": flat_mesh(10, 8),
            "right": flat_mesh(9, 7),
        }
        data = {
            "left": np.ones((mesh["left"].coordinates.shape[0],)),
            "right": np.ones((mesh["right"].coordinates.shape[0],)),
        }
        return SurfaceMasker(SurfaceImage(mesh=mesh, data=data)).fit()


# We need to use n_features > 500 to trigger the randomized_svd
@pytest.mark.parametrize("n_features", [30, 100, 550])
def test_fast_svd(n_features, rng):
    n_samples = 100
    k = 10

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


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize(
    "n_components,reduction_ratio,expected_shape_0",
    [
        (None, "auto", 8 * 5),
        (3, "auto", 8 * 3),
        (None, 0.4, 8 * 2),
    ],
)
def test_mask_reducer_multiple_image(
    data_type,
    rng,
    n_components,
    reduction_ratio,
    expected_shape_0,
    shape_3d_default,
):
    """Mask and reduce 4D images with several values of input arguments."""
    imgs = data_for_mask_and_reduce(rng, data_type=data_type)
    data = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=imgs,
        n_components=n_components,
        reduction_ratio=reduction_ratio,
    )
    if data_type == "nifti":
        expected_shape = (expected_shape_0, np.prod(shape_3d_default))
    elif data_type == "surface":
        expected_shape = (expected_shape_0, imgs[0].mesh.n_vertices)

    assert data.shape == expected_shape


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_single_image_same_with_multiple_jobs(
    rng, shape_3d_default, data_type
):
    """Mask and reduce a 3D image and check results is the same \
    when split over several CPUs.
    """
    img = data_for_mask_and_reduce(rng, data_type=data_type)[0]
    data_single = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=img,
        n_components=3,
    )
    if data_type == "nifti":
        assert data_single.shape == (3, np.prod(shape_3d_default))
    elif data_type == "surface":
        # For surface images, the shape is (n_components, n_vertices)
        assert data_single.shape == (3, img.mesh.n_vertices)

    # Test n_jobs > 1
    data = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=img,
        n_components=3,
        n_jobs=2,
        random_state=0,
    )
    if data_type == "nifti":
        assert data.shape == (3, np.prod(shape_3d_default))
    elif data_type == "surface":
        assert data.shape == (3, img.mesh.n_vertices)
    assert_array_almost_equal(data_single, data)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_reduced_data_is_orthogonal(
    rng, shape_3d_default, data_type
):
    """Test that the reduced data is orthogonal."""
    img = data_for_mask_and_reduce(rng, data_type=data_type)[0]
    data = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=img,
        n_components=3,
        random_state=0,
    )

    if data_type == "nifti":
        assert data.shape == (3, np.prod(shape_3d_default))
    elif data_type == "surface":
        assert data.shape == (3, img.mesh.n_vertices)

    cov = data.dot(data.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]

    assert_array_almost_equal(cov, cov_diag)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_reduced_reproducible(rng, data_type):
    img = data_for_mask_and_reduce(rng, data_type=data_type)[0]
    data1 = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=img,
        n_components=3,
        random_state=0,
    )
    data2 = _mask_and_reduce(
        masker=create_masker(data_type=data_type),
        imgs=[img] * 2,
        n_components=3,
        random_state=0,
    )

    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)
