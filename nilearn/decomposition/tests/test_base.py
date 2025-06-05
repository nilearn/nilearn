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

SHAPE_NIFTI = (6, 8, 10)
SHAPE_SURF = {"left": (10, 8), "right": (9, 7)}
N_SUBJECTS = 4
N_SAMPLES = 5

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


def make_data_to_reduce(data_type="nifti", with_activation=True):
    """Create "multi-subject" dataset with fake activation."""
    rng = _rng()
    imgs = []

    if data_type == "surface":
        mesh = {
            "left": flat_mesh(*SHAPE_SURF["left"]),
            "right": flat_mesh(*SHAPE_SURF["right"]),
        }
        for _ in range(N_SUBJECTS):
            data = {
                "left": rng.standard_normal(
                    size=(mesh["left"].coordinates.shape[0], N_SAMPLES)
                ),
                "right": rng.standard_normal(
                    size=(mesh["right"].coordinates.shape[0], N_SAMPLES)
                ),
            }
            if with_activation:
                data["left"][2:4, :] += 10
                data["right"][2:4, :] += 10
            imgs.append(SurfaceImage(mesh=mesh, data=data))
        mask_data = {
            "left": np.ones((mesh["left"].coordinates.shape[0],)),
            "right": np.ones((mesh["right"].coordinates.shape[0],)),
        }
        mask_img = SurfaceImage(mesh=mesh, data=mask_data)
        return imgs, mask_img

    shape = (*SHAPE_NIFTI, N_SAMPLES)
    for _ in range(N_SUBJECTS):
        this_img = rng.normal(size=shape)
        if with_activation:
            this_img[2:4, 2:4, 2:4, :] += 10
        imgs.append(Nifti1Image(this_img, _affine_eye()))
    mask_img = Nifti1Image(np.ones(shape[:3], dtype=np.int8), _affine_eye())
    return imgs, mask_img


def make_masker(data_type="nifti"):
    if data_type == "surface":
        mask_img = make_data_to_reduce(data_type=data_type)[1]
        return SurfaceMasker(mask_img=mask_img).fit()
    return MultiNiftiMasker(mask_img=_img_3d_ones()).fit()


# We need to use n_features > 500 to trigger the randomized_svd
@pytest.mark.parametrize("n_features", [30, 100, 550])
def test_fast_svd(n_features):
    n_samples = 100
    k = 10

    # generate a matrix X of approximate effective rank `rank` and no noise
    # component (very structured signal):
    U = _rng().normal(size=(n_samples, k))
    V = _rng().normal(size=(k, n_features))
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
        (None, "auto", 4 * 5),
        (3, "auto", 4 * 3),
        (None, 0.4, 4 * 2),
    ],
)
def test_mask_reducer_multiple_image(
    data_type,
    n_components,
    reduction_ratio,
    expected_shape_0,
    shape_3d_default,
):
    """Mask and reduce 4D images with several values of input arguments."""
    imgs, _ = make_data_to_reduce(data_type=data_type)
    data = _mask_and_reduce(
        masker=make_masker(data_type=data_type),
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
    shape_3d_default, data_type
):
    """Mask and reduce a 3D image and check results is the same \
    when split over several CPUs.
    """
    img = make_data_to_reduce(data_type=data_type)[0][0]
    data_single = _mask_and_reduce(
        masker=make_masker(data_type=data_type),
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
        masker=make_masker(data_type=data_type),
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
def test_mask_reducer_reduced_data_is_orthogonal(shape_3d_default, data_type):
    """Test that the reduced data is orthogonal."""
    img = make_data_to_reduce(data_type=data_type)[0][0]
    data = _mask_and_reduce(
        masker=make_masker(data_type=data_type),
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
def test_mask_reducer_reduced_reproducible(data_type):
    img = make_data_to_reduce(data_type=data_type)[0][0]
    data1 = _mask_and_reduce(
        masker=make_masker(data_type=data_type),
        imgs=img,
        n_components=3,
        random_state=0,
    )
    data2 = _mask_and_reduce(
        masker=make_masker(data_type=data_type),
        imgs=[img] * 2,
        n_components=3,
        random_state=0,
    )

    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)
