import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy import linalg

from nilearn.conftest import _rng
from nilearn.decomposition._base import _fast_svd, _mask_and_reduce
from nilearn.decomposition.tests.conftest import (
    N_SAMPLES,
    N_SUBJECTS,
    RANDOM_STATE,
)
from nilearn.tests.test_base import TestCacheMixin, TestNilearnBaseEstimator


class TestBaseDecomposition(TestCacheMixin, TestNilearnBaseEstimator):
    @classmethod
    def expected_failed_checks(cls) -> dict[str, str]:
        expected = TestNilearnBaseEstimator.expected_failed_checks()
        expected |= {
            "check_dict_unchanged": (
                "replaced by check_img_estimator_dict_unchanged"
            ),
            "check_dont_overwrite_parameters": (
                "replaced by check_img_estimator_dont_overwrite_parameters"
            ),
            "check_estimators_dtypes": "replaced by check_img_estimator_dtypes",
            "check_estimators_empty_data_messages": (
                "replaced by check_*_empty_data_messages "
            ),
            "check_estimators_fit_returns_self": (
                "replaced by check_fit_returns_self"
            ),
            "check_estimators_overwrite_params": (
                "replaced by check_img_estimator_overwrite_params"
            ),
            "check_estimators_pickle": "replaced by check_img_estimator_pickle",
            "check_fit_check_is_fitted": (
                "replaced by check_img_estimator_fit_check_is_fitted"
            ),
            "check_fit_idempotent": (
                "replaced by check_img_estimator_fit_idempotent"
            ),
            "check_fit_score_takes_y": (
                "replaced by check_img_estimator_fit_score_takes_y"
            ),
            "check_methods_sample_order_invariance": (
                "replaced by check_nilearn_methods_sample_order_invariance"
            ),
            "check_n_features_in": "replaced by check_img_estimator_n_elements",
            "check_n_features_in_after_fitting": (
                "replaced by check_img_estimator_n_elements"
            ),
            "check_pipeline_consistency": (
                "replaced by check_img_estimator_pipeline_consistency"
            ),
            "check_estimators_nan_inf": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_positive_only_tag_during_fit": "TODO",
            "check_readonly_memmap_input": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
            "check_complex_data": "not applicable for image input",
            "check_dtype_object": "not applicable for image input",
            "check_estimator_sparse_array": "not applicable for image input",
            "check_estimator_sparse_data": "not applicable for image input",
            "check_estimator_sparse_matrix": "not applicable for image input",
            "check_f_contiguous_array_estimator": "not applicable for image input",
            "check_fit1d": "not applicable for image input",
            "check_fit2d_1feature": "not applicable for image input",
            "check_fit2d_1sample": "not applicable for image input",
            "check_fit2d_predict1d": "not applicable for image input",
        }
        return expected


# We need to use n_features > 500 to trigger the randomized_svd
@pytest.mark.parametrize("n_features", [30, 100, 550])
def test_fast_svd(n_features):
    """Test fast singular value decomposition."""
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


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize(
    "n_components,reduction_ratio,expected_shape_0",
    [
        (None, "auto", N_SUBJECTS * N_SAMPLES),
        (3, "auto", N_SUBJECTS * 3),
        (None, 0.4, N_SUBJECTS * 2),
    ],
)
def test_mask_reducer_multiple_image(
    data_type,
    n_components,
    reduction_ratio,
    expected_shape_0,
    decomposition_masker,
    decomposition_images,
):
    """Mask and reduce images with several values of input arguments."""
    data = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=decomposition_images,
        n_components=n_components,
        confounds=[None] * len(decomposition_images),
        reduction_ratio=reduction_ratio,
    )
    if data_type == "nifti":
        expected_shape = (
            expected_shape_0,
            np.prod(decomposition_masker.mask_img_.shape),
        )
    elif data_type == "surface":
        expected_shape = (
            expected_shape_0,
            decomposition_images[0].mesh.n_vertices,
        )

    assert data.shape == expected_shape


@pytest.mark.thread_unsafe
@pytest.mark.single_process
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_single_image_same_with_multiple_jobs(
    data_type, decomposition_masker, decomposition_img
):
    """Mask and reduce a 3D image and check results is the same \
    when split over several CPUs.
    """
    n_components = 3
    data_single = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=decomposition_img,
        n_components=n_components,
        confounds=[None],
    )
    if data_type == "nifti":
        assert data_single.shape == (
            n_components,
            np.prod(decomposition_masker.mask_img_.shape),
        )
    elif data_type == "surface":
        # For surface images, the shape is (n_components, n_vertices)
        assert data_single.shape == (
            n_components,
            decomposition_masker.mask_img_.shape[0],
        )

    # Test n_jobs > 1
    data = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=decomposition_img,
        n_components=n_components,
        confounds=[None],
        n_jobs=2,
        random_state=RANDOM_STATE,
    )
    if data_type == "nifti":
        assert data.shape == (
            n_components,
            np.prod(decomposition_masker.mask_img_.shape),
        )
    elif data_type == "surface":
        assert data.shape == (
            n_components,
            decomposition_masker.mask_img_.shape[0],
        )
    assert_array_almost_equal(data_single, data)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_reduced_data_is_orthogonal(
    data_type, decomposition_masker, decomposition_img
):
    """Test that the reduced data is orthogonal."""
    n_components = 3
    data = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=decomposition_img,
        n_components=n_components,
        confounds=[None],
        random_state=RANDOM_STATE,
    )

    if data_type == "nifti":
        assert data.shape == (
            n_components,
            np.prod(decomposition_masker.mask_img_.shape),
        )
    elif data_type == "surface":
        assert data.shape == (
            n_components,
            decomposition_masker.mask_img_.shape[-1],
        )

    cov = data.dot(data.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]

    assert_array_almost_equal(cov, cov_diag)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_mask_reducer_reduced_reproducible(
    data_type,
    decomposition_masker,
    decomposition_img,
):
    """Check that same input image give same results."""
    n_components = 3
    data1 = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=decomposition_img,
        n_components=n_components,
        confounds=[None],
        random_state=RANDOM_STATE,
    )
    data2 = _mask_and_reduce(
        masker=decomposition_masker,
        imgs=[decomposition_img] * 2,
        n_components=n_components,
        confounds=[None] * 2,
        random_state=RANDOM_STATE,
    )

    if data_type == "nifti":
        assert data1.shape == (
            n_components,
            np.prod(decomposition_masker.mask_img_.shape),
        )
    elif data_type == "surface":
        assert data1.shape == (
            n_components,
            decomposition_masker.mask_img_.shape[-1],
        )

    assert_array_almost_equal(np.tile(data1, (2, 1)), data2)
