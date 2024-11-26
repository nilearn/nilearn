import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn import __version__ as sklearn_version

from nilearn._utils import compare_version
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.input_data import NiftiMasker
from nilearn.maskers import SurfaceMasker
from nilearn.regions.hierarchical_kmeans_clustering import (
    HierarchicalKMeans,
    _adjust_small_clusters,
    hierarchical_k_means,
)

extra_valid_checks = [
    "check_clusterer_compute_labels_predict",
    "check_clustering",
    "check_complex_data",
    "check_dict_unchanged",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_dont_overwrite_parameters",
    "check_dtype_object",
    "check_estimator_sparse_array",
    "check_estimator_sparse_matrix",
    "check_estimators_dtypes",
    "check_estimators_empty_data_messages",
    "check_estimators_fit_returns_self",
    "check_estimators_pickle",
    "check_estimators_unfitted",
    "check_f_contiguous_array_estimator",
    "check_fit2d_1sample",
    "check_fit2d_1feature",
    "check_fit_check_is_fitted",
    "check_fit1d",
    "check_fit_score_takes_y",
    "check_no_attributes_set_in_init",
    "check_pipeline_consistency",
    "check_readonly_memmap_input",
    "check_transformers_unfitted",
    "check_transformer_n_iter",
]


if compare_version(sklearn_version, ">", "1.5.2"):
    extra_valid_checks.append("check_parameters_default_constructible")

# TODO remove when dropping support for sklearn_version < 1.5.0
if compare_version(sklearn_version, "<", "1.5.0"):
    extra_valid_checks.append("check_estimator_sparse_data")


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[HierarchicalKMeans(n_clusters=8)],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[HierarchicalKMeans(n_clusters=8)],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.parametrize(
    "test_list, n_clusters",
    [
        ([2.4, 2.6], 5),
        ([2.7, 3.0, 3.3], 9),
        ([10 / 3, 10 / 3, 10 / 3], 10),
        ([1 / 3, 11 / 3, 11 / 3, 10 / 3], 11),
    ],
)
def test_adjust_small_clusters(test_list, n_clusters):
    test_list = np.asarray(test_list)

    assert np.sum(test_list) == n_clusters

    list_round = _adjust_small_clusters(test_list, n_clusters)

    assert np.all(list_round != 0)
    assert np.sum(list_round) == n_clusters
    for a in list_round:
        assert isinstance(a, (int, np.integer))


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T
    test_labels = hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


def test_hierarchical_k_means_clustering():
    data_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=15)
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img).T

    with pytest.raises(
        ValueError,
        match="n_clusters should be an integer greater than 0."
        " -2 was provided.",
    ):
        HierarchicalKMeans(n_clusters=-2).fit(X)

    hkmeans = HierarchicalKMeans(n_clusters=8)
    X_red = hkmeans.fit_transform(X)
    X_compress = hkmeans.inverse_transform(X_red)

    assert_array_almost_equal(X.shape, X_compress.shape)

    hkmeans_scaled = HierarchicalKMeans(n_clusters=8, scaling=True)
    X_red_scaled = hkmeans_scaled.fit_transform(X)
    sizes = hkmeans_scaled.sizes_
    X_compress_scaled = hkmeans_scaled.inverse_transform(X_red_scaled)

    assert_array_almost_equal(
        np.asarray([np.sqrt(s) * a for s, a in zip(sizes, X_red)]),
        X_red_scaled,
    )
    assert_array_almost_equal(X_compress, X_compress_scaled)

    del X_red, X_compress, X_red_scaled, X_compress_scaled


@pytest.mark.parametrize("n_clusters", [2, 4, 5])
def test_hierarchical_k_means_clustering_surface(
    surf_img, surf_mask, n_clusters
):
    """Test hierarchical k-means clustering on surface."""
    # create a surface masker
    masker = SurfaceMasker(surf_mask()).fit()
    # mask the surface image with 50 samples
    X = masker.transform(surf_img(50)).T
    # instantiate HierarchicalKMeans with n_clusters
    clustering = HierarchicalKMeans(n_clusters=n_clusters)
    # fit and transform the data
    X_transformed = clustering.fit_transform(X)
    # inverse transform the transformed data
    X_inverse = clustering.inverse_transform(X_transformed)

    # make sure the n_features in transformed data were reduced to n_clusters
    assert X_transformed.shape[0] == n_clusters

    # make sure the inverse transformed data has the same shape as the original
    assert X_inverse.shape == X.shape
