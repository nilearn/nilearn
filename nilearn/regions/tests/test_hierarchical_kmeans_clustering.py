import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.regions.hierarchical_kmeans_clustering import (
    HierarchicalKMeans,
    _adjust_small_clusters,
    hierarchical_k_means,
)
from nilearn.surface import SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

# IMPORTANT
# keeping the n_clusters low (< 3) to make it easier
# to run sklearn checks
ESTIMATORS_TO_CHECK = [HierarchicalKMeans(n_clusters=2)]

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


def test_hierarchical_k_means_clustering_transform():
    n_samples = 15
    n_clusters = 8
    data_img, mask_img = generate_fake_fmri(
        shape=(10, 11, 12), length=n_samples
    )
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img)
    hkmeans = HierarchicalKMeans(n_clusters=n_clusters).fit(X)
    X_red = hkmeans.transform(X)

    assert X_red.shape == (n_samples, n_clusters)


def test_hierarchical_k_means_clustering_inverse_transform():
    n_samples = 15
    n_clusters = 8
    data_img, mask_img = generate_fake_fmri(
        shape=(10, 11, 12), length=n_samples
    )
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img)
    hkmeans = HierarchicalKMeans(n_clusters=n_clusters).fit(X)
    X_red = hkmeans.transform(X)
    X_inv = hkmeans.inverse_transform(X_red)

    assert X_inv.shape == X.shape


@pytest.mark.parametrize("n_clusters", [None, -2, 0, "2"])
def test_hierarchical_k_means_clustering_error_n_clusters(n_clusters):
    n_samples = 15
    data_img, mask_img = generate_fake_fmri(
        shape=(10, 11, 12), length=n_samples
    )
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img)

    with pytest.raises(
        ValueError,
        match="n_clusters should be an integer greater than 0."
        f" {n_clusters} was provided.",
    ):
        HierarchicalKMeans(n_clusters=n_clusters).fit(X)


def test_hierarchical_k_means_clustering_scaling():
    n_samples = 15
    n_clusters = 8
    data_img, mask_img = generate_fake_fmri(
        shape=(10, 11, 12), length=n_samples
    )
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img)

    hkmeans = HierarchicalKMeans(n_clusters=n_clusters)
    X_red = hkmeans.fit_transform(X)
    X_compress = hkmeans.inverse_transform(X_red)

    hkmeans_scaled = HierarchicalKMeans(n_clusters=n_clusters, scaling=True)
    X_red_scaled = hkmeans_scaled.fit_transform(X)
    sizes = hkmeans_scaled.sizes_
    X_compress_scaled = hkmeans_scaled.inverse_transform(X_red_scaled)

    assert_array_almost_equal(
        np.asarray([np.sqrt(s) * a for s, a in zip(sizes, X_red.T)]).T,
        X_red_scaled,
    )
    assert_array_almost_equal(X_compress, X_compress_scaled)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
@pytest.mark.parametrize("n_clusters", [2, 4, 5])
def test_hierarchical_k_means_clustering_surface(
    surf_img_2d, surf_mask_dim, surf_mask_1d, surf_mask_2d, n_clusters
):
    """Test hierarchical k-means clustering on surface."""
    n_samples = 100
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    # create a surface masker
    masker = SurfaceMasker(surf_mask).fit()
    # mask the surface image with 50 samples
    X = masker.transform(surf_img_2d(n_samples))
    # instantiate HierarchicalKMeans with n_clusters
    hkmeans = HierarchicalKMeans(n_clusters=n_clusters)
    # fit and transform the data
    X_transformed = hkmeans.fit_transform(X)
    # inverse transform the transformed data
    X_inverse = hkmeans.inverse_transform(X_transformed)

    # make sure the n_features in transformed data were reduced to n_clusters
    assert X_transformed.shape == (n_samples, n_clusters)
    assert hkmeans.n_clusters == n_clusters

    # make sure the inverse transformed data has the same shape as the original
    assert X_inverse.shape == X.shape


@pytest.mark.parametrize("img_type", ["surface", "volume"])
def test_hierarchical_k_means_n_clusters_warning(img_type, rng):
    n_samples = 15
    if img_type == "surface":
        mesh = {
            "left": flat_mesh(10, 8),
            "right": flat_mesh(9, 7),
        }
        data = {
            "left": rng.standard_normal(
                size=(mesh["left"].coordinates.shape[0], n_samples)
            ),
            "right": rng.standard_normal(
                size=(mesh["right"].coordinates.shape[0], n_samples)
            ),
        }
        img = SurfaceImage(mesh=mesh, data=data)
        X = SurfaceMasker().fit_transform(img)
    else:
        img, _ = generate_fake_fmri(shape=(10, 11, 12), length=n_samples)
        X = NiftiMasker().fit_transform(img)

    with pytest.warns(
        match="n_clusters should be at most the number of features.",
    ):
        # very high number of clusters
        HierarchicalKMeans(n_clusters=1000).fit_transform(X)
