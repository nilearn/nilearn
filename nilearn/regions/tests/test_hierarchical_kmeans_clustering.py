import numpy as np
import pytest
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.input_data import NiftiMasker
from nilearn.regions.hierarchical_kmeans_clustering import (
    HierarchicalKMeans,
    _adjust_small_clusters,
    hierarchical_k_means,
)
from numpy.testing import assert_array_almost_equal


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
