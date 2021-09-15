from nilearn.input_data import NiftiMasker
import numpy as np
from numpy.testing import assert_array_almost_equal
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.regions.hierarchical_kmeans_clustering import (
    _hierarchical_k_means,
    HierarchicalKMeans)
import pytest


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T
    test_labels = _hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


def test_hierarchical_k_means_clustering():
    data_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=15)
    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img).T

    with pytest.raises(ValueError,
                       match="n_clusters should be an integer greater than 0."
                       " -2 was provided."):
        HierarchicalKMeans(n_clusters=-2).fit(X)

    hkmeans = HierarchicalKMeans(n_clusters=8)
    X_red = hkmeans.fit_transform(X)
    X_compress = hkmeans.inverse_transform(X_red)

    assert_array_almost_equal(X.shape, X_compress.shape)

    hkmeans_scaled = HierarchicalKMeans(n_clusters=8, scaling=True)
    X_red_scaled = hkmeans_scaled.fit_transform(X)
    sizes = hkmeans_scaled.sizes_
    X_compress_scaled = hkmeans_scaled.inverse_transform(X_red_scaled)

    assert_array_almost_equal(np.asarray(
        [np.sqrt(s) * a for s, a in zip(sizes, X_red)]), X_red_scaled)
    assert_array_almost_equal(X_compress, X_compress_scaled)

    del X_red, X_compress, X_red_scaled, X_compress_scaled
