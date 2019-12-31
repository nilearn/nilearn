import numpy as np
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.regions.hierarchical_kmeans_clustering import _hierarchical_k_means, HierarchicalKMeans
from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T
    test_labels = _hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


def test_hierarchical_k_means_clustering():
    data_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=15)

    data = data_img.get_data()
    mask = mask_img.get_data()

    masker = NiftiMasker(mask_img=mask_img).fit()
    X = masker.transform(data_img).T

    hkmeans = HierarchicalKMeans(n_clusters=8)
    X_red = hkmeans.fit_transform(X)
    X_compress = hkmeans.inverse_transform(X_red)

    assert_array_almost_equal(X.shape, X_compress.shape)
    hkmeans = HierarchicalKMeans(n_clusters=-2)

    del X_red, X_compress
