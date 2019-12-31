import numpy as np
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.regions.hierarchical_kmeans_clustering import _hierarchical_k_means
from sklearn.utils.testing import assert_array_almost_equal


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

    X = np.empty((data.shape[3], int(mask.sum())))
    for i in range(data.shape[3]):
        X[i, :] = np.copy(data[:, :, :, i])[mask_img.get_data() != 0]

    n_voxels = mask_img.get_data().sum()

    hkmeans = HierarchicalKMeans(n_clusters=10)

    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(mask_img=mask_img).fit()
    b = masker.transform(data_img)
    b.shape
    X.shape
    hkmeans.fit(X.T)

    unique_labels = np.unique(hkmeans.labels_)
    X.shape[1]
    mean_cluster = []
    for label in unique_labels:
        mean_cluster.append(np.mean(X.T[:, hkmeans.labels_ == label], axis=1))

    X_red = np.array(mean_cluster).T

    hkmeans.fit(b)
    hkmeans.labels_
    X_red = hkmeans.fit_transform(X)

    unique_labels = np.unique(hkmeans.labels_)
    mean_cluster = []
    for label in unique_labels:
        mean_cluster.append(np.mean(X.T[:, hkmeans.labels_ == label], axis=1))
    np.shape(mean_cluster)

    X_compress = hkmeans.inverse_transform(X_red)

    assert_equal(10, hkmeans.n_clusters_)
    assert_equal(X.shape, X_compress.shape)

    hkmeans = HierarchicalKMeans(n_clusters=-2)
    assert_raises(ValueError, hkmeans.fit(), X)

    del n_voxels, X_red, X_compress
