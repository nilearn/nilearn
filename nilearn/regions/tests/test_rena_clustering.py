import numpy as np
from nose.tools import assert_equal, assert_not_equal, assert_raises
from sklearn.externals.joblib import Memory
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.regions.rena_clustering import ReNA
from nilearn.input_data import NiftiMasker


def test_rena_clustering():
    data_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)

    data = data_img.get_data()
    mask = mask_img.get_data()

    X = np.empty((data.shape[3], int(mask.sum())))
    for i in range(data.shape[3]):
        X[i, :] = np.copy(data[:, :, :, i])[mask_img.get_data() != 0]

    nifti_masker = NiftiMasker(mask_img=mask_img).fit()
    n_voxels = nifti_masker.transform(data_img).shape[1]

    rena = ReNA(mask_img, n_clusters=10)

    X_red = rena.fit_transform(X)
    X_compress = rena.inverse_transform(X_red)

    assert_equal(10, rena.n_clusters_)
    assert_equal(X.shape, X_compress.shape)

    memory = Memory(cachedir=None)
    rena = ReNA(mask_img, n_clusters=-2, memory=memory)
    assert_raises(ValueError, rena.fit, X)

    rena = ReNA(mask_img, n_clusters=10, scaling=True)
    X_red = rena.fit_transform(X)
    X_compress = rena.inverse_transform(X_red)

    for n_iter in [-2, 0]:
        rena = ReNA(mask_img, n_iter=n_iter, memory=memory)
        assert_raises(ValueError, rena.fit, X)

    for n_clusters in [1, 2, 4, 8]:
        rena = ReNA(mask_img, n_clusters=n_clusters, n_iter=1,
                    memory=memory).fit(X)
        assert_not_equal(n_clusters, rena.n_clusters_)

    del n_voxels, X_red, X_compress
