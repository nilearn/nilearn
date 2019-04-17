from nose.tools import assert_equal, assert_not_equal, assert_raises
from sklearn.externals.joblib import Memory
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.regions.rena_clustering import ReNA
from nilearn.input_data import NiftiMasker
from nilearn.image import index_img


def test_rena_clustering():
    data, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)

    nifti_masker = NiftiMasker(mask_img=mask_img).fit()
    n_voxels = nifti_masker.transform(data).shape[1]

    rena = ReNA(n_clusters=10, mask=nifti_masker, scaling=False)

    data_red = rena.fit_transform(data)
    data_compress = rena.inverse_transform(data_red)

    assert_equal(10, rena.n_clusters_)
    assert_equal(data.shape, data_compress.shape)

    memory = Memory(cachedir=None)
    rena = ReNA(n_clusters=-2, mask=nifti_masker, memory=memory)
    assert_raises(ValueError, rena.fit, data)

    rena = ReNA(n_clusters=10, mask=None, scaling=True)
    data_red = rena.fit_transform(index_img(data, 0))
    data_compress = rena.inverse_transform(data_red)

    for n_iter in [-2, 0]:
        rena = ReNA(n_iter=n_iter, mask=nifti_masker, memory=memory)
        assert_raises(ValueError, rena.fit, data)

    for n_clusters in [1, 2, 4, 8]:
        rena = ReNA(n_clusters=n_clusters, n_iter=1, mask=nifti_masker,
                    memory=memory).fit(data)
        assert_not_equal(n_clusters, rena.n_clusters_)

    del n_voxels, data_compress
