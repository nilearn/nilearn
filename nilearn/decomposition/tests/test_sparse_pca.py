from nilearn.input_data import NiftiMasker

from nose.tools import assert_true, assert_greater
import numpy as np

from nilearn.decomposition.tests.test_canica import _make_canica_test_data
from nilearn.decomposition.sparse_pca import SparsePCA
from nilearn._utils.testing import assert_less_equal
from nilearn.image import iter_img


def test_dict_learning():
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=8)
    mask = NiftiMasker(mask_img=mask_img).fit()
    dict_init = mask.inverse_transform(components)
    sparse_pca = SparsePCA(n_components=4, random_state=0,
                           dict_init=dict_init,
                           mask=mask_img,
                           smoothing_fwhm=0., n_epochs=1, alpha=1)
    sparse_pca.fit(data)
    maps = sparse_pca.masker_.\
        inverse_transform(sparse_pca.components_).get_data()
    maps = np.reshape(np.rollaxis(maps, 3, 0),
                                 (4, 400))

    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    S = np.sqrt(np.sum(maps ** 2, axis=1))
    S[S == 0] = 1
    maps /= S[:, np.newaxis]

    K = np.abs(components.dot(maps.T))
    recovered_maps = np.sum(K > 0.9)
    assert_greater(recovered_maps, 2)

    # Smoke test n_epochs > 1
    sparse_pca = SparsePCA(n_components=4, random_state=0,
                           dict_init=dict_init,
                           mask=mask_img,
                           smoothing_fwhm=0., n_epochs=2, alpha=1)
    sparse_pca.fit(data)

    # Smoke test reduction_ratio < 1
    sparse_pca = SparsePCA(n_components=4, random_state=0,
                           dict_init=dict_init,
                           reduction_ratio=0.5,
                           mask=mask_img,
                           smoothing_fwhm=0., n_epochs=1, alpha=1)
    sparse_pca.fit(data)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(n_subjects=2,
                                                             noisy=True)
    for mp in components:
        assert_less_equal(-mp.min(), mp.max())

    sparse_pca = SparsePCA(n_components=4, random_state=rng,
                           mask=mask_img,
                           smoothing_fwhm=0., l1_ratio=1)
    sparse_pca.fit(data)
    for mp in iter_img(sparse_pca.masker_.inverse_transform(
            sparse_pca.components_)):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))
