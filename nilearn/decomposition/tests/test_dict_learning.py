from sklearn.utils.linear_assignment_ import linear_assignment
from numpy.testing import assert_array_almost_equal
import numpy as np

from nilearn.decomposition.tests.test_canica import _make_canica_test_data
from nilearn.decomposition.dict_learning import DictLearning
from nilearn._utils.testing import assert_less_equal
from nilearn.image import iter_img


def test_dict_learning():
    data, mask_img, components, rng = _make_canica_test_data()

    dict_learning = DictLearning(n_components=4, random_state=rng, mask=mask_img,
                                 smoothing_fwhm=0., n_iter=200, alpha=2)
    dict_learning.fit(data)
    maps = dict_learning.masker_.inverse_transform(dict_learning.components_).get_data()
    maps = np.reshape(np.rollaxis(maps, 3, 0), (4, 400))

    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    S = np.sqrt(np.sum(maps ** 2, axis=1))
    S[S == 0] = 1
    maps /= S[:, np.newaxis]

    K = np.abs(components.dot(maps.T))

    indices = linear_assignment(1-K)
    K = K[indices[:, 0], :][:, indices[:, 1]]
    assert_array_almost_equal(np.abs(K), np.eye(4), 1)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(n_subjects=2, noisy=True)
    for mp in components:
        assert_less_equal(-mp.min(), mp.max())

    # run CanICA many times (this is known to produce different results)
    dict_learning = DictLearning(n_components=4, random_state=rng, mask=mask_img,
                                 smoothing_fwhm=0., n_iter=100, alpha=1)
    dict_learning.fit(data)
    for mp in iter_img(dict_learning.masker_.inverse_transform(
            dict_learning.components_)):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))