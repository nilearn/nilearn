__author__ = 'arthur'

from nilearn.decomposition.tests.test_canica import _make_canica_test_data, _make_canica_components
from nilearn.decomposition.dict_learning import DictLearning
from sklearn.utils.linear_assignment_ import linear_assignment
from nilearn._utils.testing import assert_less_equal
from numpy.testing import assert_array_almost_equal
from nilearn.image import iter_img

import numpy as np


def test_dict_learning():
    data, mask_img, components, rng = _make_canica_test_data()

    dict_learning = DictLearning(n_components=4, random_state=rng, mask=mask_img,
                                 smoothing_fwhm=0., n_init=1, n_iter=100, alpha=0.1)
    dict_learning.fit(data)
    maps = dict_learning.masker_.inverse_transform(dict_learning.components_).get_data()
    maps = np.reshape(np.rollaxis(maps, 3, 0), (4, 400))

    K = np.corrcoef(np.concatenate((components, maps)))[4:, :4]
    indices = linear_assignment(-K)
    K = K.take(indices[:, 1], axis=1).take(indices[:, 0], axis=0)
    assert_array_almost_equal(np.abs(K), np.eye(4), 1)


def test_component_sign():
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(n_subjects=2, noisy=True)
    for mp in components:
        assert_less_equal(-mp.min(), mp.max())

    # run CanICA many times (this is known to produce different results)
    canica = DictLearning(n_components=4, random_state=rng, mask=mask_img)
    for _ in range(3):
        canica.fit(data)
        for mp in iter_img(canica.masker_.inverse_transform(
                canica.components_)):
            mp = mp.get_data()
            assert_less_equal(-mp.min(), mp.max())