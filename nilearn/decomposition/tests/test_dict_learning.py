import numpy as np

from nilearn._utils.testing import assert_less_equal
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.test_canica import _make_canica_test_data
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker


def test_dict_learning():
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=8)
    masker = NiftiMasker(mask_img=mask_img).fit()
    mask = mask_img.get_data() != 0
    flat_mask = mask.ravel()
    dict_init = masker.inverse_transform(components[:, flat_mask])
    dict_learning = DictLearning(n_components=4, random_state=0,
                                 dict_init=dict_init,
                                 mask=mask_img,
                                 smoothing_fwhm=0., alpha=1)

    dict_learning_auto_init = DictLearning(n_components=4, random_state=0,
                                           mask=mask_img,
                                           smoothing_fwhm=0., n_epochs=10,
                                           alpha=1)
    maps = {}
    for estimator in [dict_learning,
                      dict_learning_auto_init]:
        estimator.fit(data)
        maps[estimator] = estimator.masker_. \
            inverse_transform(estimator.components_).get_data()
        maps[estimator] = np.reshape(
                        np.rollaxis(maps[estimator], 3, 0)[:, mask],
                        (4, flat_mask.sum()))

    masked_components = components[:, flat_mask]
    for this_dict_learning in [dict_learning]:
        these_maps = maps[this_dict_learning]
        S = np.sqrt(np.sum(masked_components ** 2, axis=1))
        S[S == 0] = 1
        masked_components /= S[:, np.newaxis]

        S = np.sqrt(np.sum(these_maps ** 2, axis=1))
        S[S == 0] = 1
        these_maps /= S[:, np.newaxis]

        K = np.abs(masked_components.dot(these_maps.T))
        recovered_maps = np.sum(K > 0.9)
        assert(recovered_maps >= 2)

    # Smoke test n_epochs > 1
    dict_learning = DictLearning(n_components=4, random_state=0,
                                 dict_init=dict_init,
                                 mask=mask_img,
                                 smoothing_fwhm=0., n_epochs=2, alpha=1)
    dict_learning.fit(data)


def test_component_sign():
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(n_subjects=2,
                                                             noisy=True)
    for mp in components:
        assert_less_equal(-mp.min(), mp.max())

    dict_learning = DictLearning(n_components=4, random_state=rng,
                                 mask=mask_img,
                                 smoothing_fwhm=0., alpha=1)
    dict_learning.fit(data)
    for mp in iter_img(dict_learning.masker_.inverse_transform(
            dict_learning.components_)):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))
