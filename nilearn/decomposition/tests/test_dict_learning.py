import numpy as np
import nibabel

from nose.tools import assert_true
from nilearn._utils.testing import (assert_less_equal, assert_raises_regex,
                                    write_tmp_imgs)
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.test_canica import _make_canica_test_data
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker
from nilearn.decomposition.tests.test_multi_pca import _tmp_dir


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
        maps[estimator] = estimator.components_img_.get_data()
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
    for mp in iter_img(dict_learning.components_img_):
        mp = mp.get_data()
        assert_less_equal(np.sum(mp[mp <= 0]), np.sum(mp[mp > 0]))


def test_masker_attributes_with_fit():
    # Test base module at sub-class
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=3)
    # Passing mask_img
    dict_learning = DictLearning(n_components=3, mask=mask_img, random_state=0)
    dict_learning.fit(data)
    assert_true(dict_learning.mask_img_ == mask_img)
    assert_true(dict_learning.mask_img_ == dict_learning.masker_.mask_img_)
    # Passing masker
    masker = NiftiMasker(mask_img=mask_img)
    dict_learning = DictLearning(n_components=3, mask=masker, random_state=0)
    dict_learning.fit(data)
    assert_true(dict_learning.mask_img_ == dict_learning.masker_.mask_img_)
    dict_learning = DictLearning(mask=mask_img, n_components=3)
    assert_raises_regex(ValueError,
                        "Object has no components_ attribute. "
                        "This is probably because fit has not been called",
                        dict_learning.transform, data)
    # Test if raises an error when empty list of provided.
    assert_raises_regex(ValueError,
                        'Need one or more Niimg-like objects as input, '
                        'an empty list was given.',
                        dict_learning.fit, [])
    # Test passing masker arguments to estimator
    dict_learning = DictLearning(n_components=3,
                                 target_affine=np.eye(4),
                                 target_shape=(6, 8, 10),
                                 mask_strategy='background')
    dict_learning.fit(data)


def test_components_img():
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=3)
    n_components = 3
    dict_learning = DictLearning(n_components=n_components, mask=mask_img)
    dict_learning.fit(data)
    components_img = dict_learning.components_img_
    assert_true(isinstance(components_img, nibabel.Nifti1Image))
    check_shape = data[0].shape + (n_components,)
    assert_true(components_img.shape, check_shape)


def test_with_globbing_patterns_with_single_subject():
    # single subject
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=1)
    n_components = 3
    dictlearn = DictLearning(n_components=n_components, mask=mask_img)
    with write_tmp_imgs(data[0], create_files=True, use_wildcards=True) as img:
        input_image = _tmp_dir() + img
        dictlearn.fit(input_image)
        components_img = dictlearn.components_img_
        assert_true(isinstance(components_img, nibabel.Nifti1Image))
        # n_components = 3
        check_shape = data[0].shape[:3] + (3,)
        assert_true(components_img.shape, check_shape)


def test_with_globbing_patterns_with_multi_subjects():
    # multi subjects
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=3)
    n_components = 3
    dictlearn = DictLearning(n_components=n_components, mask=mask_img)
    with write_tmp_imgs(data[0], data[1], data[2], create_files=True,
                        use_wildcards=True) as img:
        input_image = _tmp_dir() + img
        dictlearn.fit(input_image)
        components_img = dictlearn.components_img_
        assert_true(isinstance(components_img, nibabel.Nifti1Image))
        # n_components = 3
        check_shape = data[0].shape[:3] + (3,)
        assert_true(components_img.shape, check_shape)
