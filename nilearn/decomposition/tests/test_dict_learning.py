import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.conftest import _make_canica_test_data
from nilearn.image import get_data, iter_img
from nilearn.maskers import NiftiMasker

ESTIMATORS_TO_CHECK = [DictLearning()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
        ),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.parametrize("data_type", ["nifti"])
@pytest.mark.parametrize("n_epochs", [1, 2, 10])
def test_dict_learning_check_values_epoch_argument_smoke(
    data_type, mask_img, n_epochs, canica_components
):
    """Smoke test to check different values of the epoch argument."""
    data = _make_canica_test_data()

    masker = NiftiMasker(mask_img=mask_img).fit()
    mask = get_data(mask_img) != 0
    flat_mask = mask.ravel()
    dict_init = masker.inverse_transform(canica_components[:, flat_mask])

    dict_learning = DictLearning(
        n_components=4,
        random_state=0,
        dict_init=dict_init,
        mask=mask_img,
        smoothing_fwhm=0.0,
        n_epochs=n_epochs,
        alpha=1,
    )
    dict_learning.fit(data)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_dict_learning(data_type, mask_img, canica_components):
    data = _make_canica_test_data()

    masker = NiftiMasker(mask_img=mask_img).fit()
    mask = get_data(mask_img) != 0
    flat_mask = mask.ravel()
    masked_components = canica_components[:, flat_mask]
    dict_init = masker.inverse_transform(masked_components)

    dict_learning = DictLearning(
        n_components=4,
        random_state=0,
        dict_init=dict_init,
        mask=mask_img,
        smoothing_fwhm=0.0,
        alpha=1,
    )

    dict_learning_auto_init = DictLearning(
        n_components=4,
        random_state=0,
        mask=mask_img,
        smoothing_fwhm=0.0,
        n_epochs=10,
        alpha=1,
    )
    maps = {}
    for estimator in [dict_learning, dict_learning_auto_init]:
        estimator.fit(data)
        maps[estimator] = get_data(estimator.components_img_)
        maps[estimator] = np.reshape(
            np.rollaxis(maps[estimator], 3, 0)[:, mask], (4, flat_mask.sum())
        )

    for this_dict_learning in [dict_learning]:
        these_maps = maps[this_dict_learning]
        S = np.sqrt(np.sum(masked_components**2, axis=1))
        S[S == 0] = 1
        masked_components /= S[:, np.newaxis]

        S = np.sqrt(np.sum(these_maps**2, axis=1))
        S[S == 0] = 1
        these_maps /= S[:, np.newaxis]

        K = np.abs(masked_components.dot(these_maps.T))
        recovered_maps = np.sum(K > 0.9)

        assert recovered_maps >= 2


@pytest.mark.parametrize("data_type", ["nifti"])
def test_component_sign(data_type, mask_img) -> None:
    # Regression test
    # We should have a heuristic that flips the sign of components in
    # DictLearning to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data = _make_canica_test_data(n_subjects=2)

    dict_learning = DictLearning(
        n_components=4,
        random_state=42,
        mask=mask_img,
        smoothing_fwhm=0,
        alpha=1,
    )
    dict_learning.fit(data)
    for mp in iter_img(dict_learning.components_img_):
        mp = get_data(mp)
        assert np.sum(mp[mp <= 0]) <= np.sum(mp[mp > 0])


@pytest.mark.parametrize("data_type", ["nifti"])
def test_masker_attributes_with_fit(canica_data, mask_img, data_type):
    # Test base module at sub-class

    # Passing mask_img
    dict_learning = DictLearning(n_components=3, mask=mask_img, random_state=0)
    dict_learning.fit(canica_data)

    assert dict_learning.mask_img_ == dict_learning.masker_.mask_img_

    # Passing masker
    masker = NiftiMasker(mask_img=mask_img)
    dict_learning = DictLearning(n_components=3, mask=masker, random_state=0)
    dict_learning.fit(canica_data)

    assert dict_learning.mask_img_ == dict_learning.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti"])
def test_empty_data_to_fit_error(mask_img, data_type):
    """Test if raises an error when empty list of provided."""
    dict_learning = DictLearning(mask=mask_img, n_components=3)

    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects "
        "as input, an empty list was given.",
    ):
        dict_learning.fit([])


def test_passing_masker_arguments_to_estimator(affine_eye, canica_data):
    dict_learning = DictLearning(
        n_components=3,
        target_affine=affine_eye,
        target_shape=(6, 8, 10),
        mask_strategy="background",
    )
    dict_learning.fit(canica_data)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_components_img(canica_data, mask_img, data_type):
    n_components = 3
    dict_learning = DictLearning(n_components=n_components, mask=mask_img)

    dict_learning.fit(canica_data)
    components_img = dict_learning.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = canica_data[0].shape[:3] + (n_components,)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
@pytest.mark.parametrize("n_subjects", [1, 3])
def test_with_globbing_patterns(mask_img, n_subjects, tmp_path, data_type):
    data = _make_canica_test_data(n_subjects=n_subjects)

    n_components = 3
    dict_learning = DictLearning(n_components=n_components, mask=mask_img)

    # just for test readability
    if n_subjects == 1:
        data = [data[0]]
    elif n_subjects == 3:
        data = [data[0], data[1], data[2]]

    img = write_imgs_to_path(
        *data, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    dict_learning.fit(img)
    components_img = dict_learning.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = data[0].shape[:3] + (n_components,)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
def test_dictlearning_score(canica_data, mask_img, data_type):
    # Multi subjects
    n_components = 10
    dict_learning = DictLearning(
        n_components=n_components, mask=mask_img, random_state=0
    )

    dict_learning.fit(canica_data)

    # One score for all components
    scores = dict_learning.score(canica_data, per_component=False)

    assert scores <= 1
    assert scores >= 0

    # Per component score
    scores = dict_learning.score(canica_data, per_component=True)

    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)
