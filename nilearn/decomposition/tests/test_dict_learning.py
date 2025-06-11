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
from nilearn.image import get_data, iter_img
from nilearn.maskers import NiftiMasker

ESTIMATORS_TO_CHECK = [DictLearning()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(
        estimator,
        check,
        name,  # noqa: ARG001
    ):
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
    def test_check_estimator_sklearn_invalid(
        estimator,
        check,
        name,  # noqa: ARG001
    ):
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
    decomposition_mask_img,
    n_epochs,
    canica_components,
    canica_data,
    data_type,  # noqa: ARG001
):
    """Smoke test to check different values of the epoch argument."""
    masker = NiftiMasker(mask_img=decomposition_mask_img).fit()
    mask = get_data(decomposition_mask_img) != 0
    flat_mask = mask.ravel()
    dict_init = masker.inverse_transform(canica_components[:, flat_mask])

    dict_learning = DictLearning(
        n_components=4,
        random_state=0,
        dict_init=dict_init,
        mask=decomposition_mask_img,
        smoothing_fwhm=0.0,
        n_epochs=n_epochs,
        alpha=1,
    )
    dict_learning.fit(canica_data)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_dict_learning(
    decomposition_mask_img,
    canica_components,
    canica_data,
    data_type,  # noqa: ARG001
):
    """Check content of components_img_."""
    masker = NiftiMasker(mask_img=decomposition_mask_img).fit()
    mask = get_data(decomposition_mask_img) != 0
    flat_mask = mask.ravel()
    masked_components = canica_components[:, flat_mask]
    dict_init = masker.inverse_transform(masked_components)

    dict_learning = DictLearning(
        n_components=4,
        random_state=0,
        dict_init=dict_init,
        mask=decomposition_mask_img,
        smoothing_fwhm=0.0,
        alpha=1,
    )

    dict_learning_auto_init = DictLearning(
        n_components=4,
        random_state=0,
        mask=decomposition_mask_img,
        smoothing_fwhm=0.0,
        n_epochs=10,
        alpha=1,
    )
    maps = {}
    for estimator in [dict_learning, dict_learning_auto_init]:
        estimator.fit(canica_data)
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
def test_component_sign(
    decomposition_mask_img,
    canica_data,
    data_type,  # noqa: ARG001
) -> None:
    """Check sign of extracted components.

    Regression test:
    We should have a heuristic that flips the sign of components in
    DictLearning to have more positive values than negative values, for
    instance by making sure that the largest value is positive.
    """
    dict_learning = DictLearning(
        n_components=4,
        random_state=42,
        mask=decomposition_mask_img,
        smoothing_fwhm=0,
        alpha=1,
    )
    dict_learning.fit(canica_data)
    for mp in iter_img(dict_learning.components_img_):
        mp = get_data(mp)
        assert np.sum(mp[mp <= 0]) <= np.sum(mp[mp > 0])


@pytest.mark.parametrize("data_type", ["nifti"])
def test_masker_attributes_with_fit(
    canica_data_single_img,
    decomposition_mask_img,
    data_type,  # noqa: ARG001
):
    """Test base module at sub-class."""
    # Passing mask_img
    dict_learning = DictLearning(
        n_components=3, mask=decomposition_mask_img, random_state=0
    )
    dict_learning.fit(canica_data_single_img)

    assert dict_learning.mask_img_ == dict_learning.masker_.mask_img_

    # Passing masker
    masker = NiftiMasker(mask_img=decomposition_mask_img)
    dict_learning = DictLearning(n_components=3, mask=masker, random_state=0)
    dict_learning.fit(canica_data_single_img)

    assert dict_learning.mask_img_ == dict_learning.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti"])
def test_empty_data_to_fit_error(
    decomposition_mask_img,
    data_type,  # noqa: ARG001
):
    """Test if raises an error when empty list of provided."""
    dict_learning = DictLearning(mask=decomposition_mask_img, n_components=3)

    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects "
        "as input, an empty list was given.",
    ):
        dict_learning.fit([])


def test_passing_masker_arguments_to_estimator(
    affine_eye, canica_data_single_img
):
    """Smoke test that arguments for masker are properly passed."""
    dict_learning = DictLearning(
        n_components=3,
        target_affine=affine_eye,
        target_shape=(6, 8, 10),
        mask_strategy="background",
    )
    dict_learning.fit(canica_data_single_img)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_components_img(
    canica_data_single_img,
    decomposition_mask_img,
    data_type,  # noqa: ARG001
):
    """Check content of components_img_ after fitting."""
    n_components = 3
    dict_learning = DictLearning(
        n_components=n_components, mask=decomposition_mask_img
    )

    dict_learning.fit(canica_data_single_img)
    components_img = dict_learning.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = (*canica_data_single_img.shape[:3], n_components)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
@pytest.mark.parametrize("n_subjects", [1, 3])
def test_with_globbing_patterns(
    decomposition_mask_img,
    tmp_path,
    canica_data,
    n_subjects,  # noqa: ARG001
    data_type,  # noqa: ARG001
):
    """Check DictLearning can work with files on disk."""
    n_components = 3
    dict_learning = DictLearning(
        n_components=n_components, mask=decomposition_mask_img
    )

    img = write_imgs_to_path(
        *canica_data, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    dict_learning.fit(img)
    components_img = dict_learning.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = (*canica_data[0].shape[:3], n_components)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
def test_dictlearning_score(
    canica_data_single_img,
    decomposition_mask_img,
    data_type,  # noqa: ARG001
):
    """Check content of scores after fitting."""
    # Multi subjects
    n_components = 10
    dict_learning = DictLearning(
        n_components=n_components, mask=decomposition_mask_img, random_state=0
    )

    dict_learning.fit(canica_data_single_img)

    # One score for all components
    scores = dict_learning.score(canica_data_single_img, per_component=False)

    assert scores <= 1
    assert scores >= 0

    # Per component score
    scores = dict_learning.score(canica_data_single_img, per_component=True)

    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)
