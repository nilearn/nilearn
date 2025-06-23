import numpy as np
import pytest

from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.conftest import (
    RANDOM_STATE,
    check_decomposition_estimator,
)
from nilearn.image import get_data, iter_img
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.surface.surface import get_data as get_surface_data


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize("n_epochs", [1, 2, 10])
def test_check_values_epoch_argument_smoke(
    decomposition_mask_img, n_epochs, canica_components, canica_data, data_type
):
    """Smoke test to check different values of the epoch argument."""
    if data_type == "nifti":
        masker = NiftiMasker(mask_img=decomposition_mask_img).fit()
        mask = get_data(decomposition_mask_img) != 0
    else:
        masker = SurfaceMasker(mask_img=decomposition_mask_img).fit()
        mask = get_surface_data(decomposition_mask_img) != 0

    flat_mask = mask.ravel()
    dict_init = masker.inverse_transform(canica_components[:, flat_mask])

    dict_learning = DictLearning(
        n_components=4,
        random_state=RANDOM_STATE,
        dict_init=dict_init,
        mask=decomposition_mask_img,
        n_epochs=n_epochs,
        smoothing_fwhm=None,
        alpha=1,
    )
    dict_learning.fit(canica_data)

    check_decomposition_estimator(dict_learning, data_type)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("data_type", ["nifti"])
def test_dict_learning(
    decomposition_mask_img, canica_components, canica_data, data_type
):
    """Check content of components_img_."""
    masker = NiftiMasker(mask_img=decomposition_mask_img).fit()
    mask = get_data(decomposition_mask_img) != 0
    flat_mask = mask.ravel()
    masked_components = canica_components[:, flat_mask]
    dict_init = masker.inverse_transform(masked_components)

    # Note that
    # adding smoothing will make this test break
    smoothing_fwhm = None

    dict_learning = DictLearning(
        n_components=4,
        random_state=RANDOM_STATE,
        dict_init=dict_init,
        mask=decomposition_mask_img,
        smoothing_fwhm=smoothing_fwhm,
        alpha=1,
    )

    dict_learning_auto_init = DictLearning(
        n_components=4,
        random_state=RANDOM_STATE,
        mask=decomposition_mask_img,
        n_epochs=10,
        smoothing_fwhm=smoothing_fwhm,
        alpha=1,
    )
    maps = {}
    for estimator in [dict_learning, dict_learning_auto_init]:
        estimator.fit(canica_data)

        check_decomposition_estimator(dict_learning, data_type)

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


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_component_sign(
    decomposition_mask_img, canica_data, data_type
) -> None:
    """Check sign of extracted components.

    Regression test:
    We should have a heuristic that flips the sign of components in
    DictLearning to have more positive values than negative values, for
    instance by making sure that the largest value is positive.
    """
    dict_learning = DictLearning(
        n_components=4,
        random_state=RANDOM_STATE,
        mask=decomposition_mask_img,
        smoothing_fwhm=None,
        alpha=1,
    )
    dict_learning.fit(canica_data)

    check_decomposition_estimator(dict_learning, data_type)

    for mp in iter_img(dict_learning.components_img_):
        mp = get_data(mp) if data_type == "nifti" else get_surface_data(mp)
        assert np.sum(mp[mp <= 0]) <= np.sum(mp[mp > 0])
