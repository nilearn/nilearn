"""Test the _utils.param_validation module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, SelectPercentile

from nilearn.conftest import _img_3d_rand, _surf_img_1d
from nilearn.datasets import load_mni152_brain_mask
from nilearn.decoding._utils import (
    MNI152_BRAIN_VOLUME,
    _get_mask_extent,
    check_feature_screening,
)


@pytest.mark.thread_unsafe
def test_get_mask_extent():
    """Test that hard-coded standard mask volume can be corrected computed."""
    assert _get_mask_extent(load_mni152_brain_mask()) == MNI152_BRAIN_VOLUME


@pytest.mark.parametrize("is_classif", [True, False])
@pytest.mark.parametrize("screening_percentile", [100, None, 20, 101, -1, 10])
@pytest.mark.parametrize("roi_size", ["small", "large"])
def test_feature_screening(
    affine_eye, is_classif, screening_percentile, roi_size
):
    """Check that screening percentile is correctly adjusted.

    For very small ROIs, all elements should be i_img_3d_randncluded.
    """
    mask_img_data = np.zeros((182, 218, 182))

    if roi_size == "small":
        mask_img_data[80:-80, 80:-80, 80:-80] = 1
    else:
        mask_img_data[40:-40, 40:-40, 40:-40] = 1

    mask_img = Nifti1Image(mask_img_data, affine=affine_eye)

    if screening_percentile == 100 or screening_percentile is None:
        assert (
            check_feature_screening(screening_percentile, mask_img, is_classif)
            is None
        )
    elif screening_percentile in {-1, 101}:
        with pytest.raises(ValueError):
            check_feature_screening(
                screening_percentile,
                mask_img,
                is_classif,
            )
    else:
        if roi_size == "small":
            with pytest.warns(
                UserWarning, match="screening_percentile set to '100'"
            ):
                select_percentile = check_feature_screening(
                    screening_percentile, mask_img, is_classif
                )
            assert select_percentile.percentile == 100
        else:
            select_percentile = check_feature_screening(
                screening_percentile, mask_img, is_classif
            )
            assert screening_percentile <= select_percentile.percentile < 100
        assert isinstance(select_percentile, BaseEstimator)


@pytest.mark.parametrize("mask_img", [_img_3d_rand(), _surf_img_1d()])
def test_screening_priority_logic(mask_img):
    """Test that check_feature_screening prefers percentile over n_voxels.

    Call the function with BOTH options (Conflict!)
    screening_percentile=10, screening_n_features=50

    We should get a SelectPercentile object
    If logic is wrong, this will be SelectKBest and the test will fail.
    """
    with pytest.warns(UserWarning):
        selector = check_feature_screening(
            screening_percentile=10,
            mask_img=mask_img,
            is_classification=True,
            screening_n_features=50,
        )

    assert isinstance(selector, SelectPercentile)
    assert not isinstance(selector, SelectKBest)


@pytest.mark.parametrize("mask_img", [_img_3d_rand(), _surf_img_1d()])
def test_check_feature_screening_n_features_only(mask_img):
    """Test that screening_n_features works when percentile is None."""
    # Call the function with only n_voxels specified
    selector = check_feature_screening(
        screening_percentile=None,
        mask_img=mask_img,
        is_classification=True,
        screening_n_features=7,
    )

    # Verify it returned a SelectKBest object with the right 'k'
    assert isinstance(selector, SelectKBest)
    assert selector.k == 7


def test_check_feature_screening_n_features_error(surf_img_1d):
    """Test when not enough feature in image."""
    with pytest.raises(
        ValueError,
        match="screening_n_features=100 is larger the number of features",
    ):
        check_feature_screening(
            screening_percentile=None,
            mask_img=surf_img_1d,
            is_classification=True,
            screening_n_features=100,
        )
