"""Test the _utils.param_validation module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.base import BaseEstimator

from nilearn.datasets import load_mni152_brain_mask
from nilearn.decoding._utils import (
    MNI152_BRAIN_VOLUME,
    _get_mask_extent,
    check_feature_screening,
)


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

    For very small ROIs, all elements should be included.
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
