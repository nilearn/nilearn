"""Test the _utils.param_validation module."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from nibabel import Nifti1Image, load
from sklearn.base import BaseEstimator

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (
    MNI152_BRAIN_VOLUME,
    _cast_to_int32,
    _get_mask_extent,
    check_feature_screening,
    check_threshold,
)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
)


def test_check_threshold():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    name = "threshold"
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ["0.1", "10", "10.2.3%", "asdf%"]
    for wrong_threshold in wrong_thresholds:
        with pytest.raises(
            ValueError,
            match=f"{name}.+should be a number followed by the percent sign",
        ):
            check_threshold(wrong_threshold, matrix, fast_abs_percentile, name)

    threshold = object()
    with pytest.raises(
        TypeError, match=f"{name}.+should be either a number or a string"
    ):
        check_threshold(threshold, matrix, fast_abs_percentile, name)

    # Test threshold as int, threshold=2 should return as it is
    # since it is not string
    assert check_threshold(2, matrix, fast_abs_percentile) == 2

    # check whether raises a warning if given threshold is higher than expected
    with pytest.warns(UserWarning):
        check_threshold(3.0, matrix, fast_abs_percentile)

    # test with numpy scalar as argument
    threshold = 2.0
    threshold_numpy_scalar = np.float64(threshold)
    assert check_threshold(
        threshold, matrix, fast_abs_percentile
    ) == check_threshold(threshold_numpy_scalar, matrix, fast_abs_percentile)

    # Test for threshold provided as a percentile of the data
    # (str ending with a %)
    assert (
        1.0
        < check_threshold("50%", matrix, fast_abs_percentile, name=name)
        <= 2.0
    )


def test_get_mask_extent():
    # Test that hard-coded standard mask volume can be corrected computed
    if Path(mni152_brain_mask).is_file():
        assert _get_mask_extent(load(mni152_brain_mask)) == MNI152_BRAIN_VOLUME
    else:
        warnings.warn(f"Couldn't find {mni152_brain_mask} (for testing)")


def test_feature_screening(affine_eye):
    # dummy
    mask_img_data = np.zeros((182, 218, 182))
    mask_img_data[30:-30, 30:-30, 30:-30] = 1
    mask_img = Nifti1Image(mask_img_data, affine=affine_eye)

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:
            if screening_percentile == 100 or screening_percentile is None:
                assert (
                    check_feature_screening(
                        screening_percentile, mask_img, is_classif
                    )
                    is None
                )
            elif screening_percentile in {-1, 101}:
                with pytest.raises(ValueError):
                    check_feature_screening(
                        screening_percentile,
                        mask_img,
                        is_classif,
                    )
            elif screening_percentile == 20:
                assert isinstance(
                    check_feature_screening(
                        screening_percentile, mask_img, is_classif
                    ),
                    BaseEstimator,
                )


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32, np.int8))
def test_sample_mask_signed(dtype):
    """Check unsigned sample_mask is converted to signed."""
    sample_mask = np.arange(2, dtype=dtype)
    assert _cast_to_int32(sample_mask).dtype.kind == "i"


def test_sample_mask_raises_on_negative():
    """Check for error when sample_mask has negative."""
    with pytest.raises(
        ValueError, match="sample_mask should not contain negative values"
    ):
        _cast_to_int32(np.array([-1, -2, 1]))


def test_sample_mask_raises_on_high_index():
    """Check for error when sample_mask has a very high index."""
    with pytest.raises(
        ValueError, match="Max value in sample mask is larger than"
    ):
        _cast_to_int32(np.array(2**66))
