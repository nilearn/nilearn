"""Test the _utils.param_validation module."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from nibabel import Nifti1Image, load
from scipy.stats import scoreatpercentile
from sklearn.base import BaseEstimator

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (
    MNI152_BRAIN_VOLUME,
    _get_mask_extent,
    check_feature_screening,
    check_threshold,
)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
)


@pytest.fixture
def matrix():
    return np.array(
        [[-3.0, 2.0, -1.0, 0.0, -4.0], [4.0, -6.0, 5.0, 1.0, -3.0]]
    )


def test_check_threshold_positive_and_zero_ts_true(matrix):
    """Tests nilearn._utils.param_validation.check_threshold when
    two_sided=True, threshold is specified as a number and threshold >=0.
    """
    # Test threshold=0 should return as it is since it is not string
    assert check_threshold(0, matrix, scoreatpercentile, two_sided=True) == 0

    # Test threshold=6 should return as it is since it is not string
    assert check_threshold(6, matrix, scoreatpercentile, two_sided=True) == 6

    # test with numpy scalar as argument
    threshold = 2.0
    threshold_numpy_scalar = np.float64(threshold)
    assert check_threshold(
        threshold, matrix, scoreatpercentile, two_sided=True
    ) == check_threshold(
        threshold_numpy_scalar, matrix, scoreatpercentile, two_sided=True
    )

    # check whether raises a warning if given threshold is higher than expected
    with pytest.warns(UserWarning):
        check_threshold(6.5, matrix, scoreatpercentile, two_sided=True)


def test_check_threshold_positive_and_zero_ts_false(matrix):
    """Tests nilearn._utils.param_validation.check_threshold when
    two_sided=False, threshold is specified as a number and threshold >=0.
    """
    # Test threshold=4 should return as it is since it is not string
    assert check_threshold(5, matrix, scoreatpercentile, two_sided=False) == 5

    # check whether raises a warning if given threshold is higher than expected
    # 6 will raise warning as negative values are not considered
    with pytest.warns(UserWarning):
        check_threshold(6, matrix, scoreatpercentile, two_sided=False)


def test_check_threshold_percentile_positive_and_zero_ts_true(matrix):
    """Tests nilearn._utils.param_validation.check_threshold when
    two_sided=True, threshold is specified as percentile (str ending with a %)
    and threshold >=0.
    """
    # Test for threshold provided as a percentile of the data
    # ()
    threshold = check_threshold(
        "10%", matrix, scoreatpercentile, two_sided=True
    )
    assert 0 < threshold < 1.0

    threshold = check_threshold(
        "40%", matrix, scoreatpercentile, two_sided=True
    )
    assert 2.0 < threshold < 3.0

    threshold = check_threshold(
        "90%", matrix, scoreatpercentile, two_sided=True
    )
    assert 5.0 < threshold < 6.0


def test_check_threshold_percentile_positive_and_zero_ts_false(matrix):
    """Tests nilearn._utils.param_validation.check_threshold when
    two_sided=False, threshold is specified as percentile (str ending with a %)
    and threshold >=0.
    """
    threshold = check_threshold(
        "10%", matrix, scoreatpercentile, two_sided=False
    )
    assert 0 < threshold < 1.0

    threshold = check_threshold(
        "40%", matrix, scoreatpercentile, two_sided=False
    )
    assert 1.0 < threshold < 2.0

    threshold = check_threshold(
        "90%", matrix, scoreatpercentile, two_sided=False
    )
    assert 4.0 < threshold < 5.0


def test_check_threshold_negative_ts_false(matrix):
    """Tests nilearn._utils.param_validation.check_threshold when
    two_sided=False, threshold is specified as a number and threshold <=0.
    """
    # Test threshold=0 should return as it is since it is not string
    assert check_threshold(0, matrix, scoreatpercentile, two_sided=False) == 0

    # Test threshold=4 should return as it is since it is not string
    assert (
        check_threshold(-6, matrix, scoreatpercentile, two_sided=False) == -6
    )

    # check whether raises a warning if given threshold is higher than expected
    # -7 will raise warning as negative values are not considered
    with pytest.warns(UserWarning):
        check_threshold(-7, matrix, scoreatpercentile, two_sided=False)


def test_check_threshold_for_error(matrix):
    """Tests nilearn._utils.param_validation.check_threshold for errors."""
    name = "threshold"
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ["0.1", "10", "10.2.3%", "asdf%"]
    for wrong_threshold in wrong_thresholds:
        for two_sided in [True, False]:
            with pytest.raises(
                ValueError,
                match=f"{name}.+should be a number followed",
            ):
                check_threshold(
                    wrong_threshold,
                    matrix,
                    fast_abs_percentile,
                    name,
                    two_sided,
                )

    threshold = object()
    for two_sided in [True, False]:
        with pytest.raises(
            TypeError, match=f"{name}.+should be either a number or a string"
        ):
            check_threshold(
                threshold, matrix, fast_abs_percentile, name, two_sided
            )

    two_sided = True
    # invalid threshold values when two_sided=True
    thresholds = [-10, "-10%"]
    for wrong_threshold in thresholds:
        with pytest.raises(
            ValueError, match=f"{name}.+should not be a negative"
        ):
            check_threshold(
                wrong_threshold, matrix, fast_abs_percentile, name, two_sided
            )
    with pytest.raises(ValueError, match=f"{name}.+should not be a negative"):
        check_threshold(
            "-10%", matrix, fast_abs_percentile, name, two_sided=False
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
