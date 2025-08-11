"""Test the _utils.param_validation module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from scipy.stats import scoreatpercentile
from sklearn.base import BaseEstimator

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (
    MNI152_BRAIN_VOLUME,
    _cast_to_int32,
    _get_mask_extent,
    check_feature_screening,
    check_params,
    check_threshold,
)
from nilearn.datasets import load_mni152_brain_mask


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


def test_check_params():
    """Check that passing incorrect type to a function raises TypeError."""

    def f_with_param_to_check(data_dir):
        check_params(locals())
        return data_dir

    f_with_param_to_check(data_dir="foo")

    with pytest.raises(TypeError, match="'data_dir' should be of type"):
        f_with_param_to_check(data_dir=1)


def test_check_params_not_necessary():
    """Check an error is raised when function is used when not needed."""

    def f_with_unknown_param(foo):
        check_params(locals())
        return foo

    with pytest.raises(ValueError, match="No known parameter to check."):
        f_with_unknown_param(foo=1)
