"""Tests for nilearn.mass_univariate._utils."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.ndimage import generate_binary_structure

from nilearn.mass_univariate import _utils
from nilearn.mass_univariate.tests._testing import (
    get_tvalue_with_alternative_library,
)


@pytest.fixture
def null():
    """Return a dummy null distribution that can be reused across tests."""
    return [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]


@pytest.fixture
def test_arr4d():
    test_arr4d = np.zeros((10, 10, 10, 1))
    test_arr4d[:2, :2, :2, 0] = 5  # 8-voxel cluster, high intensity
    test_arr4d[7:, 7:, 7:, 0] = 1  # 27-voxel cluster, low intensity
    test_arr4d[6, 6, 6, 0] = 1  # corner touching second cluster
    test_arr4d[6, 6, 8, 0] = 1  # edge touching second cluster
    test_arr4d[3:5, 3:5, 3:5, 0] = -10  # negative cluster, very high intensity
    test_arr4d[5:6, 3:5, 3:5, 0] = 1  # cluster touching negative one
    return test_arr4d


@pytest.mark.parametrize(
    "two_sided_test, dh, true_max_tfce",
    [
        (
            False,
            "auto",
            5050,
        ),  # One-sided where positive cluster has highest TFCE
        (
            True,
            "auto",
            5555,
        ),  # Two-sided where negative cluster has highest TFCE
        (False, 1, 550),  # One-sided with preset dh
    ],
)
def test_calculate_tfce(two_sided_test, dh, true_max_tfce):
    """Test calculate_tfce."""
    test_arr4d = np.zeros((10, 10, 10, 1))
    bin_struct = generate_binary_structure(3, 1)

    # 10-voxel positive cluster, high intensity
    test_arr4d[:2, :2, :2, 0] = 10
    test_arr4d[0, 2, 0, 0] = 10
    test_arr4d[2, 0, 0, 0] = 10

    # 10-voxel negative cluster, higher intensity
    test_arr4d[3:5, 3:5, 3:5, 0] = -11
    test_arr4d[3, 5, 3, 0] = -11
    test_arr4d[5, 3, 3, 0] = -11

    test_tfce_arr4d = _utils.calculate_tfce(
        test_arr4d,
        bin_struct=bin_struct,
        E=1,
        H=1,
        dh=dh,
        two_sided_test=two_sided_test,
    )

    assert test_tfce_arr4d.shape == test_arr4d.shape
    assert np.max(np.abs(test_tfce_arr4d)) == true_max_tfce


@pytest.mark.parametrize(
    "test_values, expected_p_value", [(9, 0.95), (-9, 0.15), (0, 0.4)]
)
def test_null_to_p_float_1_tailed_lower_tailed(
    null, test_values, expected_p_value
):
    """Test null_to_p with single float input lower-tailed ."""
    assert math.isclose(
        _utils.null_to_p(test_values, null, alternative="smaller"),
        expected_p_value,
    )


@pytest.mark.parametrize(
    "test_values, expected_p_value", [(9, 0.05), (-9, 0.95), (0, 0.65)]
)
def test_null_to_p_float_1_tailed_uppper_tailed(
    test_values, expected_p_value, null
):
    """Test null_to_p with single float input upper-tailed."""
    assert math.isclose(
        _utils.null_to_p(test_values, null, alternative="larger"),
        expected_p_value,
    )


@pytest.mark.parametrize(
    "test_values, expected_p_value",
    [
        (0, 0.95),
        (9, 0.2),
        (10, 0.05),
        (
            20,
            0.05,
        ),  # Still 0.05 because minimum valid p-value is 1 / len(null)
    ],
)
def test_null_to_p_float_2_tailed(test_values, expected_p_value, null):
    """Test null_to_p with single float input two-sided."""
    result = _utils.null_to_p(test_values, null, alternative="two-sided")
    assert result == _utils.null_to_p(
        test_values * -1, null, alternative="two-sided"
    )
    assert math.isclose(result, expected_p_value)


def test_null_to_p_float_error(null):
    with pytest.raises(ValueError):
        _utils.null_to_p(9, null, alternative="raise")


@pytest.mark.parametrize(
    "alternative, expected_p_value",
    [("two-sided", 1 / 10000), ("smaller", 1 - 1 / 10000)],
)
def test_null_to_p_float_with_extreme_values(
    alternative, expected_p_value, rng
):
    """Test that 1/n(null) is preserved with extreme values."""
    null = rng.normal(size=10000)

    result = _utils.null_to_p(20, null, alternative=alternative)
    assert math.isclose(
        result,
        expected_p_value,
    )


def test_null_to_p_array(rng):
    """Test null_to_p with 1d array input."""
    N = 10000
    nulldist = rng.normal(size=N)
    t = np.sort(rng.normal(size=N))
    p = np.sort(_utils.null_to_p(t, nulldist))

    assert p.shape == (N,)
    assert (p < 1).all()
    assert (p > 0).all()

    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02


@pytest.mark.parametrize(
    "bin_struct, two_sided_test, true_size, true_mass",
    [
        (
            generate_binary_structure(3, 1),
            False,
            27,
            39.992,
        ),  # One-sided test: largest cluster doesn't have highest mass
        (
            generate_binary_structure(3, 1),
            True,
            27,
            79.992,
        ),  # Two-sided test where negative cluster has higher mass
        (
            generate_binary_structure(3, 2),
            True,
            28,
            79.992,
        ),  # Two-sided test with edge connectivity
        # should include edge-connected single voxel cluster
        (
            generate_binary_structure(3, 3),
            True,
            29,
            79.992,
        ),  # Two-sided test with corner connectivity
        # should include corner-connected single voxel cluster
    ],
)
def test_calculate_cluster_measures(
    test_arr4d, bin_struct, two_sided_test, true_size, true_mass
):
    """Test calculate_cluster_measures.

    true_mass : (8 vox * 5 intensity) - (8 vox * 0.001 thresh)
    """
    test_size, test_mass = _utils.calculate_cluster_measures(
        test_arr4d,
        threshold=0.001,
        bin_struct=bin_struct,
        two_sided_test=two_sided_test,
    )

    assert test_size[0] == true_size
    assert test_mass[0] == true_mass


def test_calculate_cluster_measures_on_empty_array():
    test_size, test_mass = _utils.calculate_cluster_measures(
        np.zeros((10, 10, 10, 1)),
        threshold=0.001,
        bin_struct=generate_binary_structure(3, 1),
        two_sided_test=True,
    )

    true_size = 0
    true_mass = 0
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass


def test_t_score_with_covars_and_normalized_design_nocovar(rng):
    """Test t-scores computation without covariates."""
    # Normalized data
    n_samples = 50

    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)
    var2 = rng.standard_normal((n_samples, 1))
    var2 = var2 / np.sqrt(np.sum(var2**2, 0))  # normalize

    # compute t-scores with nilearn routine
    t_val_own = _utils.t_score_with_covars_and_normalized_design(var1, var2)

    # compute t-scores with linalg or statsmodels
    t_val_alt = get_tvalue_with_alternative_library(var1, var2)
    assert_array_almost_equal(t_val_own, t_val_alt)


def test_t_score_with_covars_and_normalized_design_withcovar(rng):
    """Test t-scores computation with covariates."""
    # Normalized data
    n_samples = 50

    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)  # normalized
    var2 = rng.standard_normal((n_samples, 1))
    var2 = var2 / np.sqrt(np.sum(var2**2, 0))  # normalize
    covars = np.eye(n_samples, 3)  # covars is orthogonal
    covars[3] = -1  # covars is orthogonal to var1
    covars = _utils.orthonormalize_matrix(covars)

    # nilearn t-score
    own_score = _utils.t_score_with_covars_and_normalized_design(
        var1,
        var2,
        covars,
    )

    # compute t-scores with linalg or statmodels
    ref_score = get_tvalue_with_alternative_library(var1, var2, covars)
    assert_array_almost_equal(own_score, ref_score)
