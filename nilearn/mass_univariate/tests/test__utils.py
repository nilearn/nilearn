"""Tests for nilearn.mass_univariate._utils."""
import math

import numpy as np
import pytest
from nilearn.mass_univariate import _utils
from nilearn.mass_univariate.tests.utils import (
    get_tvalue_with_alternative_library,
)
from numpy.testing import assert_array_almost_equal
from scipy.ndimage import generate_binary_structure
from sklearn.utils import check_random_state


def test__calculate_tfce():
    """Test _calculate_tfce."""
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

    # One-sided test where positive cluster has the highest TFCE
    true_max_tfce = 5050
    test_tfce_arr4d = _utils._calculate_tfce(
        test_arr4d,
        bin_struct=bin_struct,
        E=1,
        H=1,
        dh="auto",
        two_sided_test=False,
    )
    assert test_tfce_arr4d.shape == test_arr4d.shape
    assert np.max(np.abs(test_tfce_arr4d)) == true_max_tfce

    # Two-sided test where negative cluster has the highest TFCE
    true_max_tfce = 5555
    test_tfce_arr4d = _utils._calculate_tfce(
        test_arr4d,
        bin_struct=bin_struct,
        E=1,
        H=1,
        dh="auto",
        two_sided_test=True,
    )
    assert test_tfce_arr4d.shape == test_arr4d.shape
    assert np.max(np.abs(test_tfce_arr4d)) == true_max_tfce

    # One-sided test with preset dh
    true_max_tfce = 550
    test_tfce_arr4d = _utils._calculate_tfce(
        test_arr4d,
        bin_struct=bin_struct,
        E=1,
        H=1,
        dh=1,
        two_sided_test=False,
    )
    assert test_tfce_arr4d.shape == test_arr4d.shape
    assert np.max(np.abs(test_tfce_arr4d)) == true_max_tfce


def test_null_to_p_float():
    """Test _null_to_p with single float input."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Left/lower-tailed
    assert math.isclose(
        _utils._null_to_p(9, null, alternative="smaller"),
        0.95,
    )
    assert math.isclose(
        _utils._null_to_p(-9, null, alternative="smaller"),
        0.15,
    )
    assert math.isclose(_utils._null_to_p(0, null, alternative="smaller"), 0.4)

    # Right/upper-tailed
    assert math.isclose(_utils._null_to_p(9, null, alternative="larger"), 0.05)
    assert math.isclose(
        _utils._null_to_p(-9, null, alternative="larger"),
        0.95,
    )
    assert math.isclose(_utils._null_to_p(0, null, alternative="larger"), 0.65)

    # Test that 1/n(null) is preserved with extreme values
    nulldist = np.random.normal(size=10000)
    assert math.isclose(
        _utils._null_to_p(20, nulldist, alternative="two-sided"),
        1 / 10000,
    )
    assert math.isclose(
        _utils._null_to_p(20, nulldist, alternative="smaller"),
        1 - 1 / 10000,
    )

    # Two-tailed
    assert math.isclose(
        _utils._null_to_p(0, null, alternative="two-sided"),
        0.95,
    )
    result = _utils._null_to_p(9, null, alternative="two-sided")
    assert result == _utils._null_to_p(-9, null, alternative="two-sided")
    assert math.isclose(result, 0.2)
    result = _utils._null_to_p(10, null, alternative="two-sided")
    assert result == _utils._null_to_p(-10, null, alternative="two-sided")
    assert math.isclose(result, 0.05)

    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = _utils._null_to_p(20, null, alternative="two-sided")
    assert result == _utils._null_to_p(-20, null, alternative="two-sided")
    assert math.isclose(result, 0.05)

    with pytest.raises(ValueError):
        _utils._null_to_p(9, null, alternative="raise")


def test_null_to_p_array():
    """Test _null_to_p with 1d array input."""
    N = 10000
    nulldist = np.random.normal(size=N)
    t = np.sort(np.random.normal(size=N))
    p = np.sort(_utils._null_to_p(t, nulldist))
    assert p.shape == (N,)
    assert (p < 1).all()
    assert (p > 0).all()
    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02


def test_calculate_cluster_measures():
    """Test _calculate_cluster_measures."""
    threshold = 0.001
    bin_struct = generate_binary_structure(3, 1)

    test_arr4d = np.zeros((10, 10, 10, 1))
    test_arr4d[:2, :2, :2, 0] = 5  # 8-voxel cluster, high intensity
    test_arr4d[7:, 7:, 7:, 0] = 1  # 27-voxel cluster, low intensity
    test_arr4d[6, 6, 6, 0] = 1  # corner touching second cluster
    test_arr4d[6, 6, 8, 0] = 1  # edge touching second cluster
    test_arr4d[3:5, 3:5, 3:5, 0] = -10  # negative cluster, very high intensity
    test_arr4d[5:6, 3:5, 3:5, 0] = 1  # cluster touching negative one

    # One-sided test where largest cluster doesn't have the highest mass
    true_size = 27
    true_mass = 39.992  # (8 vox * 5 intensity) - (8 vox * 0.001 thresh)
    test_size, test_mass = _utils._calculate_cluster_measures(
        test_arr4d,
        threshold=threshold,
        bin_struct=bin_struct,
        two_sided_test=False,
    )
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass

    # Two-sided test where negative cluster has the higher mass
    true_size = 27
    true_mass = 79.992  # (8 vox * 5 intensity) - (8 vox * 0.001 thresh)
    test_size, test_mass = _utils._calculate_cluster_measures(
        test_arr4d,
        threshold=threshold,
        bin_struct=bin_struct,
        two_sided_test=True,
    )
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass

    # Two-sided test with edge connectivity
    bin_struct = generate_binary_structure(3, 2)

    true_size = 28  # should include edge-connected single voxel cluster
    true_mass = 79.992  # (8 vox * 5 intensity) - (8 vox * 0.001 thresh)
    test_size, test_mass = _utils._calculate_cluster_measures(
        test_arr4d,
        threshold=threshold,
        bin_struct=bin_struct,
        two_sided_test=True,
    )
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass

    # Two-sided test with corner connectivity
    bin_struct = generate_binary_structure(3, 3)

    true_size = 29  # should include corner-connected single voxel cluster
    true_mass = 79.992  # (8 vox * 5 intensity) - (8 vox * 0.001 thresh)
    test_size, test_mass = _utils._calculate_cluster_measures(
        test_arr4d,
        threshold=threshold,
        bin_struct=bin_struct,
        two_sided_test=True,
    )
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass

    # Test on empty array
    test_arr4d = np.zeros((10, 10, 10, 1))
    true_size = 0
    true_mass = 0
    test_size, test_mass = _utils._calculate_cluster_measures(
        test_arr4d,
        threshold=threshold,
        bin_struct=bin_struct,
        two_sided_test=True,
    )
    assert test_size[0] == true_size
    assert test_mass[0] == true_mass


def test_t_score_with_covars_and_normalized_design_nocovar(random_state=0):
    """Test t-scores computation without covariates."""
    rng = check_random_state(random_state)

    # Normalized data
    n_samples = 50

    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2**2, 0))  # normalize

    # compute t-scores with nilearn routine
    t_val_own = _utils._t_score_with_covars_and_normalized_design(var1, var2)

    # compute t-scores with linalg or statsmodels
    t_val_alt = get_tvalue_with_alternative_library(var1, var2)
    assert_array_almost_equal(t_val_own, t_val_alt)


def test_t_score_with_covars_and_normalized_design_withcovar(random_state=0):
    """Test t-scores computation with covariates."""
    rng = check_random_state(random_state)

    # Normalized data
    n_samples = 50

    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)  # normalized
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2**2, 0))  # normalize
    covars = np.eye(n_samples, 3)  # covars is orthogonal
    covars[3] = -1  # covars is orthogonal to var1
    covars = _utils._orthonormalize_matrix(covars)

    # nilearn t-score
    own_score = _utils._t_score_with_covars_and_normalized_design(
        var1,
        var2,
        covars,
    )

    # compute t-scores with linalg or statmodels
    ref_score = get_tvalue_with_alternative_library(var1, var2, covars)
    assert_array_almost_equal(own_score, ref_score)
