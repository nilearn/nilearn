import math

import numpy as np
import pytest
from scipy import ndimage

from nilearn.mass_univariate import _utils


def test_null_to_p_float():
    """Test _null_to_p with single float input."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Left/lower-tailed
    assert math.isclose(
        _utils._null_to_p(9, null, alternative='smaller'),
        0.95,
    )
    assert math.isclose(
        _utils._null_to_p(-9, null, alternative='smaller'),
        0.15,
    )
    assert math.isclose(_utils._null_to_p(0, null, alternative='smaller'), 0.4)

    # Right/upper-tailed
    assert math.isclose(_utils._null_to_p(9, null, alternative='larger'), 0.05)
    assert math.isclose(
        _utils._null_to_p(-9, null, alternative='larger'),
        0.95,
    )
    assert math.isclose(_utils._null_to_p(0, null, alternative='larger'), 0.65)

    # Test that 1/n(null) is preserved with extreme values
    nulldist = np.random.normal(size=10000)
    assert math.isclose(
        _utils._null_to_p(20, nulldist, alternative='two-sided'),
        1 / 10000,
    )
    assert math.isclose(
        _utils._null_to_p(20, nulldist, alternative='smaller'),
        1 - 1 / 10000,
    )

    # Two-tailed
    assert math.isclose(
        _utils._null_to_p(0, null, alternative='two-sided'),
        0.95,
    )
    result = _utils._null_to_p(9, null, alternative='two-sided')
    assert result == _utils._null_to_p(-9, null, alternative='two-sided')
    assert math.isclose(result, 0.2)
    result = _utils._null_to_p(10, null, alternative='two-sided')
    assert result == _utils._null_to_p(-10, null, alternative='two-sided')
    assert math.isclose(result, 0.05)

    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = _utils._null_to_p(20, null, alternative='two-sided')
    assert result == _utils._null_to_p(-20, null, alternative='two-sided')
    assert math.isclose(result, 0.05)

    with pytest.raises(ValueError):
        _utils._null_to_p(9, null, alternative='raise')


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


def test__calculate_cluster_measures():
    """Test _calculate_cluster_measures."""
    threshold = 0.001
    bin_struct = ndimage.generate_binary_structure(3, 1)

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
    bin_struct = ndimage.generate_binary_structure(3, 2)

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
    bin_struct = ndimage.generate_binary_structure(3, 3)

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
