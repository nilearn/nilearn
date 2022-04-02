import math

import numpy as np

from nilearn.mass_univariate._utils import _null_to_p


def test_null_to_p_float():
    """Test _null_to_p with single float input."""
    null = [-10, -9, -9, -3, -2, -1, -1, 0, 1, 1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9]

    # Left/lower-tailed
    assert math.isclose(_null_to_p(9, null, alternative='smaller'), 0.95)
    assert math.isclose(_null_to_p(-9, null, alternative='smaller'), 0.15)
    assert math.isclose(_null_to_p(0, null, alternative='smaller'), 0.4)

    # Right/upper-tailed
    assert math.isclose(_null_to_p(9, null, alternative='larger'), 0.05)
    assert math.isclose(_null_to_p(-9, null, alternative='larger'), 0.95)
    assert math.isclose(_null_to_p(0, null, alternative='larger'), 0.65)

    # Test that 1/n(null) is preserved with extreme values
    nulldist = np.random.normal(size=10000)
    assert math.isclose(
        _null_to_p(20, nulldist, alternative='two-sided'),
        1 / 10000,
    )
    assert math.isclose(
        _null_to_p(20, nulldist, alternative='smaller'),
        1 - 1 / 10000,
    )

    # Two-tailed
    assert math.isclose(_null_to_p(0, null, alternative='two-sided'), 0.95)
    result = _null_to_p(9, null, alternative='two-sided')
    assert result == _null_to_p(-9, null, alternative='two-sided')
    assert math.isclose(result, 0.2)
    result = _null_to_p(10, null, alternative='two-sided')
    assert result == _null_to_p(-10, null, alternative='two-sided')
    assert math.isclose(result, 0.05)

    # Still 0.05 because minimum valid p-value is 1 / len(null)
    result = _null_to_p(20, null, alternative='two-sided')
    assert result == _null_to_p(-20, null, alternative='two-sided')
    assert math.isclose(result, 0.05)


def test_null_to_p_array():
    """Test _null_to_p with 1d array input."""
    N = 10000
    nulldist = np.random.normal(size=N)
    t = np.sort(np.random.normal(size=N))
    p = np.sort(_null_to_p(t, nulldist))
    assert p.shape == (N,)
    assert (p < 1).all()
    assert (p > 0).all()
    # Resulting distribution should be roughly uniform
    assert np.abs(p.mean() - 0.5) < 0.02
    assert np.abs(p.var() - 1 / 12) < 0.02
