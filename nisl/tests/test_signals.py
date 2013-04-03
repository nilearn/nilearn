"""
Test the signals module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import numpy as np
from nose.tools import assert_true, assert_false, assert_raises

from .. import signals as nisignals
from ..signals import clean
import scipy.signal


def generate_signals(feature_number=17,
                     confound_number=5,
                     length=41):
    """Generate test signals.

    Returned signals have no trends at all (to machine precision).
    """
    randgen = np.random.RandomState(0)

    # Generate random confounds
    confounds = randgen.randn(length, confound_number)
    confounds = scipy.signal.detrend(confounds, axis=0)

    # Compute noise based on confounds, with random factors
    factors = randgen.randn(confound_number, feature_number)
    noises = np.dot(confounds, factors)
    noises = scipy.signal.detrend(noises, axis=0)

    # Generate random signal
    signals = 0.5 * randgen.randn(length, feature_number)
    signals = scipy.signal.detrend(signals, axis=0)
    return signals, noises, confounds


def generate_trends(feature_number=17, length=41):
    """Generate linearly-varying signals, with zero mean.

    Returns
    =======
    trends (numpy.ndarray)
        shape: (length, feature_number)
    """
    randgen = np.random.RandomState(0)
    trends = scipy.signal.detrend(np.linspace(0, 1.0, length), type="constant")
    trends = np.repeat(np.atleast_2d(trends).T, feature_number, axis=1)
    factors = randgen.randn(feature_number)
    return trends * factors


def test_standardize():
    randgen = np.random.RandomState(0)
    n_features = 10
    n_samples = 17

    # Create random signals with offsets
    a = randgen.random_sample((n_samples, n_features))
    a += np.linspace(0, 2., n_features)

    # transpose array to fit _standardize input.
    a = a.T
    b = nisignals._standardize(a)
    np.testing.assert_almost_equal((b ** 2).sum(axis=-1), np.ones(n_features))
    np.testing.assert_almost_equal(b.sum(axis=-1), np.zeros(n_features))


def test_detrend():
    """Test custom detrend implementation."""
    point_number = 703
    features = 17
    signals, _, _ = generate_signals(feature_number=features,
                                     length=point_number)
    trends = generate_trends(feature_number=features, length=point_number)
    x = signals + trends
    original = x.copy()

    # out-of-place detrending. Use scipy as a reference implementation
    detrended = nisignals._detrend(x, inplace=False)
    detrended_scipy = scipy.signal.detrend(x, axis=0)

    # "x" must be left untouched
    np.testing.assert_almost_equal(original, x, decimal=14)
    assert(abs(detrended.mean(axis=0)).max() < np.finfo(np.float).eps)
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    nisignals._detrend(x, inplace=True)
    assert(abs(x.mean(axis=0)).max() < np.finfo(np.float).eps)
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    np.testing.assert_almost_equal(x, signals, decimal=14)


# This test is inspired from scipy docstring of detrend function
def test_clean_detrending():
    point_number = 1000
    signals, noises, _ = generate_signals(feature_number=1,
                                          length=point_number)
    trend = np.atleast_2d(np.linspace(0, 2., point_number)).T
    x = signals + trend
    x_detrended = nisignals.clean(x, standardize=False, detrend=True,
                                  low_pass=None, high_pass=None)
#    print (abs(x_detrended - signals).max())
    assert_true(abs(x_detrended - signals).max() < 0.06)

    x_undetrended = nisignals.clean(x, standardize=False, detrend=False,
                                    low_pass=None, high_pass=None)
    assert_false(abs(x_undetrended - signals).max() < 0.06)


def test_clean_frequencies():
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T
    assert_true(clean(sx, standardize=False, high_pass=0.002, low_pass=None)
                .max() > 0.1)
    assert_true(clean(sx, standardize=False, high_pass=0.2, low_pass=None)
                .max() < 0.01)
    assert_true(clean(sx, standardize=False, low_pass=0.01).max() > 0.9)
    assert_true(clean(sx, standardize=False, low_pass=0.0005).max() < 0.01)
    assert_raises(ValueError, clean, sx, low_pass=0.4, high_pass=0.5)


def test_clean_confounds():
    signals, noises, confounds = generate_signals(feature_number=41,
                                                  confound_number=5, length=45)
    # No signal: output must be zero.
    eps = np.finfo(np.float).eps
    noises1 = noises.copy()
    cleaned_signals = nisignals.clean(noises, confounds=confounds,
                                      detrend=True, standardize=True)
    print(abs(cleaned_signals).max())
    assert(abs(cleaned_signals).max() < 15. * eps)

    np.testing.assert_almost_equal(noises, noises1, decimal=12)

    # With signal: output must be orthogonal to confounds
    cleaned_signals = nisignals.clean(signals + noises, confounds=confounds,
                                      detrend=True, standardize=True)
    print(abs(np.dot(confounds.T, cleaned_signals)).max())
    assert(abs(np.dot(confounds.T, cleaned_signals)).max() < 15. * eps)

    # TODO: Test with confounds read from a file
