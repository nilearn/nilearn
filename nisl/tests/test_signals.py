"""
Test the signals module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import numpy as np
from nose.tools import assert_true, assert_false, assert_raises

from .. import signals
from ..signals import clean


def generate_signals(feature_number=17,
                     confound_number=5,
                     length=41):
    """Generate test signals."""
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


def test_standardize():
    randgen = np.random.RandomState(0)
    n_features = 10
    n_samples = 17

    # Create random signals with offsets
    a = randgen.random_sample((n_samples, n_features))
    a += np.linspace(0, 2., n_features)

    # transpose array to fit _standardize input.
    a = a.T
    b = signals._standardize(a)
    np.testing.assert_almost_equal((b ** 2).sum(axis=-1), np.ones(n_features))
    np.testing.assert_almost_equal(b.sum(axis=-1), np.zeros(n_features))


# The test is inspired from scipy docstring of detrend function
def test_clean_detrending():
    randgen = np.random.RandomState(0)
    npoints = 1e3
    noise = randgen.randn(npoints)
    x = 2 * np.linspace(0, 1, npoints) + noise
    x_detrended = signals.clean([x], standardize=False, detrend=True,
                                low_pass=None)[0]
    x_undetrended = signals.clean([x], standardize=False, low_pass=None)[0]
    assert_false((x_undetrended - noise).max() < 0.06)
    assert_true((x_detrended - noise).max() < 0.06)


def test_clean_confounds():
    signals, noises, confounds = generate_signals(feature_number=41,
                                                  confound_number=5, length=45)
    # No signal: output must be zero.
    eps = np.finfo(np.float).eps
    noises1 = noises.copy()
    cleaned_signals = clean(noises, confounds=confounds,
                            detrend=True, standardize=True)
    print(abs(cleaned_signals).max())
    assert(abs(cleaned_signals).max() < 15. * eps)

    np.testing.assert_almost_equal(noises, noises1, decimal=12)

    # With signal: output must be orthogonal to confounds
    cleaned_signals = clean(signals + noises, confounds=confounds,
                            detrend=True, standardize=True)
    print(abs(np.dot(confounds.T, cleaned_signals)).max())
    assert(abs(np.dot(confounds.T, cleaned_signals)).max() < 15. * eps)

    # TODO: Test with confounds from a file


def test_clean_frequencies():
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = [sx1, sx2]
    assert_true(clean(sx, standardize=False, high_pass=0.002, low_pass=None)
                .max() > 0.1)
    assert_true(clean(sx, standardize=False, high_pass=0.2, low_pass=None)
                .max() < 0.01)
    assert_true(clean(sx, standardize=False, low_pass=0.01).max() > 0.9)
    assert_true(clean(sx, standardize=False, low_pass=0.0005).max() < 0.01)
    assert_raises(ValueError, clean, sx, low_pass=0.4, high_pass=0.5)
