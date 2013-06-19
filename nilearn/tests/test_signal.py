"""
Test the signals module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os.path

import numpy as np
from nose.tools import assert_true, assert_false, assert_raises

# Use nisignal here to avoid name collisions (using nilearn.signal is
# not possible)
from .. import signal as nisignal
from ..signal import clean
import scipy.signal


def generate_signals(feature_number=17, n_confounds=5, length=41,
                     same_variance=True, order="C"):
    """Generate test signals.

    Returned signals have no trends at all (to machine precision).
    """
    rand_gen = np.random.RandomState(0)

    # Generate random confounds
    confounds_shape = (length, n_confounds)
    confounds = np.ndarray(confounds_shape, order=order)
    confounds[...] = rand_gen.randn(*confounds_shape)
    confounds[...] = scipy.signal.detrend(confounds, axis=0)

    # Compute noise based on confounds, with random factors
    factors = rand_gen.randn(n_confounds, feature_number)
    noises_shape = (length, feature_number)
    noises = np.ndarray(noises_shape, order=order)
    noises[...] = np.dot(confounds, factors)
    noises[...] = scipy.signal.detrend(noises, axis=0)

    # Generate random signals with random amplitudes
    signals_shape = noises_shape
    signals = np.ndarray(signals_shape, order=order)
    if same_variance:
        signals[...] = rand_gen.randn(*signals_shape)
    else:
        signals[...] = (4. * abs(rand_gen.randn(signals_shape[1])) + 0.5
                        ) * rand_gen.randn(*signals_shape)

    signals[...] = scipy.signal.detrend(signals, axis=0)
    return signals, noises, confounds


def generate_trends(feature_number=17, length=41):
    """Generate linearly-varying signals, with zero mean.

    Returns
    =======
    trends (numpy.ndarray)
        shape: (length, feature_number)
    """
    rand_gen = np.random.RandomState(0)
    trends = scipy.signal.detrend(np.linspace(0, 1.0, length), type="constant")
    trends = np.repeat(np.atleast_2d(trends).T, feature_number, axis=1)
    factors = rand_gen.randn(feature_number)
    return trends * factors


def test_butterworth():
    rand_gen = np.random.RandomState(0)
    n_features = 20000
    n_samples = 100

    sampling = 100
    low_pass = 30
    high_pass = 10

    # Compare output for different options.
    # single timeseries
    data = rand_gen.randn(n_samples)
    data_original = data.copy()

    out_single = nisignal.butterworth(data, sampling,
                                       low_pass=low_pass, high_pass=high_pass,
                                       copy=True)
    np.testing.assert_almost_equal(data, data_original)
    nisignal.butterworth(data, sampling,
                          low_pass=low_pass, high_pass=high_pass,
                          copy=False, save_memory=True)
    np.testing.assert_almost_equal(out_single, data)

    # multiple timeseries
    data = rand_gen.randn(n_samples, n_features)
    data[:, 0] = data_original  # set first timeseries to previous data
    data_original = data.copy()

    out1 = nisignal.butterworth(data, sampling,
                                 low_pass=low_pass, high_pass=high_pass,
                                 copy=True)
    np.testing.assert_almost_equal(data, data_original)
    # check that multiple- and single-timeseries filtering do the same thing.
    np.testing.assert_almost_equal(out1[:, 0], out_single)
    nisignal.butterworth(data, sampling,
                          low_pass=low_pass, high_pass=high_pass,
                          copy=False)
    np.testing.assert_almost_equal(out1, data)


def test_standardize():
    rand_gen = np.random.RandomState(0)
    n_features = 10
    n_samples = 17

    # Create random signals with offsets
    a = rand_gen.random_sample((n_samples, n_features))
    a += np.linspace(0, 2., n_features)

    # transpose array to fit _standardize input.
    # Without trend removal
    b = nisignal._standardize(a, normalize=True)
    energies = (b ** 2).sum(axis=0)
    np.testing.assert_almost_equal(energies, np.ones(n_features))
    np.testing.assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # With trend removal
    a = np.atleast_2d(np.linspace(0, 2., n_features)).T
    b = nisignal._standardize(a, detrend=True, normalize=False)
    np.testing.assert_almost_equal(b, np.zeros(b.shape))


def test_detrend():
    """Test custom detrend implementation."""
    point_number = 703
    features = 17
    signals, _, _ = generate_signals(feature_number=features,
                                     length=point_number,
                                     same_variance=True)
    trends = generate_trends(feature_number=features, length=point_number)
    x = signals + trends + 1
    original = x.copy()

    # Mean removal only (out-of-place)
    detrended = nisignal._detrend(x, inplace=False, type="constant")
    assert_true(abs(detrended.mean(axis=0)).max()
                < 15. * np.finfo(np.float).eps)

    # out-of-place detrending. Use scipy as a reference implementation
    detrended = nisignal._detrend(x, inplace=False)
    detrended_scipy = scipy.signal.detrend(x, axis=0)

    # "x" must be left untouched
    np.testing.assert_almost_equal(original, x, decimal=14)
    assert_true(abs(detrended.mean(axis=0)).max() <
                15. * np.finfo(np.float).eps)
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    nisignal._detrend(x, inplace=True)
    assert_true(abs(x.mean(axis=0)).max() < 15. * np.finfo(np.float).eps)
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    np.testing.assert_almost_equal(x, signals, decimal=14)


# This test is inspired from scipy docstring of detrend function
def test_clean_detrending():
    point_number = 1000
    feature_number = 1
    signals, _, _ = generate_signals(feature_number=feature_number,
                                     length=point_number)
    trends = generate_trends(feature_number=feature_number,
                             length=point_number)
    x = signals + trends

    # This should remove trends
    x_detrended = nisignal.clean(x, standardize=False, detrend=True,
                                  low_pass=None, high_pass=None)
    np.testing.assert_almost_equal(x_detrended, signals, decimal=13)

    # This should does nothing
    x_undetrended = nisignal.clean(x, standardize=False, detrend=False,
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
    assert_raises(ValueError, clean, sx, low_pass=0.4, high_pass=0.5)


def test_clean_confounds():
    signals, noises, confounds = generate_signals(feature_number=41,
                                                  n_confounds=5, length=45)
    # No signal: output must be zero.
    eps = np.finfo(np.float).eps
    noises1 = noises.copy()
    cleaned_signals = nisignal.clean(noises, confounds=confounds,
                                      detrend=True, standardize=False)
    assert_true(abs(cleaned_signals).max() < 20. * eps)
    np.testing.assert_almost_equal(noises, noises1, decimal=12)

    # With signal: output must be orthogonal to confounds
    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                      detrend=True, standardize=False)
    assert_true(abs(np.dot(confounds.T, cleaned_signals)).max() < 20. * eps)

    # Test detrending. No trend should exist in the output.
    # Use confounds with a trend.
    temp = confounds.T
    temp += np.arange(confounds.shape[0])

    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                      detrend=False, standardize=False)
    coeffs = np.polyfit(np.arange(cleaned_signals.shape[0]),
                        cleaned_signals, 1)
    assert_true((abs(coeffs) > 1e-3).any())   # trends remain

    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                      detrend=True, standardize=False)
    coeffs = np.polyfit(np.arange(cleaned_signals.shape[0]),
                        cleaned_signals, 1)
    assert_true((abs(coeffs) < 10. * eps).all())  # trend removed

    # Test no-op
    input_signals = 10 * signals
    cleaned_signals = nisignal.clean(input_signals, detrend=False,
                                      standardize=False)
    np.testing.assert_almost_equal(cleaned_signals, input_signals)

    cleaned_signals = nisignal.clean(input_signals, detrend=False,
                                      standardize=True)
    np.testing.assert_almost_equal(cleaned_signals.var(axis=0),
                                   np.ones(cleaned_signals.shape[1]))

    # Test with confounds read from a file. Smoke test only (result has
    # no meaning).
    current_dir = os.path.split(__file__)[0]

    signals, _, confounds = generate_signals(feature_number=41,
                                                  n_confounds=3, length=20)
    filename1 = os.path.join(current_dir, "test_files", "spm_confounds.txt")
    filename2 = os.path.join(current_dir, "test_files",
                             "confounds_with_header.csv")

    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=filename1)
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=filename2)
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=confounds[:, 1])

    # Use a list containing two filenames, a 2D array and a 1D array
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=[filename1, confounds[:, 0:2],
                              filename2, confounds[:, 2]])

    # Test error handling
    assert_raises(TypeError, nisignal.clean, signals, confounds=1)
    assert_raises(ValueError, nisignal.clean, signals, confounds=np.zeros(2))
    assert_raises(ValueError, nisignal.clean, signals,
                  confounds=np.zeros((2, 2)))
    assert_raises(ValueError, nisignal.clean, signals,
                  confounds=np.zeros((2, 3, 4)))
    assert_raises(ValueError, nisignal.clean, signals[:-1, :],
                  confounds=filename1)
    assert_raises(TypeError, nisignal.clean, signals,
                  confounds=[None])


def test_high_variance_confounds():
    # C and F order might take different paths in the function. Check that the
    # result is identical.
    feature_number = 1001
    length = 20
    n_confounds = 5
    seriesC, _, _ = generate_signals(feature_number=feature_number,
                                     length=length, order="C")
    seriesF, _, _ = generate_signals(feature_number=feature_number,
                                     length=length, order="F")

    np.testing.assert_almost_equal(seriesC, seriesF, decimal=13)
    outC = nisignal.high_variance_confounds(seriesC, n_confounds=n_confounds,
                                             detrend=False)
    outF = nisignal.high_variance_confounds(seriesF, n_confounds=n_confounds,
                                             detrend=False)
    np.testing.assert_almost_equal(outC, outF, decimal=13)

    # Result must not be influenced by global scaling
    seriesG = 2 * seriesC
    outG = nisignal.high_variance_confounds(seriesG, n_confounds=n_confounds,
                                             detrend=False)
    np.testing.assert_almost_equal(outC, outG, decimal=13)
    assert(outG.shape == (length, n_confounds))

    # Changing percentile changes the result
    seriesG = seriesC
    outG = nisignal.high_variance_confounds(seriesG, percentile=2.,
                                             n_confounds=n_confounds,
                                             detrend=False)
    assert_raises(AssertionError, np.testing.assert_almost_equal,
                  outC, outG, decimal=13)
    assert(outG.shape == (length, n_confounds))

    # Check shape of output
    out = nisignal.high_variance_confounds(seriesG, n_confounds=7,
                                            detrend=False)
    assert(out.shape == (length, 7))

    # TODO: any other ideas?

    # Adding a trend and detrending should give same results as with no trend.
    seriesG = seriesC
    trends = generate_trends(feature_number=feature_number, length=length)
    seriesGt = seriesG + trends

    outG = nisignal.high_variance_confounds(seriesG, detrend=False,
                                             n_confounds=n_confounds)
    outGt = nisignal.high_variance_confounds(seriesGt, detrend=True,
                                             n_confounds=n_confounds)
    np.testing.assert_almost_equal(outG, outGt, decimal=10)
