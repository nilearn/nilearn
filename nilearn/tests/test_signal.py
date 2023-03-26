"""
Test the signals module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os.path
import warnings
from nilearn.version import _compare_version

import numpy as np
import pytest

# Use nisignal here to avoid name collisions (using nilearn.signal is
# not possible)
from nilearn import signal as nisignal
from nilearn.signal import clean
from pandas import read_csv
import scipy.signal


def generate_signals(n_features=17, n_confounds=5, length=41,
                     same_variance=True, order="C"):
    """Generate test signals.

    All returned signals have no trends at all (to machine precision).

    Parameters
    ----------
    n_features, n_confounds : int, optional
        respectively number of features to generate, and number of confounds
        to use for generating noise signals.

    length : int, optional
        number of samples for every signal.

    same_variance : bool, optional
        if True, every column of "signals" have a unit variance. Otherwise,
        a random amplitude is applied.

    order : "C" or "F"
        gives the contiguousness of the output arrays.

    Returns
    -------
    signals : numpy.ndarray, shape (length, n_features)
        unperturbed signals.

    noises : numpy.ndarray, shape (length, n_features)
        confound-based noises. Each column is a signal obtained by linear
        combination of all confounds signals (below). The coefficients in
        the linear combination are also random.

    confounds : numpy.ndarray, shape (length, n_confounds)
        random signals used as confounds.
    """
    rng = np.random.RandomState(42)

    # Generate random confounds
    confounds_shape = (length, n_confounds)
    confounds = np.ndarray(confounds_shape, order=order)
    confounds[...] = rng.standard_normal(size=confounds_shape)
    confounds[...] = scipy.signal.detrend(confounds, axis=0)

    # Compute noise based on confounds, with random factors
    factors = rng.standard_normal(size=(n_confounds, n_features))
    noises_shape = (length, n_features)
    noises = np.ndarray(noises_shape, order=order)
    noises[...] = np.dot(confounds, factors)
    noises[...] = scipy.signal.detrend(noises, axis=0)

    # Generate random signals with random amplitudes
    signals_shape = noises_shape
    signals = np.ndarray(signals_shape, order=order)
    if same_variance:
        signals[...] = rng.standard_normal(size=signals_shape)
    else:
        signals[...] = (
            4.0 * abs(rng.standard_normal(size=signals_shape[1])) + 0.5
        ) * rng.standard_normal(size=signals_shape)

    signals[...] = scipy.signal.detrend(signals, axis=0)
    return signals, noises, confounds


def generate_trends(n_features=17, length=41):
    """Generate linearly-varying signals, with zero mean.

    Parameters
    ----------
    n_features, length : int
        respectively number of signals and number of samples to generate.

    Returns
    -------
    trends : numpy.ndarray, shape (length, n_features)
        output signals, one per column.
    """
    rng = np.random.RandomState(42)
    trends = scipy.signal.detrend(np.linspace(0, 1.0, length), type="constant")
    trends = np.repeat(np.atleast_2d(trends).T, n_features, axis=1)
    factors = rng.standard_normal(size=n_features)
    return trends * factors


def generate_signals_plus_trends(n_features=17, n_samples=41):

    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples)
    trends = generate_trends(n_features=n_features,
                             length=n_samples)
    return signals + trends


def test_butterworth():
    rng = np.random.RandomState(42)
    n_features = 20000
    n_samples = 100

    sampling = 100
    low_pass = 30
    high_pass = 10

    # Compare output for different options.
    # single timeseries
    data = rng.standard_normal(size=n_samples)
    data_original = data.copy()
    out_single = nisignal.butterworth(data, sampling,
                                      low_pass=low_pass, high_pass=high_pass,
                                      copy=True)
    np.testing.assert_almost_equal(data, data_original)
    nisignal.butterworth(data, sampling,
                         low_pass=low_pass, high_pass=high_pass,
                         copy=False)
    np.testing.assert_almost_equal(out_single, data)
    np.testing.assert_(id(out_single) != id(data))

    # multiple timeseries
    data = rng.standard_normal(size=(n_samples, n_features))
    data[:, 0] = data_original  # set first timeseries to previous data
    data_original = data.copy()

    out1 = nisignal.butterworth(data, sampling,
                                low_pass=low_pass, high_pass=high_pass,
                                copy=True)
    np.testing.assert_almost_equal(data, data_original)
    np.testing.assert_(id(out1) != id(data_original))

    # check that multiple- and single-timeseries filtering do the same thing.
    np.testing.assert_almost_equal(out1[:, 0], out_single)
    nisignal.butterworth(data, sampling,
                         low_pass=low_pass, high_pass=high_pass,
                         copy=False)
    np.testing.assert_almost_equal(out1, data)

    # Test nyquist frequency clipping, issue #482
    out1 = nisignal.butterworth(data, sampling,
                                low_pass=50.,
                                copy=True)
    out2 = nisignal.butterworth(data, sampling,
                                low_pass=80.,  # Greater than nyq frequency
                                copy=True)
    np.testing.assert_almost_equal(out1, out2)
    np.testing.assert_(id(out1) != id(out2))

    # Test check for equal values in critical frequencies
    sampling = 1
    low_pass = 2
    high_pass = 1
    with pytest.warns(
        UserWarning,
        match=(
            'Signals are returned unfiltered because '
            'band-pass critical frequencies are equal. '
            'Please check that inputs for sampling_rate, '
            'low_pass, and high_pass are valid.'
        ),
    ):
        out = nisignal.butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert (out == data).all()

    # Test check for frequency higher than allowed (>=Nyquist).
    # The frequency should be modified and the filter should be run.
    sampling = 1
    high_pass = 0.01
    low_pass = 0.5
    with pytest.warns(
        UserWarning,
        match='The frequency specified for the low pass filter is too high',
    ):
        out = nisignal.butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert not np.array_equal(data, out)

    # Test check for frequency lower than allowed (<0).
    # The frequency should be modified and the filter should be run.
    sampling = 1
    high_pass = -1
    low_pass = 0.4
    with pytest.warns(
        UserWarning,
        match='The frequency specified for the high pass filter is too low',
    ):
        out = nisignal.butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert not np.array_equal(data, out)

    # Test check for high-pass frequency higher than low-pass frequency.
    # An error should be raised.
    sampling = 1
    high_pass = 0.2
    low_pass = 0.1
    with pytest.raises(
        ValueError,
        match=(
            r'High pass cutoff frequency \([0-9.]+\) is greater than or '
            r'equal to low pass filter frequency \([0-9.]+\)\.'
        ),
    ):
        nisignal.butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )


def test_standardize():
    rng = np.random.RandomState(42)
    n_features = 10
    n_samples = 17

    # Create random signals with offsets
    a = rng.random_sample((n_samples, n_features))
    a += np.linspace(0, 2., n_features)

    # Test raise error when strategy is not valid option
    with pytest.raises(ValueError, match="no valid standardize strategy"):
        nisignal._standardize(a, standardize="foo")

    # test warning for strategy that will be removed
    with pytest.warns(FutureWarning, match="default strategy for standardize"):
        nisignal._standardize(a, standardize="zscore")

    # transpose array to fit _standardize input.
    # Without trend removal
    b = nisignal._standardize(a, standardize='zscore')
    stds = np.std(b)
    np.testing.assert_almost_equal(stds, np.ones(n_features))
    np.testing.assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # Repeating test above but for new correct strategy
    b = nisignal._standardize(a, standardize='zscore_sample')
    stds = np.std(b)
    np.testing.assert_almost_equal(stds, np.ones(n_features), decimal=1)
    np.testing.assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # With trend removal
    a = np.atleast_2d(np.linspace(0, 2., n_features)).T
    b = nisignal._standardize(a, detrend=True, standardize=False)
    np.testing.assert_almost_equal(b, np.zeros(b.shape))

    b = nisignal._standardize(a, detrend=True, standardize="zscore_sample")
    np.testing.assert_almost_equal(b, np.zeros(b.shape))

    length_1_signal = np.atleast_2d(np.linspace(0, 2., n_features))
    np.testing.assert_array_equal(length_1_signal,
                                  nisignal._standardize(length_1_signal,
                                                        standardize='zscore'))

    # Repeating test above but for new correct strategy
    length_1_signal = np.atleast_2d(np.linspace(0, 2., n_features))
    np.testing.assert_array_equal(
        length_1_signal,
        nisignal._standardize(length_1_signal, standardize="zscore_sample")
    )


def test_detrend():
    """Test custom detrend implementation."""
    point_number = 703
    features = 17
    signals, _, _ = generate_signals(n_features=features,
                                     length=point_number,
                                     same_variance=True)
    trends = generate_trends(n_features=features, length=point_number)
    x = signals + trends + 1
    original = x.copy()

    # Mean removal only (out-of-place)
    detrended = nisignal._detrend(x, inplace=False, type="constant")
    assert (abs(detrended.mean(axis=0)).max()
                < 15. * np.finfo(np.float64).eps)

    # out-of-place detrending. Use scipy as a reference implementation
    detrended = nisignal._detrend(x, inplace=False)
    detrended_scipy = scipy.signal.detrend(x, axis=0)

    # "x" must be left untouched
    np.testing.assert_almost_equal(original, x, decimal=14)
    assert abs(detrended.mean(axis=0)).max() < 15. * np.finfo(np.float64).eps
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    nisignal._detrend(x, inplace=True)
    assert abs(x.mean(axis=0)).max() < 15. * np.finfo(np.float64).eps
    # for this to work, there must be no trends at all in "signals"
    np.testing.assert_almost_equal(detrended_scipy, detrended, decimal=14)
    np.testing.assert_almost_equal(x, signals, decimal=14)

    length_1_signal = x[0]
    length_1_signal = length_1_signal[np.newaxis, :]
    np.testing.assert_array_equal(length_1_signal,
                                  nisignal._detrend(length_1_signal))

    # Mean removal on integers
    detrended = nisignal._detrend(x.astype(np.int64), inplace=True,
                                  type="constant")
    assert (abs(detrended.mean(axis=0)).max() <
                20. * np.finfo(np.float64).eps)


def test_mean_of_squares():
    """Test _mean_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples,
                                     same_variance=True)
    # Reference computation
    var1 = np.copy(signals)
    var1 **= 2
    var1 = var1.mean(axis=0)

    var2 = nisignal._mean_of_squares(signals)

    np.testing.assert_almost_equal(var1, var2)


def test_row_sum_of_squares():
    """Test _row_sum_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples,
                                     same_variance=True)
    # Reference computation
    var1 = signals ** 2
    var1 = var1.sum(axis=0)

    var2 = nisignal._row_sum_of_squares(signals)

    np.testing.assert_almost_equal(var1, var2)


# This test is inspired from Scipy docstring of detrend function
def test_clean_detrending():
    n_samples = 21
    n_features = 501  # Must be higher than 500
    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples)
    trends = generate_trends(n_features=n_features,
                             length=n_samples)
    x = signals + trends
    x_orig = x.copy()

    # if NANs, data out should be False with ensure_finite=True
    y = signals + trends
    y[20, 150] = np.nan
    y[5, 500] = np.nan
    y[15, 14] = np.inf
    y_orig = y.copy()

    y_clean = nisignal.clean(y, ensure_finite=True)
    assert np.any(np.isfinite(y_clean)), True
    # clean should not modify inputs
    # using assert_almost_equal instead of array_equal due to NaNs
    np.testing.assert_almost_equal(y_orig, y, decimal=13)

    # test boolean is not given to signal.clean
    pytest.raises(TypeError, nisignal.clean, x, low_pass=False)
    pytest.raises(TypeError, nisignal.clean, x, high_pass=False)

    # This should remove trends
    x_detrended = nisignal.clean(x, standardize=False, detrend=True,
                                 low_pass=None, high_pass=None)
    np.testing.assert_almost_equal(x_detrended, signals, decimal=13)
    # clean should not modify inputs
    assert np.array_equal(x_orig, x)

    # This should do nothing
    x_undetrended = nisignal.clean(x, standardize=False, detrend=False,
                                   low_pass=None, high_pass=None)
    assert not abs(x_undetrended - signals).max() < 0.06
    # clean should not modify inputs
    assert np.array_equal(x_orig, x)


def test_clean_t_r():
    """Different TRs must produce different results after butterworth filtering"""
    rng = np.random.RandomState(42)
    n_samples = 34
    # n_features  Must be higher than 500
    n_features = 501
    x_orig = generate_signals_plus_trends(n_features=n_features,
                                          n_samples=n_samples)
    random_tr_list1 = np.round(rng.uniform(size=3) * 10, decimals=2)
    random_tr_list2 = np.round(rng.uniform(size=3) * 10, decimals=2)
    for tr1, tr2 in zip(random_tr_list1, random_tr_list2):
        low_pass_freq_list = tr1 * np.array([1.0 / 100, 1.0 / 110])
        high_pass_freq_list = tr1 * np.array([1.0 / 210, 1.0 / 190])
        for low_cutoff, high_cutoff in zip(low_pass_freq_list,
                                           high_pass_freq_list):
            det_one_tr = nisignal.clean(x_orig, t_r=tr1, low_pass=low_cutoff,
                                        high_pass=high_cutoff)
            det_diff_tr = nisignal.clean(x_orig, t_r=tr2, low_pass=low_cutoff,
                                         high_pass=high_cutoff)

            if not np.isclose(tr1, tr2, atol=0.3):
                msg = ('results do not differ for different TRs: {} and {} '
                       'at cutoffs: low_pass={}, high_pass={} '
                       'n_samples={}, n_features={}'.format(
                           tr1, tr2, low_cutoff, high_cutoff,
                           n_samples, n_features))
                np.testing.assert_(np.any(np.not_equal(det_one_tr,
                                                       det_diff_tr)),
                                   msg)
                del det_one_tr, det_diff_tr


def test_clean_kwargs():
    """Providing kwargs to clean should change the filtered results."""
    n_samples = 34
    n_features = 501
    x_orig = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )
    kwargs = [
        {
            "butterworth__padtype": "even",
            "butterworth__padlen": 10,
            "butterworth__order": 3,
        },
        {
            "butterworth__padtype": None,
            "butterworth__padlen": None,
            "butterworth__order": 1,
        },
        {
            "butterworth__padtype": "constant",
            "butterworth__padlen": 20,
            "butterworth__order": 10,
        },
    ]
    # Base result
    t_r, high_pass, low_pass = 0.8, 0.01, 0.08
    base_filtered = nisignal.clean(
        x_orig, t_r=t_r, low_pass=low_pass, high_pass=high_pass
    )
    for kwarg_set in kwargs:
        test_filtered = nisignal.clean(
            x_orig,
            t_r=t_r,
            low_pass=low_pass,
            high_pass=high_pass,
            **kwarg_set,
        )
        # Check that results are **not** the same.
        np.testing.assert_(np.any(np.not_equal(
            base_filtered, test_filtered
        )))


def test_clean_frequencies():
    '''Using butterworth method.'''
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T
    sx_orig = sx.copy()
    assert clean(sx, standardize=False, high_pass=0.002, low_pass=None,
                      t_r=2.5).max() > 0.1
    assert clean(sx, standardize=False, high_pass=0.2, low_pass=None,
                      t_r=2.5) .max() < 0.01
    assert clean(sx, standardize=False, low_pass=0.01, t_r=2.5).max() > 0.9
    pytest.raises(ValueError, clean, sx, low_pass=0.4, high_pass=0.5, t_r=2.5)

    # clean should not modify inputs
    sx_cleaned = clean(sx, standardize=False, detrend=False, low_pass=0.2, t_r=2.5)
    assert np.array_equal(sx_orig, sx)


def test_clean_runs():
    n_samples = 21
    n_features = 501  # Must be higher than 500
    signals, _, confounds = generate_signals(n_features=n_features,
                                     length=n_samples)
    trends = generate_trends(n_features=n_features,
                             length=n_samples)
    x = signals + trends
    x_orig = x.copy()
    # Create run info
    runs = np.ones(n_samples)
    runs[0:n_samples // 2] = 0
    x_detrended = nisignal.clean(x, confounds=confounds, standardize=False, detrend=True,
                                 low_pass=None, high_pass=None,
                                 runs=runs)
    # clean should not modify inputs
    assert np.array_equal(x_orig, x)

    # check the runs are individually cleaned
    x_run1 = nisignal.clean(x[0:n_samples // 2, :],
                            confounds=confounds[0:n_samples // 2, :],
                            standardize=False, detrend=True,
                            low_pass=None, high_pass=None)
    assert np.array_equal(x_run1, x_detrended[0:n_samples // 2, :])


def test_clean_confounds():
    signals, noises, confounds = generate_signals(n_features=41,
                                                  n_confounds=5, length=45)
    # No signal: output must be zero.
    eps = np.finfo(np.float64).eps
    noises1 = noises.copy()
    cleaned_signals = nisignal.clean(noises, confounds=confounds,
                                     detrend=True, standardize=False)
    assert abs(cleaned_signals).max() < 100. * eps
    # clean should not modify inputs
    assert np.array_equal(noises, noises1)

    # With signal: output must be orthogonal to confounds
    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                     detrend=False, standardize=True)
    assert abs(np.dot(confounds.T, cleaned_signals)).max() < 1000. * eps

    # Same output when a constant confound is added
    confounds1 = np.hstack((np.ones((45, 1)), confounds))
    cleaned_signals1 = nisignal.clean(signals + noises, confounds=confounds1,
                                      detrend=False, standardize=True)
    np.testing.assert_almost_equal(cleaned_signals1, cleaned_signals)

    # Test detrending. No trend should exist in the output.
    # Use confounds with a trend.
    temp = confounds.T
    temp += np.arange(confounds.shape[0])

    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                     detrend=False, standardize=False)
    coeffs = np.polyfit(np.arange(cleaned_signals.shape[0]),
                        cleaned_signals, 1)
    assert (abs(coeffs) > 1e-3).any()   # trends remain

    cleaned_signals = nisignal.clean(signals + noises, confounds=confounds,
                                     detrend=True, standardize=False)
    coeffs = np.polyfit(np.arange(cleaned_signals.shape[0]),
                        cleaned_signals, 1)
    assert (abs(coeffs) < 1000. * eps).all()  # trend removed

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

    signals, _, confounds = generate_signals(n_features=41,
                                             n_confounds=3, length=20)
    filename1 = os.path.join(current_dir, "data", "spm_confounds.txt")
    filename2 = os.path.join(current_dir, "data",
                             "confounds_with_header.csv")

    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=filename1)
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=filename2)
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=confounds[:, 1])

    # test with confounds as a pandas DataFrame
    confounds_df = read_csv(filename2, sep='\t')
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=confounds_df.values)
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=confounds_df)

    # test array-like signals
    list_signal = signals.tolist()
    nisignal.clean(list_signal)

    # Use a list containing two filenames, a 2D array and a 1D array
    nisignal.clean(signals, detrend=False, standardize=False,
                   confounds=[filename1, confounds[:, 0:2],
                              filename2, confounds[:, 2]])

    # Test error handling
    pytest.raises(TypeError, nisignal.clean, signals, confounds=1)
    pytest.raises(ValueError, nisignal.clean, signals, confounds=np.zeros(2))
    pytest.raises(ValueError, nisignal.clean, signals,
                  confounds=np.zeros((2, 2)))
    pytest.raises(ValueError, nisignal.clean, signals,
                  confounds=np.zeros((2, 3, 4)))
    pytest.raises(ValueError, nisignal.clean, signals[:-1, :],
                  confounds=filename1)
    pytest.raises(TypeError, nisignal.clean, signals,
                  confounds=[None])
    error_msg = pytest.raises(ValueError, nisignal.clean, signals, filter='cosine',
                              t_r=None, high_pass=0.008)
    assert "t_r='None'" in str(error_msg.value)
    pytest.raises(ValueError, nisignal.clean, signals, t_r=None,
                  low_pass=.01)  # using butterworth filter here
    pytest.raises(ValueError, nisignal.clean, signals, filter='not_implemented')
    pytest.raises(ValueError, nisignal.clean, signals, ensure_finite=None)
    # Check warning message when no confound methods were specified,
    # but cutoff frequency provided.
    pytest.warns(UserWarning, nisignal.clean, signals,
                t_r=2.5, filter=False, low_pass=.01, match='not perform filtering')

    # Test without standardizing that constant parts of confounds are
    # accounted for
    # passing standardize_confounds=False, detrend=False should raise warning
    warning_message = r"must perform detrend and/or standardize confounds"
    with pytest.warns(UserWarning, match=warning_message):
        np.testing.assert_almost_equal(
            nisignal.clean(np.ones((20, 2)),
                           standardize=False,
                           confounds=np.ones(20),
                           standardize_confounds=False,
                           detrend=False,
                           ).mean(),
            np.zeros((20, 2)))

    # Test to check that confounders effects are effectively removed from
    # the signals when having a detrending and filtering operation together.
    # This did not happen originally due to a different order in which
    # these operations were being applied to the data and confounders
    # (it thus solves issue # 2730).
    signals_clean = nisignal.clean(signals,
                                   detrend=True,
                                   high_pass=0.01,
                                   standardize_confounds=True,
                                   standardize=True,
                                   confounds=confounds)
    confounds_clean = nisignal.clean(confounds,
                                     detrend=True,
                                     high_pass=0.01,
                                     standardize=True)
    assert abs(np.dot(confounds_clean.T, signals_clean)).max() < 1000. * eps


def test_clean_frequencies_using_power_spectrum_density():

    # Create signal
    sx = np.array([np.sin(np.linspace(0, 100, 100) * 1.5),
                   np.sin(np.linspace(0, 100, 100) * 3.),
                   np.sin(np.linspace(0, 100, 100) / 8.),
                   ]).T

    # Create confound
    _, _, confounds = generate_signals(
        n_features=10, n_confounds=10, length=100)

    # Apply low- and high-pass filter with butterworth (separately)
    t_r = 1.0
    low_pass = 0.1
    high_pass = 0.4
    res_low = clean(sx, detrend=False, standardize=False,
                    filter='butterworth', low_pass=low_pass, high_pass=None,
                    t_r=t_r)
    res_high = clean(sx, detrend=False, standardize=False,
                     filter='butterworth', low_pass=None, high_pass=high_pass,
                     t_r=t_r)

    # cosine high pass filter
    res_cos = clean(sx, detrend=False, standardize=False,
                    filter='cosine', low_pass=None, high_pass=high_pass,
                    t_r=t_r)

    # Compute power spectrum density for both test
    f, Pxx_den_low = scipy.signal.welch(np.mean(res_low.T, axis=0), fs=t_r)
    f, Pxx_den_high = scipy.signal.welch(np.mean(res_high.T, axis=0), fs=t_r)
    f, Pxx_den_cos = scipy.signal.welch(np.mean(res_cos.T, axis=0), fs=t_r)

    # Verify that the filtered frequencies are removed
    assert np.sum(Pxx_den_low[f >= low_pass * 2.]) <= 1e-4
    assert np.sum(Pxx_den_high[f <= high_pass / 2.]) <= 1e-4
    assert np.sum(Pxx_den_cos[f <= high_pass / 2.]) <= 1e-4


def test_clean_finite_no_inplace_mod():
    """
    Test for verifying that the passed in signal array is not modified.
    For PR #2125 . This test is failing on main, passing in this PR.
    """
    n_samples = 2
    # n_features  Must be higher than 500
    n_features = 501
    x_orig, _, _ = generate_signals(n_features=n_features,
                              length=n_samples)
    x_orig_inital_copy = x_orig.copy()

    x_orig_with_nans = x_orig.copy()
    x_orig_with_nans[0, 0] = np.nan
    x_orig_with_nans_initial_copy = x_orig_with_nans.copy()

    cleaned_x_orig = clean(x_orig)
    assert np.array_equal(x_orig, x_orig_inital_copy)

    cleaned_x_orig_with_nans = clean(x_orig_with_nans, ensure_finite=True)
    assert np.isnan(x_orig_with_nans_initial_copy[0, 0])
    assert np.isnan(x_orig_with_nans[0, 0])


def test_high_variance_confounds():

    # C and F order might take different paths in the function. Check that the
    # result is identical.
    n_features = 1001
    length = 20
    n_confounds = 5
    seriesC, _, _ = generate_signals(n_features=n_features,
                                     length=length, order="C")
    seriesF, _, _ = generate_signals(n_features=n_features,
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
    outG = nisignal.high_variance_confounds(seriesG, percentile=1.,
                                            n_confounds=n_confounds,
                                            detrend=False)
    pytest.raises(AssertionError, np.testing.assert_almost_equal,
                  outC, outG, decimal=13)
    assert(outG.shape == (length, n_confounds))

    # Check shape of output
    out = nisignal.high_variance_confounds(seriesG, n_confounds=7,
                                           detrend=False)
    assert(out.shape == (length, 7))

    # Adding a trend and detrending should give same results as with no trend.
    seriesG = seriesC
    trends = generate_trends(n_features=n_features, length=length)
    seriesGt = seriesG + trends

    outG = nisignal.high_variance_confounds(seriesG, detrend=False,
                                            n_confounds=n_confounds)
    outGt = nisignal.high_variance_confounds(seriesGt, detrend=True,
                                             n_confounds=n_confounds)
    # Since sign flips could occur, we look at the absolute values of the
    # covariance, rather than the absolute difference, and compare this to
    # the identity matrix
    np.testing.assert_almost_equal(np.abs(outG.T.dot(outG)),
                                   np.identity(outG.shape[1]),
                                   decimal=13)
    # Control for sign flips by taking the min of both possibilities
    np.testing.assert_almost_equal(
        np.min(np.abs(np.dstack([outG - outGt, outG + outGt])), axis=2),
        np.zeros(outG.shape))

    # Control robustness to NaNs
    seriesG[:, 0] = 0
    out1 = nisignal.high_variance_confounds(seriesG, n_confounds=n_confounds)
    seriesG[:, 0] = np.nan
    out2 = nisignal.high_variance_confounds(seriesG, n_confounds=n_confounds)
    np.testing.assert_almost_equal(out1, out2, decimal=13)


def test_clean_psc():
    rng = np.random.RandomState(0)
    n_samples = 500
    n_features = 5

    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples)

    # positive mean signal
    means = rng.randn(1, n_features)
    signals_pos_mean = signals + means

    # a mix of pos and neg mean signal
    signals_mixed_mean = signals + np.append(means[:, :-3], -1 * means[:, -3:])

    # both types should pass
    for s in [signals_pos_mean, signals_mixed_mean]:
        cleaned_signals = clean(s, standardize='psc')
        np.testing.assert_almost_equal(cleaned_signals.mean(0), 0)

        cleaned_signals.std(axis=0)
        np.testing.assert_almost_equal(cleaned_signals.mean(0), 0)

        tmp = (s - s.mean(0)) / np.abs(s.mean(0))
        tmp *= 100
        np.testing.assert_almost_equal(cleaned_signals, tmp)

    # leave out the last 3 columns with a mean of zero to test user warning
    signals_w_zero = signals + np.append(means[:, :-3], np.zeros((1, 3)))
    cleaned_w_zero = clean(signals_w_zero, standardize='psc')
    with pytest.warns(UserWarning) as records:
        cleaned_w_zero = clean(signals_w_zero, standardize='psc')
    psc_warning = sum('psc standardization strategy' in str(r.message)
                         for r in records)
    assert psc_warning == 1
    np.testing.assert_equal(cleaned_w_zero[:, -3:].mean(0), 0)


def test_clean_zscore():
    rng = np.random.RandomState(42)
    n_samples = 500
    n_features = 5

    signals, _, _ = generate_signals(n_features=n_features,
                                     length=n_samples)

    signals += rng.standard_normal(size=(1, n_features))
    cleaned_signals_ = clean(signals, standardize='zscore')
    np.testing.assert_almost_equal(cleaned_signals_.mean(0), 0)
    np.testing.assert_almost_equal(cleaned_signals_.std(0), 1)

    # Repeating test above but for new correct strategy
    cleaned_signals = clean(signals, standardize='zscore_sample')
    np.testing.assert_almost_equal(cleaned_signals.mean(0), 0)
    np.testing.assert_almost_equal(cleaned_signals.std(0), 1, decimal=3)

    # Show outcome from two zscore strategies is not equal
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(cleaned_signals_, cleaned_signals)


def test_create_cosine_drift_terms():
    '''Testing cosine filter interface and output.'''
    from nilearn.glm.first_level.design_matrix import _cosine_drift
    # fmriprep high pass cutoff is 128s, it's around 0.008 hz
    t_r, high_pass = 2.5, 0.008
    signals, _, confounds = generate_signals(n_features=41, n_confounds=5,
                                             length=45)

    # Not passing confounds it will return drift terms only
    frame_times = np.arange(signals.shape[0]) * t_r
    cosine_drift = _cosine_drift(high_pass, frame_times)[:, :-1]
    confounds_with_drift = np.hstack((confounds, cosine_drift))

    cosine_confounds = nisignal._create_cosine_drift_terms(
        signals, confounds, high_pass, t_r)
    np.testing.assert_almost_equal(cosine_confounds,
                                   np.hstack((confounds, cosine_drift)))

    # Not passing confounds it will return drift terms only
    drift_terms_only = nisignal._create_cosine_drift_terms(
        signals, None, high_pass, t_r)
    np.testing.assert_almost_equal(drift_terms_only, cosine_drift)

    # drift terms in confounds will create warning and no change to confounds
    with pytest.warns(UserWarning, match='user supplied confounds'):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds_with_drift, high_pass, t_r)
    np.testing.assert_array_equal(cosine_confounds, confounds_with_drift)

    # raise warning if cosine drift term is not created
    high_pass_fail = 0.002
    with pytest.warns(UserWarning, match='Cosine filter was not create'):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds, high_pass_fail, t_r)
    np.testing.assert_array_equal(cosine_confounds, confounds)


def test_sample_mask():
    """Test sample_mask related feature."""
    signals, _, confounds = generate_signals(n_features=11,
                                             n_confounds=5, length=40)

    sample_mask = np.arange(signals.shape[0])
    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]
    sample_mask = np.delete(sample_mask, scrub_index)
    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False

    scrub_clean = clean(signals, confounds=confounds, sample_mask=sample_mask)
    assert scrub_clean.shape[0] == sample_mask.shape[0]

    # test the binary mask
    scrub_clean_bin = clean(signals, confounds=confounds,
                            sample_mask=sample_mask_binary)
    np.testing.assert_equal(scrub_clean_bin, scrub_clean)

    # list of sample_mask for each run
    runs = np.ones(signals.shape[0])
    runs[:signals.shape[0] // 2] = 0
    sample_mask_sep = [np.arange(20), np.arange(20)]
    scrub_index = [[6, 7, 8], [10, 11, 12]]
    sample_mask_sep = [np.delete(sm, si)
                       for sm, si in zip(sample_mask_sep, scrub_index)]
    scrub_sep_mask = clean(signals, confounds=confounds,
                           sample_mask=sample_mask_sep, runs=runs)
    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6

    # test for binary mask per run
    sample_mask_sep_binary = [np.full(signals.shape[0] // 2, True),
                              np.full(signals.shape[0] // 2, True)]
    sample_mask_sep_binary[0][scrub_index[0]] = False
    sample_mask_sep_binary[1][scrub_index[1]] = False
    scrub_sep_mask = clean(signals, confounds=confounds,
                           sample_mask=sample_mask_sep_binary, runs=runs)
    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6

    # 1D sample mask with runs labels
    with pytest.raises(ValueError,
                       match=r'Number of sample_mask \(\d\) not matching'):
        clean(signals, sample_mask=sample_mask, runs=runs)

    # invalid input for sample_mask
    with pytest.raises(TypeError, match='unhandled type'):
        clean(signals, sample_mask='not_supported')

    # sample_mask too long
    with pytest.raises(IndexError,
                       match='more timepoints than the current run'):
        clean(signals, sample_mask=np.hstack((sample_mask, sample_mask)))

    # list of sample_mask with one that's too long
    invalid_sample_mask_sep = [np.arange(10), np.arange(30)]
    with pytest.raises(IndexError,
                       match='more timepoints than the current run'):
        clean(signals, sample_mask=invalid_sample_mask_sep, runs=runs)

    # list of sample_mask  with invalid indexing in one
    sample_mask_sep[-1][-1] = 100
    with pytest.raises(IndexError, match='invalid index'):
        clean(signals, sample_mask=sample_mask_sep, runs=runs)

    # invalid index in 1D sample_mask
    sample_mask[-1] = 999
    with pytest.raises(IndexError, match=r'invalid index \[\d*\]'):
        clean(signals, sample_mask=sample_mask)


def test_handle_scrubbed_volumes():
    """Check interpolation/censoring of signals based on filter type. """
    signals, _, confounds = generate_signals(n_features=11,
                                             n_confounds=5, length=40)

    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.array([2, 3, 6, 7, 8, 30, 31, 32])
    sample_mask = np.delete(sample_mask, scrub_index)

    interpolated_signals, interpolated_confounds = \
        nisignal._handle_scrubbed_volumes(signals, confounds, sample_mask,
                                          'butterworth', 2.5)
    np.testing.assert_equal(interpolated_signals[sample_mask, :],
                            signals[sample_mask, :])
    np.testing.assert_equal(interpolated_confounds[sample_mask, :],
                            confounds[sample_mask, :])

    scrubbed_signals, scrubbed_confounds = \
        nisignal._handle_scrubbed_volumes(signals, confounds, sample_mask,
                                          'cosine', 2.5)
    np.testing.assert_equal(scrubbed_signals,
                            signals[sample_mask, :])
    np.testing.assert_equal(scrubbed_confounds,
                            confounds[sample_mask, :])
