"""Test the signals module"""

# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import os.path

import numpy as np
import pytest
import scipy.signal

# Use nisignal here to avoid name collisions
# (using nilearn.signal is not possible)
from nilearn import signal as nisignal
from nilearn._utils.data_gen import generate_signals
from nilearn.signal import (
    _detrend,
    _handle_scrubbed_volumes,
    _row_sum_of_squares,
    _standardize,
    butterworth,
    clean,
    high_variance_confounds,
)
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
from pandas import read_csv


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
    signals, *_ = generate_signals(n_features=n_features, length=n_samples)
    trends = generate_trends(n_features=n_features, length=n_samples)
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

    out_single = butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=True
    )
    assert_almost_equal(data, data_original)

    butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=False
    )

    assert_almost_equal(out_single, data)
    assert_(id(out_single) != id(data))

    # multiple timeseries
    data = rng.standard_normal(size=(n_samples, n_features))
    data[:, 0] = data_original  # set first timeseries to previous data
    data_original = data.copy()

    out1 = butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=True
    )

    assert_almost_equal(data, data_original)
    assert_(id(out1) != id(data_original))

    # check that multiple- and single-timeseries filtering do the same thing.
    assert_almost_equal(out1[:, 0], out_single)

    butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=False
    )

    assert_almost_equal(out1, data)


def test_butterworth_nyquist_frequency_clipping():
    """Test nyquist frequency clipping.

    issue https://github.com/nilearn/nilearn/issues/482
    """
    rng = np.random.RandomState(42)

    n_samples = 100
    sampling = 100
    data = rng.standard_normal(size=n_samples)

    out1 = butterworth(data, sampling, low_pass=50.0, copy=True)
    # Greater than nyq frequency
    out2 = butterworth(data, sampling, low_pass=80.0, copy=True)

    assert_almost_equal(out1, out2)
    assert_(id(out1) != id(out2))


def test_butterworth_single_errors_warnings():
    rng = np.random.RandomState(42)

    # Compare output for different options.
    # single timeseries
    n_samples = 100
    data = rng.standard_normal(size=n_samples)

    with pytest.warns(
        UserWarning,
        match=(
            "Signals are returned unfiltered because "
            "band-pass critical frequencies are equal. "
            "Please check that inputs for sampling_rate, "
            "low_pass, and high_pass are valid."
        ),
    ):
        out = butterworth(
            data,
            sampling_rate=1,
            low_pass=2,
            high_pass=1,
            copy=True,
        )
    assert (out == data).all()

    # Test check for frequency higher than allowed (>=Nyquist).
    # The frequency should be modified and the filter should be run.
    with pytest.warns(
        UserWarning,
        match="frequency .* low pass filter is too high",
    ):
        out = butterworth(
            data,
            sampling_rate=1,
            low_pass=0.5,
            high_pass=0.01,
            copy=True,
        )
    assert not np.array_equal(data, out)

    # Test check for frequency lower than allowed (<0).
    # The frequency should be modified and the filter should be run.
    with pytest.warns(
        UserWarning,
        match="frequency .* high pass filter is too low",
    ):
        out = butterworth(
            data,
            sampling_rate=1,
            low_pass=0.4,
            high_pass=-1,
            copy=True,
        )
    assert not np.array_equal(data, out)

    # Test check for high-pass frequency higher than low-pass frequency.
    # An error should be raised.
    with pytest.raises(
        ValueError,
        match=(
            r"High pass cutoff frequency \([0-9.]+\) is greater than or "
            r"equal to low pass filter frequency \([0-9.]+\)\."
        ),
    ):
        butterworth(
            data,
            sampling_rate=1,
            low_pass=0.1,
            high_pass=0.2,
            copy=True,
        )


def test_standardize_errors_warnings():
    rng = np.random.RandomState(42)

    # Create random signals with offsets
    n_features = 10
    n_samples = 17
    a = rng.random_sample((n_samples, n_features))
    a += np.linspace(0, 2.0, n_features)

    # Test raise error when strategy is not valid option
    with pytest.raises(ValueError, match="no valid standardize strategy"):
        _standardize(a, standardize="foo")

    # test warning for strategy that will be removed
    with pytest.warns(FutureWarning, match="default strategy for standardize"):
        _standardize(a, standardize="zscore")


def test_standardize():
    rng = np.random.RandomState(42)

    # Create random signals with offsets
    n_features = 10
    n_samples = 17
    a = rng.random_sample((n_samples, n_features))
    a += np.linspace(0, 2.0, n_features)

    # transpose array to fit _standardize input.
    # Without trend removal
    b = _standardize(a, standardize="zscore")
    stds = np.std(b)
    assert_almost_equal(stds, np.ones(n_features))
    assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # Repeating test above but for new correct strategy
    b = _standardize(a, standardize="zscore_sample")
    stds = np.std(b)
    assert_almost_equal(stds, np.ones(n_features), decimal=1)
    assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # With trend removal
    a = np.atleast_2d(np.linspace(0, 2.0, n_features)).T
    b = _standardize(a, detrend=True, standardize=False)
    assert_almost_equal(b, np.zeros(b.shape))

    b = _standardize(a, detrend=True, standardize="zscore_sample")
    assert_almost_equal(b, np.zeros(b.shape))

    length_1_signal = np.atleast_2d(np.linspace(0, 2.0, n_features))
    assert_array_equal(
        _standardize(length_1_signal, standardize="zscore"),
        length_1_signal,
    )

    # Repeating test above but for new correct strategy
    length_1_signal = np.atleast_2d(np.linspace(0, 2.0, n_features))
    assert_array_equal(
        _standardize(length_1_signal, standardize="zscore_sample"),
        length_1_signal,
    )


def test_detrend():
    """Test custom detrend implementation."""
    point_number = 703
    features = 17
    signals, *_ = generate_signals(
        n_features=features, length=point_number, same_variance=True
    )
    trends = generate_trends(n_features=features, length=point_number)
    x = signals + trends + 1
    original = x.copy()

    # Mean removal only (out-of-place)
    detrended = _detrend(x, inplace=False, type="constant")
    assert abs(detrended.mean(axis=0)).max() < 15.0 * np.finfo(np.float64).eps

    # out-of-place detrending. Use scipy as a reference implementation
    detrended = _detrend(x, inplace=False)
    detrended_scipy = scipy.signal.detrend(x, axis=0)

    # "x" must be left untouched
    assert_almost_equal(x, original, decimal=14)
    assert abs(detrended.mean(axis=0)).max() < 15.0 * np.finfo(np.float64).eps
    assert_almost_equal(detrended, detrended_scipy, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    _detrend(x, inplace=True)
    assert abs(x.mean(axis=0)).max() < 15.0 * np.finfo(np.float64).eps
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended, detrended_scipy, decimal=14)
    assert_almost_equal(signals, x, decimal=14)

    length_1_signal = x[0]
    length_1_signal = length_1_signal[np.newaxis, :]
    detrended_scipy = _detrend(length_1_signal)
    assert_array_equal(length_1_signal, detrended_scipy)

    # Mean removal on integers
    detrended = _detrend(x.astype(np.int64), inplace=True, type="constant")
    assert abs(detrended.mean(axis=0)).max() < 20.0 * np.finfo(np.float64).eps


def test_mean_of_squares():
    """Test _mean_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, *_ = generate_signals(
        n_features=n_features, length=n_samples, same_variance=True
    )
    var = nisignal._mean_of_squares(signals)

    # Reference computation
    expected_var = np.copy(signals)
    expected_var **= 2
    expected_var = expected_var.mean(axis=0)
    assert_almost_equal(var, expected_var)


def test_row_sum_of_squares():
    """Test _row_sum_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, *_ = generate_signals(
        n_features=n_features, length=n_samples, same_variance=True
    )
    var = _row_sum_of_squares(signals)

    # Reference computation
    expected_var = signals**2
    expected_var = expected_var.sum(axis=0)
    assert_almost_equal(var, expected_var)


# This test is inspired from Scipy docstring of detrend function
def test_clean_detrending():
    n_samples = 21
    n_features = 501  # Must be higher than 500
    signals, *_ = generate_signals(n_features=n_features, length=n_samples)
    trends = generate_trends(n_features=n_features, length=n_samples)
    x = signals + trends
    x_orig = x.copy()

    # if NANs, data out should be False with ensure_finite=True
    y = signals + trends
    y[20, 150] = np.nan
    y[5, 500] = np.nan
    y[15, 14] = np.inf
    y_orig = y.copy()

    y_clean = clean(y, ensure_finite=True)

    assert np.any(np.isfinite(y_clean)), True
    # clean should not modify inputs
    # using assert_almost_equal instead of array_equal due to NaNs
    assert_almost_equal(y, y_orig, decimal=13)

    # This should remove trends
    x_detrended = clean(
        x, standardize=False, detrend=True, low_pass=None, high_pass=None
    )
    assert_almost_equal(x_detrended, signals, decimal=13)
    # clean should not modify inputs
    assert np.array_equal(x, x_orig)

    # This should do nothing
    x_undetrended = clean(
        x, standardize=False, detrend=False, low_pass=None, high_pass=None
    )
    assert abs(x_undetrended - signals).max() >= 0.06
    # clean should not modify inputs
    assert np.array_equal(x, x_orig)


def test_clean_errors():
    signals, *_ = generate_signals()
    trends = generate_trends()
    x = signals + trends

    # test boolean is not given to signal.clean
    with pytest.raises(TypeError, match="high/low pass must be float or None"):
        clean(x, low_pass=False)
    with pytest.raises(TypeError, match="high/low pass must be float or None"):
        clean(x, high_pass=False)


def test_clean_t_r():
    """Different TRs must produce different results \
    after butterworth filtering."""
    rng = np.random.RandomState(42)
    n_samples = 34
    # n_features  Must be higher than 500
    n_features = 501
    x_orig = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )
    random_tr_list1 = np.round(rng.uniform(size=3) * 10, decimals=2)
    random_tr_list2 = np.round(rng.uniform(size=3) * 10, decimals=2)
    for tr1, tr2 in zip(random_tr_list1, random_tr_list2):
        low_pass_freq_list = tr1 * np.array([1.0 / 100, 1.0 / 110])
        high_pass_freq_list = tr1 * np.array([1.0 / 210, 1.0 / 190])
        for low_cutoff, high_cutoff in zip(
            low_pass_freq_list, high_pass_freq_list
        ):
            det_one_tr = clean(
                x_orig, t_r=tr1, low_pass=low_cutoff, high_pass=high_cutoff
            )
            det_diff_tr = clean(
                x_orig, t_r=tr2, low_pass=low_cutoff, high_pass=high_cutoff
            )

            if not np.isclose(tr1, tr2, atol=0.3):
                msg = (
                    "results do not differ for "
                    f"different TRs: {tr1} and {tr2} "
                    f"at cutoffs: low_pass={low_cutoff}, "
                    f"high_pass={high_cutoff} n_samples={n_samples}, "
                    f"n_features={n_features}"
                )
                assert_(np.any(np.not_equal(det_one_tr, det_diff_tr)), msg)
                del det_one_tr, det_diff_tr


@pytest.mark.parametrize(
    "kwargs",
    [
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
    ],
)
def test_clean_kwargs(kwargs):
    """Providing kwargs to clean should change the filtered results."""
    n_samples = 34
    n_features = 501
    x_orig = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )
    # Base result
    t_r, high_pass, low_pass = 0.8, 0.01, 0.08
    base_filtered = clean(
        x_orig, t_r=t_r, low_pass=low_pass, high_pass=high_pass
    )

    test_filtered = clean(
        x_orig,
        t_r=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        **kwargs,
    )

    # Check that results are **not** the same.
    assert_(np.any(np.not_equal(base_filtered, test_filtered)))


def test_clean_frequencies():
    """Using butterworth method."""
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T
    sx_orig = sx.copy()

    result = clean(
        sx, standardize=False, high_pass=0.002, low_pass=None, t_r=2.5
    )

    assert result.max() > 0.1

    result = clean(
        sx, standardize=False, high_pass=0.2, low_pass=None, t_r=2.5
    )

    assert result.max() < 0.01

    result = clean(sx, standardize=False, low_pass=0.01, t_r=2.5)

    assert result.max() > 0.9

    pytest.raises(ValueError, clean, sx, low_pass=0.4, high_pass=0.5, t_r=2.5)

    # clean should not modify inputs
    clean(sx, standardize=False, detrend=False, low_pass=0.2, t_r=2.5)
    assert np.array_equal(sx_orig, sx)


def test_clean_runs():
    n_samples = 21
    n_features = 501  # Must be higher than 500
    signals, _, confounds = generate_signals(
        n_features=n_features, length=n_samples
    )
    trends = generate_trends(n_features=n_features, length=n_samples)
    x = signals + trends
    x_orig = x.copy()
    # Create run info
    runs = np.ones(n_samples)
    runs[: n_samples // 2] = 0
    x_detrended = clean(
        x,
        confounds=confounds,
        standardize=False,
        detrend=True,
        low_pass=None,
        high_pass=None,
        runs=runs,
    )
    # clean should not modify inputs
    assert np.array_equal(x_orig, x)

    # check the runs are individually cleaned
    x_run1 = clean(
        x[0 : n_samples // 2, :],
        confounds=confounds[0 : n_samples // 2, :],
        standardize=False,
        detrend=True,
        low_pass=None,
        high_pass=None,
    )
    assert np.array_equal(x_run1, x_detrended[0 : n_samples // 2, :])


def test_clean_confounds():
    signals, noises, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )
    # No signal: output must be zero.
    eps = np.finfo(np.float64).eps
    noises1 = noises.copy()

    cleaned_signals = clean(
        noises, confounds=confounds, detrend=True, standardize=False
    )

    assert abs(cleaned_signals).max() < 100.0 * eps
    # clean should not modify inputs
    assert np.array_equal(noises, noises1)

    # With signal: output must be orthogonal to confounds
    cleaned_signals = clean(
        signals + noises, confounds=confounds, detrend=False, standardize=True
    )

    assert abs(np.dot(confounds.T, cleaned_signals)).max() < 1000.0 * eps

    # Same output when a constant confound is added
    confounds1 = np.hstack((np.ones((45, 1)), confounds))

    cleaned_signals1 = clean(
        signals + noises, confounds=confounds1, detrend=False, standardize=True
    )

    assert_almost_equal(cleaned_signals1, cleaned_signals)

    # Test detrending. No trend should exist in the output.
    # Use confounds with a trend.
    temp = confounds.T
    temp += np.arange(confounds.shape[0])

    cleaned_signals = clean(
        signals + noises, confounds=confounds, detrend=False, standardize=False
    )

    coeffs = np.polyfit(
        np.arange(cleaned_signals.shape[0]), cleaned_signals, 1
    )
    assert (abs(coeffs) > 1e-3).any()  # trends remain

    cleaned_signals = clean(
        signals + noises, confounds=confounds, detrend=True, standardize=False
    )

    coeffs = np.polyfit(
        np.arange(cleaned_signals.shape[0]), cleaned_signals, 1
    )
    assert (abs(coeffs) < 1000.0 * eps).all()  # trend removed

    # Test no-op
    input_signals = 10 * signals

    cleaned_signals = clean(input_signals, detrend=False, standardize=False)

    assert_almost_equal(cleaned_signals, input_signals)

    cleaned_signals = clean(input_signals, detrend=False, standardize=True)

    assert_almost_equal(
        cleaned_signals.var(axis=0), np.ones(cleaned_signals.shape[1])
    )


def test_clean_confounds_from_file():
    # Test with confounds read from a file. Smoke test only (result has
    # no meaning).
    current_dir = os.path.split(__file__)[0]

    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=3, length=20
    )
    filename1 = os.path.join(current_dir, "data", "spm_confounds.txt")
    filename2 = os.path.join(current_dir, "data", "confounds_with_header.csv")

    clean(signals, detrend=False, standardize=False, confounds=filename1)
    clean(signals, detrend=False, standardize=False, confounds=filename2)
    clean(signals, detrend=False, standardize=False, confounds=confounds[:, 1])

    # test with confounds as a pandas DataFrame
    confounds_df = read_csv(filename2, sep="\t")
    clean(
        signals,
        detrend=False,
        standardize=False,
        confounds=confounds_df.values,
    )
    clean(signals, detrend=False, standardize=False, confounds=confounds_df)

    # test array-like signals
    list_signal = signals.tolist()
    clean(list_signal)

    # Use a list containing two filenames, a 2D array and a 1D array
    clean(
        signals,
        detrend=False,
        standardize=False,
        confounds=[filename1, confounds[:, 0:2], filename2, confounds[:, 2]],
    )


def test_clean_confounds_check_confounders_are_removed():
    """Check that confounders effects are effectively removed from \
    the signals when having a detrending and filtering operation together.

    This did not happen originally due to a different order in which
    these operations were being applied to the data and confounders

    Thus solves issue # 2730
    """
    eps = np.finfo(np.float64).eps

    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=3, length=20
    )

    signals_clean = clean(
        signals,
        detrend=True,
        high_pass=0.01,
        standardize_confounds=True,
        standardize=True,
        confounds=confounds,
    )
    confounds_clean = clean(
        confounds, detrend=True, high_pass=0.01, standardize=True
    )
    assert abs(np.dot(confounds_clean.T, signals_clean)).max() < 1000.0 * eps


def test_clean_confounds_errors():
    signals, *_ = generate_signals(n_features=41, n_confounds=3, length=20)

    # Test error handling
    pytest.raises(TypeError, clean, signals, confounds=1)
    pytest.raises(ValueError, clean, signals, confounds=np.zeros(2))
    pytest.raises(ValueError, clean, signals, confounds=np.zeros((2, 2)))
    pytest.raises(ValueError, clean, signals, confounds=np.zeros((2, 3, 4)))
    pytest.raises(TypeError, clean, signals, confounds=[None])
    error_msg = pytest.raises(
        ValueError,
        clean,
        signals,
        filter="cosine",
        t_r=None,
        high_pass=0.008,
    )
    assert "t_r='None'" in str(error_msg.value)
    pytest.raises(
        ValueError, clean, signals, t_r=None, low_pass=0.01
    )  # using butterworth filter here
    pytest.raises(ValueError, clean, signals, filter="not_implemented")
    pytest.raises(ValueError, clean, signals, ensure_finite=None)

    current_dir = os.path.split(__file__)[0]
    filename1 = os.path.join(current_dir, "data", "spm_confounds.txt")
    pytest.raises(ValueError, clean, signals[:-1, :], confounds=filename1)


def test_clean_confounds_warnings():
    signals, *_ = generate_signals(n_features=41, n_confounds=3, length=20)

    # Check warning message when no confound methods were specified,
    # but cutoff frequency provided.
    with pytest.warns(UserWarning, match="not perform filtering"):
        clean(signals, t_r=2.5, filter=False, low_pass=0.01)

    # Test without standardizing that constant parts of confounds are
    # accounted for
    # passing standardize_confounds=False, detrend=False should raise warning
    warning_message = r"must perform detrend and/or standardize confounds"
    with pytest.warns(UserWarning, match=warning_message):
        assert_almost_equal(
            clean(
                np.ones((20, 2)),
                standardize=False,
                confounds=np.ones(20),
                standardize_confounds=False,
                detrend=False,
            ).mean(),
            np.zeros((20, 2)),
        )


@pytest.mark.parametrize(
    "low_pass, high_pass, filter",
    [
        (0.1, None, "butterworth"),
        (None, 0.4, "butterworth"),
        (None, 0.4, "cosine"),
    ],
)
def test_clean_frequencies_using_power_spectrum_density(
    low_pass, high_pass, filter
):
    # Create signal
    sx = np.array(
        [
            np.sin(np.linspace(0, 100, 100) * 1.5),
            np.sin(np.linspace(0, 100, 100) * 3.0),
            np.sin(np.linspace(0, 100, 100) / 8.0),
        ]
    ).T

    # Apply low- and high-pass filter with butterworth (separately)
    t_r = 1.0

    results = clean(
        sx,
        detrend=False,
        standardize=False,
        filter=filter,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
    )

    # Compute power spectrum density for both test
    sample_freq, spectral_density = scipy.signal.welch(
        np.mean(results.T, axis=0), fs=t_r
    )

    # Verify that the filtered frequencies are removed
    if low_pass:
        assert np.sum(spectral_density[sample_freq >= low_pass * 2.0]) <= 1e-4
    else:
        assert np.sum(spectral_density[sample_freq <= high_pass / 2.0]) <= 1e-4


def test_clean_finite_no_inplace_mod():
    """Verify that the passed in signal array is not modified.

    See PR https://github.com/nilearn/nilearn/pull/2125
    """
    n_samples = 2
    # n_features  Must be higher than 500
    n_features = 501
    x_orig, *_ = generate_signals(n_features=n_features, length=n_samples)
    x_orig_inital_copy = x_orig.copy()

    x_orig_with_nans = x_orig.copy()
    x_orig_with_nans[0, 0] = np.nan
    x_orig_with_nans_initial_copy = x_orig_with_nans.copy()

    clean(x_orig)
    assert np.array_equal(x_orig, x_orig_inital_copy)

    clean(x_orig_with_nans, ensure_finite=True)
    assert np.isnan(x_orig_with_nans_initial_copy[0, 0])
    assert np.isnan(x_orig_with_nans[0, 0])


def test_high_variance_confounds():
    # C and F order might take different paths in the function. Check that the
    # result is identical.
    n_features = 1001
    length = 20
    n_confounds = 5

    seriesC, *_ = generate_signals(
        n_features=n_features, length=length, order="C"
    )
    seriesF, *_ = generate_signals(
        n_features=n_features, length=length, order="F"
    )

    assert_almost_equal(seriesC, seriesF, decimal=13)

    outC = high_variance_confounds(
        seriesC, n_confounds=n_confounds, detrend=False
    )
    outF = high_variance_confounds(
        seriesF, n_confounds=n_confounds, detrend=False
    )

    assert_almost_equal(outC, outF, decimal=13)

    # Result must not be influenced by global scaling
    seriesG = 2 * seriesC
    outG = high_variance_confounds(
        seriesG, n_confounds=n_confounds, detrend=False
    )

    assert_almost_equal(outC, outG, decimal=13)
    assert outG.shape == (length, n_confounds)

    # Changing percentile changes the result
    seriesG = seriesC
    outG = high_variance_confounds(
        seriesG, percentile=1.0, n_confounds=n_confounds, detrend=False
    )
    pytest.raises(AssertionError, assert_almost_equal, outC, outG, decimal=13)
    assert outG.shape == (length, n_confounds)

    # Check shape of output
    out = high_variance_confounds(seriesG, n_confounds=7, detrend=False)
    assert out.shape == (length, 7)

    # Adding a trend and detrending should give same results as with no trend.
    seriesG = seriesC
    trends = generate_trends(n_features=n_features, length=length)
    seriesGt = seriesG + trends

    outG = high_variance_confounds(
        seriesG, detrend=False, n_confounds=n_confounds
    )
    outGt = high_variance_confounds(
        seriesGt, detrend=True, n_confounds=n_confounds
    )
    # Since sign flips could occur, we look at the absolute values of the
    # covariance, rather than the absolute difference, and compare this to
    # the identity matrix
    assert_almost_equal(
        np.abs(outG.T.dot(outG)), np.identity(outG.shape[1]), decimal=13
    )
    # Control for sign flips by taking the min of both possibilities
    assert_almost_equal(
        np.min(np.abs(np.dstack([outG - outGt, outG + outGt])), axis=2),
        np.zeros(outG.shape),
    )

    # Control robustness to NaNs
    seriesG[:, 0] = 0
    out1 = high_variance_confounds(seriesG, n_confounds=n_confounds)
    seriesG[:, 0] = np.nan
    out2 = high_variance_confounds(seriesG, n_confounds=n_confounds)
    assert_almost_equal(out1, out2, decimal=13)


def test_clean_psc():
    rng = np.random.RandomState(0)
    n_samples = 500
    n_features = 5

    signals, *_ = generate_signals(n_features=n_features, length=n_samples)

    # positive mean signal
    means = rng.randn(1, n_features)
    signals_pos_mean = signals + means

    # a mix of pos and neg mean signal
    signals_mixed_mean = signals + np.append(means[:, :-3], -1 * means[:, -3:])

    # both types should pass
    for s in [signals_pos_mean, signals_mixed_mean]:
        cleaned_signals = clean(s, standardize="psc")
        assert_almost_equal(cleaned_signals.mean(0), 0)

        cleaned_signals.std(axis=0)
        assert_almost_equal(cleaned_signals.mean(0), 0)

        tmp = (s - s.mean(0)) / np.abs(s.mean(0))
        tmp *= 100
        assert_almost_equal(cleaned_signals, tmp)

    # leave out the last 3 columns with a mean of zero to test user warning
    signals_w_zero = signals + np.append(means[:, :-3], np.zeros((1, 3)))
    cleaned_w_zero = clean(signals_w_zero, standardize="psc")
    with pytest.warns(UserWarning) as records:
        cleaned_w_zero = clean(signals_w_zero, standardize="psc")
    psc_warning = sum(
        "psc standardization strategy" in str(r.message) for r in records
    )
    assert psc_warning == 1
    assert_equal(cleaned_w_zero[:, -3:].mean(0), 0)


def test_clean_zscore():
    rng = np.random.RandomState(42)
    n_samples = 500
    n_features = 5

    signals, *_ = generate_signals(n_features=n_features, length=n_samples)

    signals += rng.standard_normal(size=(1, n_features))
    cleaned_signals_ = clean(signals, standardize="zscore")
    assert_almost_equal(cleaned_signals_.mean(0), 0)
    assert_almost_equal(cleaned_signals_.std(0), 1)

    # Repeating test above but for new correct strategy
    cleaned_signals = clean(signals, standardize="zscore_sample")
    assert_almost_equal(cleaned_signals.mean(0), 0)
    assert_almost_equal(cleaned_signals.std(0), 1, decimal=3)

    # Show outcome from two zscore strategies is not equal
    with pytest.raises(AssertionError):
        assert_array_equal(cleaned_signals_, cleaned_signals)


def test_sample_mask():
    """Test sample_mask related feature."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]

    sample_mask = np.arange(signals.shape[0])
    sample_mask = np.delete(sample_mask, scrub_index)

    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False

    scrub_clean = clean(signals, confounds=confounds, sample_mask=sample_mask)
    assert scrub_clean.shape[0] == sample_mask.shape[0]

    # test the binary mask
    scrub_clean_bin = clean(
        signals, confounds=confounds, sample_mask=sample_mask_binary
    )
    assert_equal(scrub_clean_bin, scrub_clean)


def test_sample_mask_per_run():
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]

    sample_mask = np.arange(signals.shape[0])
    sample_mask = np.delete(sample_mask, scrub_index)

    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False

    # list of sample_mask for each run
    runs = np.ones(signals.shape[0])
    runs[: signals.shape[0] // 2] = 0
    sample_mask_sep = [np.arange(20), np.arange(20)]
    scrub_index = [[6, 7, 8], [10, 11, 12]]
    sample_mask_sep = [
        np.delete(sm, si) for sm, si in zip(sample_mask_sep, scrub_index)
    ]
    scrub_sep_mask = clean(
        signals, confounds=confounds, sample_mask=sample_mask_sep, runs=runs
    )
    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6

    # test for binary mask per run
    sample_mask_sep_binary = [
        np.full(signals.shape[0] // 2, True),
        np.full(signals.shape[0] // 2, True),
    ]
    sample_mask_sep_binary[0][scrub_index[0]] = False
    sample_mask_sep_binary[1][scrub_index[1]] = False
    scrub_sep_mask = clean(
        signals,
        confounds=confounds,
        sample_mask=sample_mask_sep_binary,
        runs=runs,
    )
    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6


def test_sample_mask_errors():
    signals, *_ = generate_signals(n_features=11, n_confounds=5, length=40)

    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]

    sample_mask = np.arange(signals.shape[0])
    sample_mask = np.delete(sample_mask, scrub_index)

    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False

    # list of sample_mask for each run
    runs = np.ones(signals.shape[0])
    runs[: signals.shape[0] // 2] = 0

    # 1D sample mask with runs labels
    with pytest.raises(
        ValueError, match=r"Number of sample_mask \(\d\) not matching"
    ):
        clean(signals, sample_mask=sample_mask, runs=runs)

    # invalid input for sample_mask
    with pytest.raises(TypeError, match="unhandled type"):
        clean(signals, sample_mask="not_supported")

    # sample_mask too long
    with pytest.raises(
        IndexError, match="more timepoints than the current run"
    ):
        clean(signals, sample_mask=np.hstack((sample_mask, sample_mask)))

    # list of sample_mask with one that's too long
    invalid_sample_mask_sep = [np.arange(10), np.arange(30)]
    with pytest.raises(
        IndexError, match="more timepoints than the current run"
    ):
        clean(signals, sample_mask=invalid_sample_mask_sep, runs=runs)

    # list of sample_mask  with invalid indexing in one
    scrub_index = [[6, 7, 8], [10, 11, 12]]

    sample_mask_sep = [np.arange(20), np.arange(20)]
    sample_mask_sep = [
        np.delete(sm, si) for sm, si in zip(sample_mask_sep, scrub_index)
    ]
    sample_mask_sep[-1][-1] = 100

    with pytest.raises(IndexError, match="invalid index"):
        clean(signals, sample_mask=sample_mask_sep, runs=runs)

    # invalid index in 1D sample_mask
    sample_mask[-1] = 999
    with pytest.raises(IndexError, match=r"invalid index \[\d*\]"):
        clean(signals, sample_mask=sample_mask)


def test_handle_scrubbed_volumes():
    """Check interpolation/censoring of signals based on filter type."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )
    scrub_index = np.array([2, 3, 6, 7, 8, 30, 31, 32])

    sample_mask = np.arange(signals.shape[0])
    sample_mask = np.delete(sample_mask, scrub_index)

    (
        interpolated_signals,
        interpolated_confounds,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "butterworth", 2.5
    )

    assert_equal(interpolated_signals[sample_mask, :], signals[sample_mask, :])
    assert_equal(
        interpolated_confounds[sample_mask, :], confounds[sample_mask, :]
    )

    scrubbed_signals, scrubbed_confounds = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "cosine", 2.5
    )

    assert_equal(scrubbed_signals, signals[sample_mask, :])
    assert_equal(scrubbed_confounds, confounds[sample_mask, :])
