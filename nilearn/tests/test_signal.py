"""Test the signals module."""

# Author: Gael Varoquaux, Alexandre Abraham

from pathlib import Path

import numpy as np
import pytest
import scipy.signal
from numpy import array_equal
from numpy.testing import assert_almost_equal
from pandas import read_csv

from nilearn._utils.exceptions import AllVolumesRemovedError
from nilearn.conftest import _rng
from nilearn.signal import (
    _censor_signals,
    _create_cosine_drift_terms,
    _detrend,
    _handle_scrubbed_volumes,
    _mean_of_squares,
    butterworth,
    clean,
    high_variance_confounds,
    row_sum_of_squares,
    standardize_signal,
)

EPS = np.finfo(np.float64).eps


def generate_signals(
    n_features=17, n_confounds=5, length=41, same_variance=True, order="C"
):
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
    rng = _rng()

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
    rng = _rng()
    trends = scipy.signal.detrend(np.linspace(0, 1.0, length), type="constant")
    trends = np.repeat(np.atleast_2d(trends).T, n_features, axis=1)
    factors = rng.standard_normal(size=n_features)
    return trends * factors


def generate_signals_plus_trends(n_features=17, n_samples=41):
    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)
    trends = generate_trends(n_features=n_features, length=n_samples)
    return signals + trends


@pytest.fixture
def data_butterworth_single_timeseries(rng):
    n_samples = 100
    return rng.standard_normal(size=n_samples)


@pytest.fixture
def data_butterworth_multiple_timeseries(
    rng, data_butterworth_single_timeseries
):
    n_features = 20000
    n_samples = 100
    data = rng.standard_normal(size=(n_samples, n_features))
    # set first timeseries to previous data
    data[:, 0] = data_butterworth_single_timeseries
    return data


def test_butterworth(data_butterworth_single_timeseries):
    sampling = 100
    low_pass = 30
    high_pass = 10

    # Compare output for different options.
    # single timeseries
    data = data_butterworth_single_timeseries
    data_original = data.copy()

    out_single = butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=True
    )

    assert_almost_equal(data, data_original)

    butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=False
    )

    assert_almost_equal(out_single, data)
    assert id(out_single) != id(data)


def test_butterworth_multiple_timeseries(
    data_butterworth_single_timeseries, data_butterworth_multiple_timeseries
):
    sampling = 100
    low_pass = 30
    high_pass = 10

    data = data_butterworth_multiple_timeseries
    data_original = data.copy()

    out_single = butterworth(
        data_butterworth_single_timeseries,
        sampling,
        low_pass=low_pass,
        high_pass=high_pass,
        copy=True,
    )

    out1 = butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=True
    )
    assert_almost_equal(data, data_original)
    assert id(out1) != id(data_original)

    # check that multiple- and single-timeseries filtering do the same thing.
    assert_almost_equal(out1[:, 0], out_single)
    butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=False
    )
    assert_almost_equal(out1, data)

    # Test nyquist frequency clipping, issue #482
    out1 = butterworth(data, sampling, low_pass=50.0, copy=True)
    out2 = butterworth(
        data,
        sampling,
        low_pass=80.0,
        copy=True,  # Greater than nyq frequency
    )
    assert_almost_equal(out1, out2)
    assert id(out1) != id(out2)


def test_butterworth_warnings_critical_frequencies(
    data_butterworth_single_timeseries,
):
    """Check for equal values in critical frequencies."""
    data = data_butterworth_single_timeseries
    sampling = 1
    low_pass = 2
    high_pass = 1

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
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert (out == data).all()


def test_butterworth_warnings_lpf_too_high(data_butterworth_single_timeseries):
    """Check for frequency higher than allowed (>=Nyquist).

    The frequency should be modified and the filter should be run.
    """
    data = data_butterworth_single_timeseries

    sampling = 1
    high_pass = 0.01
    low_pass = 0.5
    with pytest.warns(
        UserWarning,
        match="The frequency specified for the low pass filter is too high",
    ):
        out = butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert not array_equal(data, out)


def test_butterworth_warnings_hpf_too_low(data_butterworth_single_timeseries):
    """Check for frequency lower than allowed (<0).

    The frequency should be modified and the filter should be run.
    """
    data = data_butterworth_single_timeseries
    sampling = 1
    high_pass = -1
    low_pass = 0.4

    with pytest.warns(
        UserWarning,
        match="The frequency specified for the high pass filter is too low",
    ):
        out = butterworth(
            data,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )
    assert not array_equal(data, out)


def test_butterworth_errors(data_butterworth_single_timeseries):
    """Check for high-pass frequency higher than low-pass frequency."""
    sampling = 1
    high_pass = 0.2
    low_pass = 0.1
    with pytest.raises(
        ValueError,
        match=(
            r"High pass cutoff frequency \([0-9.]+\) is greater than or "
            r"equal to low pass filter frequency \([0-9.]+\)\."
        ),
    ):
        butterworth(
            data_butterworth_single_timeseries,
            sampling,
            low_pass=low_pass,
            high_pass=high_pass,
            copy=True,
        )


def test_standardize(rng):
    n_features = 10
    n_samples = 17

    # Create random signals with offsets and and negative mean
    a = rng.random((n_samples, n_features))
    a += np.linspace(0, 2.0, n_features)

    # Test raise error when strategy is not valid option
    with pytest.raises(ValueError, match="no valid standardize strategy"):
        standardize_signal(a, standardize="foo")

    # test warning for strategy that will be removed
    with pytest.warns(
        DeprecationWarning, match="default strategy for standardize"
    ):
        standardize_signal(a, standardize="zscore")

    # ensure PSC rescaled correctly, correlation should be 1
    z = standardize_signal(a, standardize="zscore_sample")
    psc = standardize_signal(a, standardize="psc")
    corr_coef_feature = np.corrcoef(z[:, 0], psc[:, 0])[0, 1]
    assert corr_coef_feature.mean() == 1

    # transpose array to fit standardize input.
    # Without trend removal
    b = standardize_signal(a, standardize="zscore_sample")
    stds = np.std(b)
    assert_almost_equal(stds, np.ones(n_features), decimal=1)
    assert_almost_equal(b.sum(axis=0), np.zeros(n_features))

    # With trend removal
    a = np.atleast_2d(np.linspace(0, 2.0, n_features)).T
    b = standardize_signal(a, detrend=True, standardize=False)
    assert_almost_equal(b, np.zeros(b.shape))

    b = standardize_signal(a, detrend=True, standardize="zscore_sample")
    assert_almost_equal(b, np.zeros(b.shape))

    length_1_signal = np.atleast_2d(np.linspace(0, 2.0, n_features))
    np.testing.assert_array_equal(
        length_1_signal,
        standardize_signal(length_1_signal, standardize="zscore_sample"),
    )


def test_detrend():
    """Test custom detrend implementation."""
    point_number = 703
    features = 17
    signals, _, _ = generate_signals(
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
    assert_almost_equal(original, x, decimal=14)
    assert abs(detrended.mean(axis=0)).max() < 15.0 * np.finfo(np.float64).eps
    assert_almost_equal(detrended_scipy, detrended, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    _detrend(x, inplace=True)
    assert abs(x.mean(axis=0)).max() < 15.0 * np.finfo(np.float64).eps
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended_scipy, detrended, decimal=14)
    assert_almost_equal(x, signals, decimal=14)

    length_1_signal = x[0]
    length_1_signal = length_1_signal[np.newaxis, :]
    np.testing.assert_array_equal(length_1_signal, _detrend(length_1_signal))

    # Mean removal on integers
    detrended = _detrend(x.astype(np.int64), inplace=True, type="constant")
    assert abs(detrended.mean(axis=0)).max() < 20.0 * np.finfo(np.float64).eps


def test_mean_of_squares():
    """Test _mean_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, _, _ = generate_signals(
        n_features=n_features, length=n_samples, same_variance=True
    )
    # Reference computation
    var1 = np.copy(signals)
    var1 **= 2
    var1 = var1.mean(axis=0)

    var2 = _mean_of_squares(signals)

    assert_almost_equal(var1, var2)


def test_row_sum_of_squares():
    """Test row_sum_of_squares."""
    n_samples = 11
    n_features = 501  # Higher than 500 required
    signals, _, _ = generate_signals(
        n_features=n_features, length=n_samples, same_variance=True
    )
    # Reference computation
    var1 = signals**2
    var1 = var1.sum(axis=0)

    var2 = row_sum_of_squares(signals)

    assert_almost_equal(var1, var2)


# This test is inspired from Scipy docstring of detrend function
def test_clean_detrending():
    n_samples = 21
    n_features = 501  # Must be higher than 500
    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)
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
    assert np.any(np.isfinite(y_clean))
    # clean should not modify inputs
    # using assert_almost_equal instead of array_equal due to NaNs
    assert_almost_equal(y_orig, y, decimal=13)

    # test boolean is not given to signal.clean
    with pytest.raises(TypeError):
        clean(x, low_pass=False)
    with pytest.raises(TypeError):
        clean(x, high_pass=False)

    # This should remove trends
    x_detrended = clean(
        x, standardize=False, detrend=True, low_pass=None, high_pass=None
    )
    assert_almost_equal(x_detrended, signals, decimal=13)
    # clean should not modify inputs
    assert array_equal(x_orig, x)

    # This should do nothing
    x_undetrended = clean(
        x, standardize=False, detrend=False, low_pass=None, high_pass=None
    )
    assert abs(x_undetrended - signals).max() >= 0.06
    # clean should not modify inputs
    assert array_equal(x_orig, x)


def test_clean_t_r(rng):
    """Different TRs produce different results after butterworth filtering."""
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
                    "results do not differ "
                    f"for different TRs: {tr1} and {tr2} "
                    f"at cutoffs: low_pass={low_cutoff}, "
                    f"high_pass={high_cutoff} "
                    f"n_samples={n_samples}, n_features={n_features}"
                )
                assert np.any(np.not_equal(det_one_tr, det_diff_tr)), msg
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
    base_filtered = clean(
        x_orig, t_r=t_r, low_pass=low_pass, high_pass=high_pass
    )
    for kwarg_set in kwargs:
        test_filtered = clean(
            x_orig,
            t_r=t_r,
            low_pass=low_pass,
            high_pass=high_pass,
            **kwarg_set,
        )
        # Check that results are **not** the same.
        assert np.any(np.not_equal(base_filtered, test_filtered))


def test_clean_frequencies():
    """Using butterworth method."""
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T
    sx_orig = sx.copy()
    assert (
        clean(
            sx, standardize=False, high_pass=0.002, low_pass=None, t_r=2.5
        ).max()
        > 0.1
    )
    assert (
        clean(
            sx, standardize=False, high_pass=0.2, low_pass=None, t_r=2.5
        ).max()
        < 0.01
    )
    assert clean(sx, standardize=False, low_pass=0.01, t_r=2.5).max() > 0.9
    with pytest.raises(ValueError):
        clean(sx, low_pass=0.4, high_pass=0.5, t_r=2.5)

    # clean should not modify inputs
    _ = clean(sx, standardize=False, detrend=False, low_pass=0.2, t_r=2.5)
    assert array_equal(sx_orig, sx)


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
    assert array_equal(x_orig, x)

    # check the runs are individually cleaned
    x_run1 = clean(
        x[0 : n_samples // 2, :],
        confounds=confounds[0 : n_samples // 2, :],
        standardize=False,
        detrend=True,
        low_pass=None,
        high_pass=None,
    )
    assert array_equal(x_run1, x_detrended[0 : n_samples // 2, :])


@pytest.fixture
def signals():
    return generate_signals(n_features=41, n_confounds=5, length=45)[0]


@pytest.fixture
def confounds():
    return generate_signals(n_features=41, n_confounds=5, length=45)[2]


def test_clean_confounds_errros(signals):
    """Test error handling."""
    with pytest.raises(
        TypeError, match="confounds keyword has an unhandled type"
    ):
        clean(signals, confounds=1)

    with pytest.raises(TypeError, match="confound has an unhandled type"):
        clean(signals, confounds=[None])

    with pytest.raises(
        ValueError, match="Confound signal has an incorrect length"
    ):
        clean(signals, confounds=np.zeros(2))
    with pytest.raises(
        ValueError, match="Confound signal has an incorrect length"
    ):
        clean(signals, confounds=np.zeros((2, 2)))
    with pytest.raises(
        ValueError, match="Confound signal has an incorrect length."
    ):
        current_dir = Path(__file__).parent
        filename1 = current_dir / "data" / "spm_confounds.txt"
        clean(signals[:-1, :], confounds=filename1)

    with pytest.raises(
        ValueError,
        match="confound array has an incorrect number of dimensions",
    ):
        clean(signals, confounds=np.zeros((2, 3, 4)))

    with pytest.raises(
        ValueError,
        match="Repetition time .* and low cutoff frequency .*",
    ):
        clean(signals, filter="cosine", t_r=None, high_pass=0.008)

    with pytest.raises(
        ValueError,
        match="Repetition time .* must be specified for butterworth.",
    ):
        # using butterworth filter here
        clean(signals, t_r=None, low_pass=0.01)

    with pytest.raises(
        ValueError, match="Filter method not_implemented not implemented."
    ):
        clean(signals, filter="not_implemented")

    with pytest.raises(
        ValueError, match="'ensure_finite' must be boolean type True or False"
    ):
        clean(signals, ensure_finite=None)


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
    assert array_equal(noises, noises1)

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


def test_clean_confounds_inputs():
    """Check several types of supported inputs."""
    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=3, length=20
    )
    # Test with confounds read from a file.
    # Smoke test only (result has no meaning).
    current_dir = Path(__file__).parent
    filename1 = current_dir / "data" / "spm_confounds.txt"
    filename2 = current_dir / "data" / "confounds_with_header.csv"

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


def test_clean_warning(signals):
    """Check warnings are thrown."""
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


def test_clean_confounds_are_removed(signals, confounds):
    """Check that confounders effects are effectively removed.

    Check that confounders effects are effectively removed from
    the signals when having a detrending and filtering operation together.
    This did not happen originally due to a different order in which
    these operations were being applied to the data and confounders.
    see https://github.com/nilearn/nilearn/issues/2730
    """
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
    assert abs(np.dot(confounds_clean.T, signals_clean)).max() < 1000.0 * EPS


def test_clean_frequencies_using_power_spectrum_density():
    # Create signal
    sx = np.array(
        [
            np.sin(np.linspace(0, 100, 100) * 1.5),
            np.sin(np.linspace(0, 100, 100) * 3.0),
            np.sin(np.linspace(0, 100, 100) / 8.0),
        ]
    ).T

    # Create confound
    _, _, confounds = generate_signals(
        n_features=10, n_confounds=10, length=100
    )

    # Apply low- and high-pass filter with butterworth (separately)
    t_r = 1.0
    low_pass = 0.1
    high_pass = 0.4
    res_low = clean(
        sx,
        detrend=False,
        standardize=False,
        filter="butterworth",
        low_pass=low_pass,
        high_pass=None,
        t_r=t_r,
    )
    res_high = clean(
        sx,
        detrend=False,
        standardize=False,
        filter="butterworth",
        low_pass=None,
        high_pass=high_pass,
        t_r=t_r,
    )

    # cosine high pass filter
    res_cos = clean(
        sx,
        detrend=False,
        standardize=False,
        filter="cosine",
        low_pass=None,
        high_pass=high_pass,
        t_r=t_r,
    )

    # Compute power spectrum density for both test
    f, Pxx_den_low = scipy.signal.welch(np.mean(res_low.T, axis=0), fs=t_r)
    f, Pxx_den_high = scipy.signal.welch(np.mean(res_high.T, axis=0), fs=t_r)
    f, Pxx_den_cos = scipy.signal.welch(np.mean(res_cos.T, axis=0), fs=t_r)

    # Verify that the filtered frequencies are removed
    assert np.sum(Pxx_den_low[f >= low_pass * 2.0]) <= 1e-4
    assert np.sum(Pxx_den_high[f <= high_pass / 2.0]) <= 1e-4
    assert np.sum(Pxx_den_cos[f <= high_pass / 2.0]) <= 1e-4


@pytest.mark.parametrize("t_r", [1, 1.0])
@pytest.mark.parametrize("high_pass", [1, 1.0])
def test_clean_t_r_highpass_float_int(t_r, high_pass):
    """Make sure t_r and high_pass can be int.

    Regression test for: https://github.com/nilearn/nilearn/issues/4803
    """
    # Create signal
    sx = np.array(
        [
            np.sin(np.linspace(0, 100, 100) * 1.5),
            np.sin(np.linspace(0, 100, 100) * 3.0),
            np.sin(np.linspace(0, 100, 100) / 8.0),
        ]
    ).T

    # Create confound
    _, _, confounds = generate_signals(
        n_features=10, n_confounds=10, length=100
    )
    clean(
        sx,
        detrend=False,
        standardize=False,
        filter="cosine",
        low_pass=None,
        high_pass=high_pass,
        t_r=t_r,
    )


def test_clean_finite_no_inplace_mod():
    """Test for verifying that the passed in signal array is not modified.

    For PR #2125 . This test is failing on main, passing in this PR.
    """
    n_samples = 2
    # n_features  Must be higher than 500
    n_features = 501
    x_orig, _, _ = generate_signals(n_features=n_features, length=n_samples)
    x_orig_inital_copy = x_orig.copy()

    x_orig_with_nans = x_orig.copy()
    x_orig_with_nans[0, 0] = np.nan
    x_orig_with_nans_initial_copy = x_orig_with_nans.copy()

    _ = clean(x_orig)
    assert array_equal(x_orig, x_orig_inital_copy)

    _ = clean(x_orig_with_nans, ensure_finite=True)
    assert np.isnan(x_orig_with_nans_initial_copy[0, 0])
    assert np.isnan(x_orig_with_nans[0, 0])


def test_high_variance_confounds():
    # C and F order might take different paths in the function. Check that the
    # result is identical.
    n_features = 1001
    length = 20
    n_confounds = 5
    seriesC, _, _ = generate_signals(
        n_features=n_features, length=length, order="C"
    )
    seriesF, _, _ = generate_signals(
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
    with pytest.raises(AssertionError):
        assert_almost_equal(outC, outG, decimal=13)
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


def test_clean_standardize_false():
    n_samples = 500
    n_features = 5
    t_r = 2

    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)
    cleaned_signals = clean(signals, standardize=False, detrend=False)
    assert_almost_equal(cleaned_signals, signals)

    # these show return the same results
    cleaned_butterworth_signals = clean(
        signals,
        detrend=False,
        standardize=False,
        filter="butterworth",
        high_pass=0.01,
        t_r=t_r,
    )
    butterworth_signals = butterworth(
        signals,
        sampling_rate=1 / t_r,
        high_pass=0.01,
    )
    np.testing.assert_equal(cleaned_butterworth_signals, butterworth_signals)


def test_clean_psc(rng):
    n_samples = 500
    n_features = 5

    signals = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )
    # positive mean signal
    means = rng.standard_normal((1, n_features))
    signals_pos_mean = signals + means

    # a mix of pos and neg mean signal
    signals_mixed_mean = signals + np.append(means[:, :-3], -1 * means[:, -3:])

    # both types should pass
    for s in [signals_pos_mean, signals_mixed_mean]:
        # no detrend
        cleaned_signals = clean(s, standardize="psc", detrend=False)
        ss_signals = standardize_signal(s, detrend=False, standardize="psc")
        assert_almost_equal(cleaned_signals.mean(0), 0)
        assert_almost_equal(cleaned_signals, ss_signals)

        # psc signal should correlate with z score, since it's just difference
        # in scaling
        z_signals = clean(s, standardize="zscore_sample", detrend=False)
        assert_almost_equal(
            np.corrcoef(z_signals[:, 0], cleaned_signals[:, 0])[0, 1],
            0.99999,
            decimal=5,
        )

        cleaned_signals = clean(s, standardize="psc", detrend=True)
        z_signals = clean(s, standardize="zscore_sample", detrend=True)
        assert_almost_equal(cleaned_signals.mean(0), 0)
        assert_almost_equal(
            np.corrcoef(z_signals[:, 0], cleaned_signals[:, 0])[0, 1],
            0.99999,
            decimal=5,
        )

        # test with high pass with butterworth
        hp_butterworth_signals = clean(
            s,
            detrend=False,
            filter="butterworth",
            high_pass=0.01,
            t_r=2,
            standardize="psc",
        )
        z_butterworth_signals = clean(
            s,
            detrend=False,
            filter="butterworth",
            high_pass=0.01,
            t_r=2,
            standardize="zscore_sample",
        )
        assert_almost_equal(hp_butterworth_signals.mean(0), 0)
        assert_almost_equal(
            np.corrcoef(
                z_butterworth_signals[:, 0], hp_butterworth_signals[:, 0]
            )[0, 1],
            0.99999,
            decimal=5,
        )

    # leave out the last 3 columns with a mean of zero to test user warning
    signals_w_zero = signals + np.append(means[:, :-3], np.zeros((1, 3)))
    with pytest.warns(UserWarning) as records:
        cleaned_w_zero = clean(signals_w_zero, standardize="psc")
    psc_warning = sum(
        "psc standardization strategy" in str(r.message) for r in records
    )
    assert psc_warning == 1
    np.testing.assert_equal(cleaned_w_zero[:, -3:].mean(0), 0)


def test_clean_zscore(rng):
    n_samples = 500
    n_features = 5

    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)

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
        np.testing.assert_array_equal(cleaned_signals_, cleaned_signals)


def test_create_cosine_drift_terms():
    """Testing cosine filter interface and output."""
    from nilearn.glm.first_level.design_matrix import create_cosine_drift

    # fmriprep high pass cutoff is 128s, it's around 0.008 hz
    t_r, high_pass = 2.5, 0.008
    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )

    # Not passing confounds it will return drift terms only
    frame_times = np.arange(signals.shape[0]) * t_r
    cosine_drift = create_cosine_drift(high_pass, frame_times)[:, :-1]
    confounds_with_drift = np.hstack((confounds, cosine_drift))

    cosine_confounds = _create_cosine_drift_terms(
        signals, confounds, high_pass, t_r
    )
    assert_almost_equal(cosine_confounds, np.hstack((confounds, cosine_drift)))

    # Not passing confounds it will return drift terms only
    drift_terms_only = _create_cosine_drift_terms(
        signals, None, high_pass, t_r
    )
    assert_almost_equal(drift_terms_only, cosine_drift)

    # drift terms in confounds will create warning and no change to confounds
    with pytest.warns(UserWarning, match="user supplied confounds"):
        cosine_confounds = _create_cosine_drift_terms(
            signals, confounds_with_drift, high_pass, t_r
        )
    np.testing.assert_array_equal(cosine_confounds, confounds_with_drift)

    # raise warning if cosine drift term is not created
    high_pass_fail = 0.002
    with pytest.warns(UserWarning, match="Cosine filter was not created"):
        cosine_confounds = _create_cosine_drift_terms(
            signals, confounds, high_pass_fail, t_r
        )
    np.testing.assert_array_equal(cosine_confounds, confounds)


def test_sample_mask():
    """Test sample_mask related feature."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    sample_mask = np.arange(signals.shape[0])
    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]
    sample_mask = np.delete(sample_mask, scrub_index)
    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False

    scrub_clean = clean(signals, confounds=confounds, sample_mask=sample_mask)
    assert scrub_clean.shape[0] == sample_mask.shape[0]

    # test the binary mask
    scrub_clean_bin = clean(
        signals, confounds=confounds, sample_mask=sample_mask_binary
    )
    np.testing.assert_equal(scrub_clean_bin, scrub_clean)

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

    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.array([2, 3, 6, 7, 8, 30, 31, 32])
    sample_mask = np.delete(sample_mask, scrub_index)

    (
        interpolated_signals,
        interpolated_confounds,
        sample_mask,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "butterworth", 2.5, True
    )
    np.testing.assert_equal(
        interpolated_signals[sample_mask, :], signals[sample_mask, :]
    )
    np.testing.assert_equal(
        interpolated_confounds[sample_mask, :], confounds[sample_mask, :]
    )

    (
        scrubbed_signals,
        scrubbed_confounds,
        sample_mask,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "cosine", 2.5, True
    )
    np.testing.assert_equal(scrubbed_signals, signals[sample_mask, :])
    np.testing.assert_equal(scrubbed_confounds, confounds[sample_mask, :])


def test_handle_scrubbed_volumes_with_extrapolation():
    """Check interpolation of signals with extrapolation."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.concatenate((np.arange(5), [10, 20, 30]))
    sample_mask = np.delete(sample_mask, scrub_index)

    # Test cubic spline interpolation (enabled extrapolation) in the
    # very first n=5 samples of generated signal
    extrapolate_warning = (
        "By default the cubic spline interpolator extrapolates "
        "the out-of-bounds censored volumes in the data run. This "
        "can lead to undesired filtered signal results. Starting in "
        "version 0.13, the default strategy will be not to extrapolate "
        "but to discard those volumes at filtering."
    )
    with pytest.warns(FutureWarning, match=extrapolate_warning):
        (
            extrapolated_signals,
            extrapolated_confounds,
            extrapolated_sample_mask,
        ) = _handle_scrubbed_volumes(
            signals, confounds, sample_mask, "butterworth", 2.5, True
        )
    np.testing.assert_equal(signals.shape[0], extrapolated_signals.shape[0])
    np.testing.assert_equal(
        confounds.shape[0], extrapolated_confounds.shape[0]
    )
    np.testing.assert_equal(sample_mask, extrapolated_sample_mask)


def test_handle_scrubbed_volumes_without_extrapolation():
    """Check interpolation of signals disabling extrapolation."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    outer_samples = [0, 1, 2, 3, 4]
    inner_samples = [10, 20, 30]
    total_samples = len(outer_samples) + len(inner_samples)
    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.concatenate((outer_samples, inner_samples))
    sample_mask = np.delete(sample_mask, scrub_index)

    # Test cubic spline interpolation without predicting values outside
    # the range of the signal available (disabled extrapolation), discarding
    # the first n censored samples of generated signal
    (
        interpolated_signals,
        interpolated_confounds,
        interpolated_sample_mask,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "butterworth", 2.5, False
    )
    np.testing.assert_equal(
        signals.shape[0], interpolated_signals.shape[0] + len(outer_samples)
    )
    np.testing.assert_equal(
        confounds.shape[0],
        interpolated_confounds.shape[0] + len(outer_samples),
    )
    np.testing.assert_equal(
        sample_mask - sample_mask[0], interpolated_sample_mask
    )

    # Assert that the modified sample mask (interpolated_sample_mask)
    # can be applied to the interpolated signals and confounds
    (
        censored_signals,
        censored_confounds,
    ) = _censor_signals(
        interpolated_signals, interpolated_confounds, interpolated_sample_mask
    )
    np.testing.assert_equal(
        signals.shape[0], censored_signals.shape[0] + total_samples
    )
    np.testing.assert_equal(
        confounds.shape[0], censored_confounds.shape[0] + total_samples
    )


def test_handle_scrubbed_volumes_exception():
    """Check if an exception is raised when the sample mask is empty."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.arange(signals.shape[0])
    sample_mask = np.delete(sample_mask, scrub_index)

    with pytest.raises(
        AllVolumesRemovedError,
        match="The size of the sample mask is 0. "
        "All volumes were marked as motion outliers "
        "can not proceed. ",
    ):
        _handle_scrubbed_volumes(
            signals, confounds, sample_mask, "butterworth", 2.5, True
        )
