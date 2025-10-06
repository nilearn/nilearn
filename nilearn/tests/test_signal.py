"""Test the signals module."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
from numpy import array_equal
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from pandas import read_csv

from nilearn.conftest import _rng
from nilearn.exceptions import AllVolumesRemovedError
from nilearn.signal import (
    _censor_signals,
    _create_cosine_drift_terms,
    _detrend,
    _handle_scrubbed_volumes,
    _mean_of_squares,
    butterworth,
    clean,
    create_cosine_drift,
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
    """Generate signal with a trend."""
    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)
    trends = generate_trends(n_features=n_features, length=n_samples)
    return signals + trends


@pytest.fixture
def data_butterworth_single_timeseries(rng):
    """Generate single timeseries for butterworth tests."""
    n_samples = 100
    return rng.standard_normal(size=n_samples)


@pytest.fixture
def data_butterworth_multiple_timeseries(
    rng, data_butterworth_single_timeseries
):
    """Generate mutltiple timeseries for butterworth tests."""
    n_features = 20000
    n_samples = 100
    data = rng.standard_normal(size=(n_samples, n_features))
    # set first timeseries to previous data
    data[:, 0] = data_butterworth_single_timeseries
    return data


def test_butterworth(data_butterworth_single_timeseries):
    """Check butterworth onsingle timeseries."""
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
    """Check butterworth on multiple / single timeseries do the same thing."""
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

    assert_almost_equal(out1[:, 0], out_single)

    butterworth(
        data, sampling, low_pass=low_pass, high_pass=high_pass, copy=False
    )

    assert_almost_equal(out1, data)


def test_butterworth_nyquist_frequency_clipping(
    data_butterworth_multiple_timeseries,
):
    """Test nyquist frequency clipping.

    issue #482
    """
    sampling = 100

    data = data_butterworth_multiple_timeseries

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


@pytest.mark.parametrize("high_pass", [0.1, 0.2])
def test_butterworth_errors(data_butterworth_single_timeseries, high_pass):
    """Check for high-pass frequency higher or equal to low-pass frequency."""
    sampling = 1
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


def test_standardize_error(rng):
    """Test raise error for wrong strategy."""
    n_features = 10
    n_samples = 17

    # Create random signals with offsets and and negative mean
    a = rng.random((n_samples, n_features))
    a += np.linspace(0, 2.0, n_features)

    with pytest.raises(ValueError, match="'standardize' must be one of"):
        standardize_signal(a, standardize="foo")

    # test warning for strategy that will be removed
    with pytest.warns(FutureWarning, match="default strategy for standardize"):
        standardize_signal(a, standardize="zscore")


def test_standardize(rng):
    """Test starndardize_signal with several options."""
    n_features = 10
    n_samples = 17

    # Create random signals with offsets and and negative mean
    a = rng.random((n_samples, n_features))
    a += np.linspace(0, 2.0, n_features)

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
    b = standardize_signal(a, detrend=True, standardize=None)

    assert_almost_equal(b, np.zeros(b.shape))

    b = standardize_signal(a, detrend=True, standardize="zscore_sample")

    assert_almost_equal(b, np.zeros(b.shape))

    length_1_signal = np.atleast_2d(np.linspace(0, 2.0, n_features))

    assert_array_equal(
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

    assert abs(detrended.mean(axis=0)).max() < 15.0 * EPS

    # out-of-place detrending. Use scipy as a reference implementation
    detrended = _detrend(x, inplace=False)

    detrended_scipy = scipy.signal.detrend(x, axis=0)

    # "x" must be left untouched
    assert_almost_equal(original, x, decimal=14)
    assert abs(detrended.mean(axis=0)).max() < 15.0 * EPS
    assert_almost_equal(detrended_scipy, detrended, decimal=14)
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended, signals, decimal=14)

    # inplace detrending
    _detrend(x, inplace=True)

    assert abs(x.mean(axis=0)).max() < 15.0 * EPS
    # for this to work, there must be no trends at all in "signals"
    assert_almost_equal(detrended_scipy, detrended, decimal=14)
    assert_almost_equal(x, signals, decimal=14)

    length_1_signal = x[0]
    length_1_signal = length_1_signal[np.newaxis, :]
    assert_array_equal(length_1_signal, _detrend(length_1_signal))

    # Mean removal on integers
    detrended = _detrend(x.astype(np.int64), inplace=True, type="constant")

    assert abs(detrended.mean(axis=0)).max() < 20.0 * EPS


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


def test_clean_detrending():
    """Check effect of clean with detrending.

    This test is inspired from Scipy docstring of detrend function.

    - clean should not modify inputs
    - check effect when fintie results requested
    """
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

    y_clean = clean(y, ensure_finite=True, standardize="zscore_sample")

    assert np.any(np.isfinite(y_clean))
    # clean should not modify inputs
    # using assert_almost_equal instead of array_equal due to NaNs
    assert_almost_equal(y_orig, y, decimal=13)

    # This should remove trends as detrend is True by default
    match = "boolean values for 'standardize' will be deprecated"
    with pytest.deprecated_call(match=match):
        x_detrended = clean(x, standardize=False)

    assert_almost_equal(x_detrended, signals, decimal=13)
    # clean should not modify inputs
    assert array_equal(x_orig, x)

    # This should do nothing
    x_undetrended = clean(x, standardize=None, detrend=False)

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
    for tr1, tr2 in zip(random_tr_list1, random_tr_list2, strict=False):
        low_pass_freq_list = tr1 * np.array([1.0 / 100, 1.0 / 110])
        high_pass_freq_list = tr1 * np.array([1.0 / 210, 1.0 / 190])
        for low_cutoff, high_cutoff in zip(
            low_pass_freq_list, high_pass_freq_list, strict=False
        ):
            det_one_tr = clean(
                x_orig,
                t_r=tr1,
                low_pass=low_cutoff,
                high_pass=high_cutoff,
                standardize="zscore_sample",
            )
            det_diff_tr = clean(
                x_orig,
                t_r=tr2,
                low_pass=low_cutoff,
                high_pass=high_cutoff,
                standardize="zscore_sample",
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


@pytest.mark.parametrize(
    "kwarg_set",
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
def test_clean_kwargs(kwarg_set):
    """Providing kwargs to clean should change the filtered results."""
    n_samples = 34
    n_features = 501
    x_orig = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )

    # Base result
    t_r, high_pass, low_pass = 0.8, 0.01, 0.08
    base_filtered = clean(
        x_orig,
        t_r=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        standardize="zscore_sample",
    )

    test_filtered = clean(
        x_orig,
        t_r=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        standardize="zscore_sample",
        **kwarg_set,
    )

    # Check that results are **not** the same.
    assert np.any(np.not_equal(base_filtered, test_filtered))


@pytest.mark.parametrize("cast_to", [int, float, np.int32, np.float32])
def test_clean_t_r_type(cast_to):
    """Check that several types are supported for TR.

    Regression test for https://github.com/nilearn/nilearn/issues/5545.
    """
    n_samples = 34
    n_features = 501
    x_orig = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )

    t_r, high_pass, low_pass = cast_to(1.8), 0.01, 0.08
    clean(
        x_orig,
        t_r=t_r,
        low_pass=low_pass,
        high_pass=high_pass,
        standardize="zscore_sample",
    )


def test_clean_frequencies():
    """Check several values for low and high pass."""
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T

    t_r = 2.5
    standardize = None

    cleaned_signal = clean(
        sx, standardize=standardize, high_pass=0.002, low_pass=None, t_r=t_r
    )
    assert cleaned_signal.max() > 0.1

    cleaned_signal = clean(
        sx, standardize=standardize, high_pass=0.2, low_pass=None, t_r=t_r
    )
    assert cleaned_signal.max() < 0.01

    cleaned_signal = clean(sx, standardize=standardize, low_pass=0.01, t_r=t_r)
    assert cleaned_signal.max() > 0.9

    with pytest.raises(
        ValueError, match=r"High pass .* greater than .* low pass"
    ):
        clean(sx, low_pass=0.4, high_pass=0.5, t_r=t_r)


def test_clean_leaves_input_untouched():
    """Clean should not modify inputs."""
    sx1 = np.sin(np.linspace(0, 100, 2000))
    sx2 = np.sin(np.linspace(0, 100, 2000))
    sx = np.vstack((sx1, sx2)).T
    sx_orig = sx.copy()

    t_r = 2.5
    standardize = None

    _ = clean(
        sx, standardize=standardize, detrend=False, low_pass=0.2, t_r=t_r
    )

    assert array_equal(sx_orig, sx)


def test_clean_runs():
    """Check cleaning across runs."""
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

    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
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
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
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
    """Return generic signal."""
    return generate_signals(n_features=41, n_confounds=5, length=45)[0]


@pytest.fixture
def confounds():
    """Return generic condounds."""
    return generate_signals(n_features=41, n_confounds=5, length=45)[2]


def test_clean_confounds_errors(signals):
    """Test error handling."""
    with pytest.raises(TypeError, match="must be of type"):
        clean(signals, confounds=1)

    with pytest.raises(TypeError, match="confound has an unhandled type"):
        clean(signals, confounds=[None])

    msg = "Confound signal has an incorrect length."
    with pytest.raises(ValueError, match=msg):
        clean(signals, confounds=np.zeros(2))
    with pytest.raises(ValueError, match=msg):
        clean(signals, confounds=np.zeros((2, 2)))
    with pytest.raises(ValueError, match=msg):
        current_dir = Path(__file__).parent
        filename1 = current_dir / "data" / "spm_confounds.txt"
        clean(signals[:-1, :], confounds=filename1)


def test_clean_errros(signals):
    """Test error handling."""
    with pytest.raises(
        ValueError,
        match="confound array has an incorrect number of dimensions",
    ):
        clean(signals, confounds=np.zeros((2, 3, 4)))

    with pytest.raises(
        ValueError,
        match=r"Repetition time .* and low cutoff frequency .*",
    ):
        clean(signals, filter="cosine", t_r=None, high_pass=0.008)

    with pytest.raises(
        ValueError,
        match=r"Repetition time .* must be specified for butterworth.",
    ):
        # using butterworth filter here
        clean(signals, t_r=None, low_pass=0.01)

    with pytest.raises(
        ValueError, match=r"Filter method not_implemented not implemented."
    ):
        clean(signals, filter="not_implemented")

    with pytest.raises(ValueError, match="'ensure_finite' must be one of"):
        clean(signals, ensure_finite=None)

    # test boolean is not given to signal.clean
    with pytest.raises(TypeError, match="high/low pass must be float or None"):
        clean(signals, low_pass=False)

    with pytest.raises(TypeError, match="high/low pass must be float or None"):
        clean(signals, high_pass=False)


def test_clean_confounds():
    """Check output of cleaning when counfoun is passed."""
    signals, noises, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )
    # No signal: output must be zero.
    noises1 = noises.copy()
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        cleaned_signals = clean(
            noises, confounds=confounds, detrend=True, standardize=False
        )

    assert abs(cleaned_signals).max() < 100.0 * EPS
    # clean should not modify inputs
    assert array_equal(noises, noises1)

    # With signal: output must be orthogonal to confounds
    # TODO (nilearn >= 0.14) remove catch FutureWarning, DeprecationWarning
    with pytest.warns(FutureWarning), pytest.warns(DeprecationWarning):
        cleaned_signals = clean(
            signals + noises,
            confounds=confounds,
            detrend=False,
            standardize=True,
        )

    assert abs(np.dot(confounds.T, cleaned_signals)).max() < 1000.0 * EPS

    # Same output when a constant confound is added
    confounds1 = np.hstack((np.ones((45, 1)), confounds))
    # TODO (nilearn >= 0.15) remove catch catch_warnings
    with warnings.catch_warnings(record=True) as warning_lists:
        cleaned_signals1 = clean(
            signals + noises,
            confounds=confounds1,
            detrend=False,
            standardize=True,
        )
        assert any(
            issubclass(x.category, FutureWarning)
            and "boolean values for 'standardize' will be deprecated" in str(x)
            for x in warning_lists
        )
        # TODO (nilearn >= 0.14)
        # remove 'the default strategy will be replaced' catch
        assert any(
            issubclass(x.category, FutureWarning)
            and "the default strategy will be replaced by the new strategy"
            in str(x)
            for x in warning_lists
        )

    assert_almost_equal(cleaned_signals1, cleaned_signals)


def test_clean_confounds_detrending():
    """Test detrending.

    No trend should exist in the output.
    """
    signals, noises, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )
    # Use confounds with a trend.
    temp = confounds.T
    temp += np.arange(confounds.shape[0])
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        cleaned_signals = clean(
            signals + noises,
            confounds=confounds,
            detrend=False,
            standardize=False,
        )
    coeffs = np.polyfit(
        np.arange(cleaned_signals.shape[0]), cleaned_signals, 1
    )

    assert (abs(coeffs) > 1e-3).any()  # trends remain
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        cleaned_signals = clean(
            signals + noises,
            confounds=confounds,
            detrend=True,
            standardize=False,
        )
    coeffs = np.polyfit(
        np.arange(cleaned_signals.shape[0]), cleaned_signals, 1
    )

    assert (abs(coeffs) < 1000.0 * EPS).all()  # trend removed


def test_clean_standardize_true_false():
    """Check difference between standardize False and True."""
    signals, _, _ = generate_signals(n_features=41, n_confounds=5, length=45)

    input_signals = 10 * signals
    cleaned_signals = clean(input_signals, detrend=False, standardize=None)

    assert_almost_equal(cleaned_signals, input_signals)

    # TODO (nilearn >= 0.15) remove catch_warnings
    with warnings.catch_warnings(record=True) as warning_lists:
        cleaned_signals = clean(input_signals, detrend=False, standardize=True)
        assert any(
            issubclass(x.category, FutureWarning)
            and "boolean values for 'standardize' will be deprecated" in str(x)
            for x in warning_lists
        )
        # TODO (nilearn >= 0.14)
        # remove 'the default strategy will be replaced' catch
        assert any(
            issubclass(x.category, FutureWarning)
            and "the default strategy will be replaced by the new strategy"
            in str(x)
            for x in warning_lists
        )

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
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        clean(signals, detrend=False, standardize=False, confounds=filename1)
        clean(signals, detrend=False, standardize=False, confounds=filename2)
        clean(
            signals,
            detrend=False,
            standardize=False,
            confounds=confounds[:, 1],
        )

    # test with confounds as a pandas DataFrame
    confounds_df = read_csv(filename2, sep="\t")
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        clean(
            signals,
            detrend=False,
            standardize=False,
            confounds=confounds_df.values,
        )
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        clean(
            signals, detrend=False, standardize=False, confounds=confounds_df
        )

    # test array-like signals
    list_signal = signals.tolist()
    clean(list_signal, standardize=None)

    # Use a list containing two filenames, a 2D array and a 1D array
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        clean(
            signals,
            detrend=False,
            standardize=False,
            confounds=[
                filename1,
                confounds[:, 0:2],
                filename2,
                confounds[:, 2],
            ],
        )


def test_clean_warning(signals):
    """Check warnings are thrown."""
    # Check warning message when no confound methods were specified,
    # but cutoff frequency provided.
    with pytest.warns(UserWarning, match="not perform filtering"):
        clean(
            signals,
            t_r=2.5,
            filter=False,
            low_pass=0.01,
            standardize="zscore_sample",
        )

    # Test without standardizing that constant parts of confounds are
    # accounted for
    # passing standardize_confounds=False, detrend=False should raise warning
    warning_message = r"must perform detrend and/or standardize confounds"
    with pytest.warns(UserWarning, match=warning_message):
        assert_almost_equal(
            clean(
                np.ones((20, 2)),
                standardize=None,
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
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        signals_clean = clean(
            signals,
            detrend=True,
            high_pass=0.01,
            standardize_confounds=True,
            standardize="zscore_sample",
            confounds=confounds,
        )
        confounds_clean = clean(
            confounds,
            detrend=True,
            high_pass=0.01,
            standardize="zscore_sample",
        )
    assert abs(np.dot(confounds_clean.T, signals_clean)).max() < 1000.0 * EPS


def test_clean_frequencies_using_power_spectrum_density():
    """Check on power spectrum that expected frequencies were removed."""
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
    low_pass = 0.1
    high_pass = 0.4
    res_low = clean(
        sx,
        detrend=False,
        standardize=None,
        filter="butterworth",
        low_pass=low_pass,
        high_pass=None,
        t_r=t_r,
    )
    res_high = clean(
        sx,
        detrend=False,
        standardize=None,
        filter="butterworth",
        low_pass=None,
        high_pass=high_pass,
        t_r=t_r,
    )

    # cosine high pass filter
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
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

    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
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

    _ = clean(x_orig, standardize="zscore_sample")
    assert array_equal(x_orig, x_orig_inital_copy)

    _ = clean(
        x_orig_with_nans, ensure_finite=True, standardize="zscore_sample"
    )
    assert np.isnan(x_orig_with_nans_initial_copy[0, 0])
    assert np.isnan(x_orig_with_nans[0, 0])


def test_high_variance_confounds_c_f():
    """Check C and F order give same result.

    They might take different paths in the function.
    """
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


def test_high_variance_confounds_scaling():
    """Check result not be influenced by global scaling."""
    n_features = 1001
    length = 20
    n_confounds = 5

    seriesC, _, _ = generate_signals(
        n_features=n_features, length=length, order="C"
    )

    seriesG = 2 * seriesC
    outG = high_variance_confounds(
        seriesG, n_confounds=n_confounds, detrend=False
    )

    outC = high_variance_confounds(
        seriesC, n_confounds=n_confounds, detrend=False
    )

    assert_almost_equal(outC, outG, decimal=13)
    assert outG.shape == (length, n_confounds)


def test_high_variance_confounds_percentile():
    """Check changing percentile changes the result."""
    n_features = 1001
    length = 20
    n_confounds = 5

    seriesC, _, _ = generate_signals(
        n_features=n_features, length=length, order="C"
    )
    seriesG = seriesC
    outG = high_variance_confounds(
        seriesG, percentile=1.0, n_confounds=n_confounds, detrend=False
    )

    outC = high_variance_confounds(
        seriesC, n_confounds=n_confounds, detrend=False
    )

    with pytest.raises(AssertionError):
        assert_almost_equal(outC, outG, decimal=13)
    assert outG.shape == (length, n_confounds)


def test_high_variance_confounds_detrend():
    """Check adding a trend and detrending give same results as no trend."""
    n_features = 1001
    length = 20
    n_confounds = 5

    seriesC, _, _ = generate_signals(
        n_features=n_features, length=length, order="C"
    )
    seriesG = seriesC

    # detrend is True by default
    out_detrended = high_variance_confounds(seriesG, n_confounds=7)
    out_not_detrended = high_variance_confounds(
        seriesG, n_confounds=7, detrend=False
    )
    with pytest.raises(AssertionError):
        assert_equal(out_detrended, out_not_detrended)

    # Check shape of output
    assert out_not_detrended.shape == (length, 7)

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


def test_high_variance_confounds_nan():
    """Control robustness to NaNs."""
    n_features = 1001
    length = 20
    n_confounds = 5
    seriesC, _, _ = generate_signals(
        n_features=n_features, length=length, order="C"
    )

    seriesC[:, 0] = 0
    out1 = high_variance_confounds(seriesC, n_confounds=n_confounds)

    seriesC[:, 0] = np.nan
    out2 = high_variance_confounds(seriesC, n_confounds=n_confounds)

    assert_almost_equal(out1, out2, decimal=13)


def test_clean_standardize_none():
    """Check output cleaning butterworth filter and no standardization."""
    n_samples = 500
    n_features = 5
    t_r = 2

    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)

    cleaned_signals = clean(signals, standardize=None, detrend=False)

    assert_almost_equal(cleaned_signals, signals)

    # these show return the same results
    cleaned_butterworth_signals = clean(
        signals,
        detrend=False,
        standardize=None,
        filter="butterworth",
        high_pass=0.01,
        t_r=t_r,
    )
    butterworth_signals = butterworth(
        signals,
        sampling_rate=1 / t_r,
        high_pass=0.01,
    )

    assert_equal(cleaned_butterworth_signals, butterworth_signals)


def test_clean_psc(rng):
    """Test clean with percent signal change."""
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

        _assert_correlation_almost_1(z_signals, cleaned_signals)

        cleaned_signals = clean(s, standardize="psc", detrend=True)
        z_signals = clean(s, standardize="zscore_sample", detrend=True)

        assert_almost_equal(cleaned_signals.mean(0), 0)
        _assert_correlation_almost_1(z_signals, cleaned_signals)


def test_clean_psc_butterworth(rng):
    """Test clean with percent signal change and a butterworth filter."""
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
        _assert_correlation_almost_1(
            z_butterworth_signals, hp_butterworth_signals
        )


def _assert_correlation_almost_1(signal_1, signal_2):
    """Check that correlation between 2 signals equal to 1."""
    assert_almost_equal(
        np.corrcoef(signal_1[:, 0], signal_2[:, 0])[0, 1],
        0.99999,
        decimal=5,
    )


def test_clean_psc_warning(rng):
    """Leave out the last 3 columns with a mean of zero \
       to test user warning positive mean signal.
    """
    n_samples = 500
    n_features = 5

    signals = generate_signals_plus_trends(
        n_features=n_features, n_samples=n_samples
    )

    means = rng.standard_normal((1, n_features))

    signals_w_zero = signals + np.append(means[:, :-3], np.zeros((1, 3)))

    with pytest.warns(UserWarning) as records:
        cleaned_w_zero = clean(signals_w_zero, standardize="psc")

    psc_warning = sum(
        "psc standardization strategy" in str(r.message) for r in records
    )
    assert psc_warning == 1
    assert_equal(cleaned_w_zero[:, -3:].mean(0), 0)


def test_clean_zscore(rng):
    """Check that cleaning with Z scoring gives expected results.

    - mean of 0
    - std of 1
    - difference between and sample and population z-scoring.
    """
    n_samples = 500
    n_features = 5

    signals, _, _ = generate_signals(n_features=n_features, length=n_samples)

    signals += rng.standard_normal(size=(1, n_features))

    # TODO (nilearn >= 0.14) remove catch of FutureWarning
    with pytest.warns(FutureWarning):
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


def test_clean_sample_mask():
    """Test sample_mask related feature."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    sample_mask = np.arange(signals.shape[0])
    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]
    sample_mask = np.delete(sample_mask, scrub_index)

    sample_mask_binary = np.full(signals.shape[0], True)
    sample_mask_binary[scrub_index] = False
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        scrub_clean = clean(
            signals,
            confounds=confounds,
            sample_mask=sample_mask,
            standardize="zscore_sample",
        )

    assert scrub_clean.shape[0] == sample_mask.shape[0]

    # test the binary mask
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        scrub_clean_bin = clean(
            signals,
            confounds=confounds,
            sample_mask=sample_mask_binary,
            standardize="zscore_sample",
        )
    assert_equal(scrub_clean_bin, scrub_clean)


def test_sample_mask_across_runs():
    """Test sample_mask related feature but with several runs."""
    # list of sample_mask for each run
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    runs = np.ones(signals.shape[0])
    runs[: signals.shape[0] // 2] = 0

    sample_mask_sep = [np.arange(20), np.arange(20)]
    scrub_index = [[6, 7, 8], [10, 11, 12]]
    sample_mask_sep = list(map(np.delete, sample_mask_sep, scrub_index))
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        scrub_sep_mask = clean(
            signals,
            confounds=confounds,
            sample_mask=sample_mask_sep,
            runs=runs,
            standardize="zscore_sample",
        )

    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6

    # test for binary mask per run
    sample_mask_sep_binary = [
        np.full(signals.shape[0] // 2, True),
        np.full(signals.shape[0] // 2, True),
    ]
    sample_mask_sep_binary[0][scrub_index[0]] = False
    sample_mask_sep_binary[1][scrub_index[1]] = False
    # TODO (nilearn >= 0.14) remove catch DeprecationWarning
    with pytest.warns(DeprecationWarning):
        scrub_sep_mask = clean(
            signals,
            confounds=confounds,
            sample_mask=sample_mask_sep_binary,
            runs=runs,
            standardize="zscore_sample",
        )

    assert scrub_sep_mask.shape[0] == signals.shape[0] - 6


def test_clean_sample_mask_error():
    """Check proper errors are thrown when using clean with sample_mask."""
    signals, _, _ = generate_signals(n_features=11, n_confounds=5, length=40)

    sample_mask = np.arange(signals.shape[0])
    scrub_index = [2, 3, 6, 7, 8, 30, 31, 32]
    sample_mask = np.delete(sample_mask, scrub_index)

    # list of sample_mask for each run
    runs = np.ones(signals.shape[0])
    runs[: signals.shape[0] // 2] = 0

    sample_mask_sep = [np.arange(20), np.arange(20)]
    scrub_index = [[6, 7, 8], [10, 11, 12]]
    sample_mask_sep = list(map(np.delete, sample_mask_sep, scrub_index))

    # 1D sample mask with runs labels
    with pytest.raises(
        ValueError, match=r"Number of sample_mask \(\d\) not matching"
    ):
        clean(signals, sample_mask=sample_mask, runs=runs)

    # invalid input for sample_mask
    with pytest.raises(TypeError, match="must be of type"):
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

    assert_equal(interpolated_signals[sample_mask, :], signals[sample_mask, :])
    assert_equal(
        interpolated_confounds[sample_mask, :], confounds[sample_mask, :]
    )

    (
        scrubbed_signals,
        scrubbed_confounds,
        sample_mask,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "cosine", 2.5, True
    )

    assert_equal(scrubbed_signals, signals[sample_mask, :])
    assert_equal(scrubbed_confounds, confounds[sample_mask, :])


def test_handle_scrubbed_volumes_with_extrapolation():
    """Check interpolation of signals with extrapolation."""
    signals, _, confounds = generate_signals(
        n_features=11, n_confounds=5, length=40
    )

    sample_mask = np.arange(signals.shape[0])
    scrub_index = np.concatenate((np.arange(5), [10, 20, 30]))
    sample_mask = np.delete(sample_mask, scrub_index)

    (
        extrapolated_signals,
        extrapolated_confounds,
        extrapolated_sample_mask,
    ) = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, "butterworth", 2.5, True
    )
    assert_equal(signals.shape[0], extrapolated_signals.shape[0])
    assert_equal(confounds.shape[0], extrapolated_confounds.shape[0])
    assert_equal(sample_mask, extrapolated_sample_mask)


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
    assert_equal(
        signals.shape[0], interpolated_signals.shape[0] + len(outer_samples)
    )
    assert_equal(
        confounds.shape[0],
        interpolated_confounds.shape[0] + len(outer_samples),
    )
    assert_equal(sample_mask - sample_mask[0], interpolated_sample_mask)

    # Assert that the modified sample mask (interpolated_sample_mask)
    # can be applied to the interpolated signals and confounds
    (
        censored_signals,
        censored_confounds,
    ) = _censor_signals(
        interpolated_signals, interpolated_confounds, interpolated_sample_mask
    )
    assert_equal(signals.shape[0], censored_signals.shape[0] + total_samples)
    assert_equal(
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
        AllVolumesRemovedError, match="The size of the sample mask is 0"
    ):
        _handle_scrubbed_volumes(
            signals, confounds, sample_mask, "butterworth", 2.5, True
        )


def test_create_cosine_drift_terms():
    """Testing cosine filter interface and output."""
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
    assert_array_equal(cosine_confounds, confounds_with_drift)

    # raise warning if cosine drift term is not created
    high_pass_fail = 0.002
    with pytest.warns(UserWarning, match="Cosine filter was not created"):
        cosine_confounds = _create_cosine_drift_terms(
            signals, confounds, high_pass_fail, t_r
        )
    assert_array_equal(cosine_confounds, confounds)


# load the spm file to test cosine basis
my_path = Path(__file__).parents[1] / "glm" / "tests"
full_path_design_matrix_file = my_path / "spm_dmtx.npz"
DESIGN_MATRIX = np.load(full_path_design_matrix_file)


def test_cosine_drift():
    """Check cosine drift created buy nilearn."""
    spm_drifts = DESIGN_MATRIX["cosbf_dt_1_nt_20_hcut_0p1"]
    frame_times = np.arange(20)
    high_pass_frequency = 0.1
    nilearn_drifts = create_cosine_drift(high_pass_frequency, frame_times)
    assert_almost_equal(spm_drifts[:, 1:], nilearn_drifts[:, :-2])
    # nilearn_drifts is placing the constant at the end [:, : - 1]
