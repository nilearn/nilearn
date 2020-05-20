"""
Preprocessing functions for time series.

All functions in this module should take X matrices with samples x
features
"""
# Authors: Alexandre Abraham, Gael Varoquaux, Philippe Gervais
# License: simplified BSD

import distutils.version
import warnings

import numpy as np
from scipy import linalg, signal as sp_signal
from sklearn.utils import gen_even_slices, as_float_array

from ._utils.numpy_conversions import csv_to_array, as_ndarray

NP_VERSION = distutils.version.LooseVersion(np.version.short_version).version


def _standardize(signals, detrend=False, standardize='zscore'):
    """ Center and standardize a given signal (time is along first axis)

    Parameters
    ----------
    signals: numpy.ndarray
        Timeseries to standardize

    detrend: bool
        if detrending of timeseries is requested

    standardize: {'zscore', 'psc', True, False}, default is 'zscore'
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.

    Returns
    -------
    std_signals: numpy.ndarray
        copy of signals, standardized.
    """

    if standardize not in [True, False, 'psc', 'zscore']:
        raise ValueError('{} is no valid standardize strategy.'
                         .format(standardize))

    if detrend:
        signals = _detrend(signals, inplace=False)
    else:
        signals = signals.copy()

    if standardize:
        if signals.shape[0] == 1:
            warnings.warn('Standardization of 3D signal has been requested but '
                          'would lead to zero values. Skipping.')
            return signals

        elif (standardize == 'zscore') or (standardize is True):
            if not detrend:
                # remove mean if not already detrended
                signals = signals - signals.mean(axis=0)

            std = signals.std(axis=0)
            std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
            signals /= std

        elif standardize == 'psc':
            mean_signal = signals.mean(axis=0)
            invalid_ix = mean_signal < np.finfo(np.float).eps
            signals = (signals / mean_signal) * 100
            signals -= 100

            if np.any(invalid_ix):
                warnings.warn('psc standardization strategy is meaningless '
                              'for features that have a mean of 0 or '
                              'less. These time series are set to 0.')
                signals[:, invalid_ix] = 0

    return signals


def _mean_of_squares(signals, n_batches=20):
    """Compute mean of squares for each signal.
    This function is equivalent to

        var = np.copy(signals)
        var **= 2
        var = var.mean(axis=0)

    but uses a lot less memory.

    Parameters
    ----------
    signals : numpy.ndarray, shape (n_samples, n_features)
        signal whose mean of squares must be computed.

    n_batches : int, optional
        number of batches to use in the computation. Tweaking this value
        can lead to variation of memory usage and computation time. The higher
        the value, the lower the memory consumption.

    """
    # No batching for small arrays
    if signals.shape[1] < 500:
        n_batches = 1

    # Fastest for C order
    var = np.empty(signals.shape[1])
    for batch in gen_even_slices(signals.shape[1], n_batches):
        tvar = np.copy(signals[:, batch])
        tvar **= 2
        var[batch] = tvar.mean(axis=0)

    return var


def _detrend(signals, inplace=False, type="linear", n_batches=10):
    """Detrend columns of input array.

    Signals are supposed to be columns of `signals`.
    This function is significantly faster than scipy.signal.detrend on this
    case and uses a lot less memory.

    Parameters
    ----------
    signals : numpy.ndarray
        This parameter must be two-dimensional.
        Signals to detrend. A signal is a column.

    inplace : bool, optional
        Tells if the computation must be made inplace or not (default
        False).

    type : str, optional
        Detrending type ("linear" or "constant").
        See also scipy.signal.detrend.

    n_batches : int, optional
        number of batches to use in the computation. Tweaking this value
        can lead to variation of memory usage and computation time. The higher
        the value, the lower the memory consumption.

    Returns
    -------
    detrended_signals: numpy.ndarray
        Detrended signals. The shape is that of 'signals'.

    Notes
    -----

    If a signal of lenght 1 is given, it is returned unchanged.

    """
    signals = as_float_array(signals, copy=not inplace)
    if signals.shape[0] == 1:
        warnings.warn('Detrending of 3D signal has been requested but '
                      'would lead to zero values. Skipping.')
        return signals

    signals -= np.mean(signals, axis=0)
    if type == "linear":
        # Keeping "signals" dtype avoids some type conversion further down,
        # and can save a lot of memory if dtype is single-precision.
        regressor = np.arange(signals.shape[0], dtype=signals.dtype)
        regressor -= regressor.mean()
        std = np.sqrt((regressor ** 2).sum())
        # avoid numerical problems
        if not std < np.finfo(np.float).eps:
            regressor /= std
        regressor = regressor[:, np.newaxis]

        # No batching for small arrays
        if signals.shape[1] < 500:
            n_batches = 1

        # This is fastest for C order.
        for batch in gen_even_slices(signals.shape[1], n_batches):
            signals[:, batch] -= np.dot(regressor[:, 0], signals[:, batch]
                                        ) * regressor
    return signals


def _check_wn(btype, freq, nyq):
    wn = freq / float(nyq)
    if wn >= 1.:
        # results looked unstable when the critical frequencies are
        # exactly at the Nyquist frequency. See issue at SciPy
        # https://github.com/scipy/scipy/issues/6265. Before, SciPy 1.0.0 ("wn
        # should be btw 0 and 1"). But, after ("0 < wn < 1"). Due to unstable
        # results as pointed in the issue above. Hence, we forced the
        # critical frequencies to be slightly less than 1. but not 1.
        wn = 1 - 10 * np.finfo(1.).eps
        warnings.warn(
            'The frequency specified for the %s pass filter is '
            'too high to be handled by a digital filter (superior to '
            'nyquist frequency). It has been lowered to %.2f (nyquist '
            'frequency).' % (btype, wn))

    if wn < 0.0: # equal to 0.0 is okay
        wn = np.finfo(1.).eps
        warnings.warn(
            'The frequency specified for the %s pass filter is '
            'too low to be handled by a digital filter (must be non-negative).'
            ' It has been set to eps: %.5e' % (btype, wn))

    return wn


def butterworth(signals, sampling_rate, low_pass=None, high_pass=None,
                order=5, copy=False):
    """ Apply a low-pass, high-pass or band-pass Butterworth filter

    Apply a filter to remove signal below the `low` frequency and above the
    `high` frequency.

    Parameters
    ----------
    signals: numpy.ndarray (1D sequence or n_samples x n_sources)
        Signals to be filtered. A signal is assumed to be a column
        of `signals`.

    sampling_rate: float
        Number of samples per time unit (sample frequency)

    low_pass: float, optional
        If specified, signals above this frequency will be filtered out
        (low pass). This is -3dB cutoff frequency.

    high_pass: float, optional
        If specified, signals below this frequency will be filtered out
        (high pass). This is -3dB cutoff frequency.

    order: integer, optional
        Order of the Butterworth filter. When filtering signals, the
        filter has a decay to avoid ringing. Increasing the order
        sharpens this decay. Be aware that very high orders could lead
        to numerical instability.

    copy: bool, optional
        If False, `signals` is modified inplace, and memory consumption is
        lower than for copy=True, though computation time is higher.

    Returns
    -------
    filtered_signals: numpy.ndarray
        Signals filtered according to the parameters
    """
    if low_pass is None and high_pass is None:
        if copy:
            return signals.copy()
        else:
            return signals

    if low_pass is not None and high_pass is not None \
            and high_pass >= low_pass:
        raise ValueError(
            "High pass cutoff frequency (%f) is greater or equal"
            "to low pass filter frequency (%f). This case is not handled "
            "by this function."
            % (high_pass, low_pass))

    nyq = sampling_rate * 0.5

    critical_freq = []
    if high_pass is not None:
        btype = 'high'
        critical_freq.append(_check_wn(btype, high_pass, nyq))

    if low_pass is not None:
        btype = 'low'
        critical_freq.append(_check_wn(btype, low_pass, nyq))

    if len(critical_freq) == 2:
        btype = 'band'
    else:
        critical_freq = critical_freq[0]

    b, a = sp_signal.butter(order, critical_freq, btype=btype, output='ba')
    if signals.ndim == 1:
        # 1D case
        output = sp_signal.filtfilt(b, a, signals)
        if copy:  # filtfilt does a copy in all cases.
            signals = output
        else:
            signals[...] = output
    else:
        if copy:
            # No way to save memory when a copy has been requested,
            # because filtfilt does out-of-place processing
            signals = sp_signal.filtfilt(b, a, signals, axis=0)
        else:
            # Lesser memory consumption, slower.
            for timeseries in signals.T:
                timeseries[:] = sp_signal.filtfilt(b, a, timeseries)

            # results returned in-place

    return signals


def high_variance_confounds(series, n_confounds=5, percentile=2.,
                            detrend=True):
    """ Return confounds time series extracted from series with highest
    variance.

    Parameters
    ----------
    series: numpy.ndarray
        Timeseries. A timeseries is a column in the "series" array.
        shape (sample number, feature number)

    n_confounds: int, optional
        Number of confounds to return

    percentile: float, optional
        Highest-variance series percentile to keep before computing the
        singular value decomposition, 0. <= `percentile` <= 100.
        series.shape[0] * percentile / 100 must be greater than n_confounds

    detrend: bool, optional
        If True, detrend timeseries before processing.

    Returns
    -------
    v: numpy.ndarray
        highest variance confounds. Shape: (samples, n_confounds)

    Notes
    -----
    This method is related to what has been published in the literature
    as 'CompCor' (Behzadi NeuroImage 2007).

    The implemented algorithm does the following:

    - compute sum of squares for each time series (no mean removal)
    - keep a given percentile of series with highest variances (percentile)
    - compute an svd of the extracted series
    - return a given number (n_confounds) of series from the svd with
      highest singular values.

    See also
    --------
    nilearn.image.high_variance_confounds
    """

    if detrend:
        series = _detrend(series)  # copy

    # Retrieve the voxels|features with highest variance

    # Compute variance without mean removal.
    var = _mean_of_squares(series)
    var_thr = np.nanpercentile(var, 100. - percentile)
    series = series[:, var > var_thr]  # extract columns (i.e. features)
    # Return the singular vectors with largest singular values
    # We solve the symmetric eigenvalue problem here, increasing stability
    s, u = linalg.eigh(series.dot(series.T) / series.shape[0])
    ix_ = np.argsort(s)[::-1]
    u = u[:, ix_[:n_confounds]].copy()
    return u


def _ensure_float(data):
    "Make sure that data is a float type"
    if not data.dtype.kind == 'f':
        if data.dtype.itemsize == '8':
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float32)
    return data


def clean(signals, sessions=None, detrend=True, standardize='zscore',
          confounds=None, low_pass=None,
          high_pass=None, t_r=2.5, ensure_finite=False):
    """Improve SNR on masked fMRI signals.

    This function can do several things on the input signals, in
    the following order:

    - detrend
    - low- and high-pass filter
    - remove confounds
    - standardize

    Low-pass filtering improves specificity.

    High-pass filtering should be kept small, to keep some
    sensitivity.

    Filtering is only meaningful on evenly-sampled signals.

    According to Lindquist et al. (2018), removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified.

    Parameters
    ----------
    signals: numpy.ndarray
        Timeseries. Must have shape (instant number, features number).
        This array is not modified.

    sessions : numpy array, optional
        Add a session level to the cleaning process. Each session will be
        cleaned independently. Must be a 1D array of n_samples elements.

    confounds: numpy.ndarray, str or list of
        Confounds timeseries. Shape must be
        (instant number, confound number), or just (instant number,)
        The number of time instants in signals and confounds must be
        identical (i.e. signals.shape[0] == confounds.shape[0]).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.

    t_r: float
        Repetition time, in second (sampling period). Set to None if not.

    low_pass, high_pass: float
        Respectively high and low cutoff frequencies, in Hertz.

    detrend: bool
        If detrending should be applied on timeseries (before
        confound removal)

    standardize: {'zscore', 'psc', False}, default is 'zscore'
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        False : Do not standardize the data.

    ensure_finite: bool
        If True, the non-finite values (NANs and infs) found in the data
        will be replaced by zeros.

    Returns
    -------
    cleaned_signals: numpy.ndarray
        Input signals, cleaned. Same shape as `signals`.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space. See `Friston, K. J., A. P. Holmes,
    K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak.
    "Statistical Parametric Maps in Functional Imaging: A General
    Linear Approach". Human Brain Mapping 2, no 4 (1994): 189-210.
    <http://dx.doi.org/10.1002/hbm.460020402>`_

    Orthogonalization between temporal filters and confound removal is based on
    suggestions in `Lindquist, M., Geuter, S., Wager, T., & Caffo, B. (2018).
    Modular preprocessing pipelines can reintroduce artifacts into fMRI data.
    bioRxiv, 407676. <http://dx.doi.org/10.1101/407676>`_

    See Also
    --------
        nilearn.image.clean_img
    """
    if isinstance(low_pass, bool):
        raise TypeError("low pass must be float or None but you provided "
                        "low_pass='{0}'".format(low_pass))
    if isinstance(high_pass, bool):
        raise TypeError("high pass must be float or None but you provided "
                        "high_pass='{0}'".format(high_pass))

    if not isinstance(confounds,
                      (list, tuple, str, np.ndarray, type(None))):
        raise TypeError("confounds keyword has an unhandled type: %s"
                        % confounds.__class__)

    if not isinstance(ensure_finite, bool):
        raise ValueError("'ensure_finite' must be boolean type True or False "
                         "but you provided ensure_finite={0}"
                         .format(ensure_finite))

    if not isinstance(signals, np.ndarray):
        signals = as_ndarray(signals)

    if ensure_finite:
        mask = np.logical_not(np.isfinite(signals))
        if mask.any():
            signals = signals.copy()
            signals[mask] = 0

    # Read confounds
    if confounds is not None:
        if not isinstance(confounds, (list, tuple)):
            confounds = (confounds, )

        all_confounds = []
        for confound in confounds:
            if isinstance(confound, str):
                filename = confound
                confound = csv_to_array(filename)
                if np.isnan(confound.flat[0]):
                    # There may be a header
                    if NP_VERSION >= [1, 4, 0]:
                        confound = csv_to_array(filename, skip_header=1)
                    else:
                        confound = csv_to_array(filename, skiprows=1)
                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")

            elif isinstance(confound, np.ndarray):
                if confound.ndim == 1:
                    confound = np.atleast_2d(confound).T
                elif confound.ndim != 2:
                    raise ValueError("confound array has an incorrect number "
                                     "of dimensions: %d" % confound.ndim)

                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")
            else:
                raise TypeError("confound has an unhandled type: %s"
                                % confound.__class__)
            all_confounds.append(confound)

        # Restrict the signal to the orthogonal of the confounds
        confounds = np.hstack(all_confounds)
        del all_confounds

    if sessions is not None:
        if not len(sessions) == len(signals):
            raise ValueError(('The length of the session vector (%i) '
                              'does not match the length of the signals (%i)')
                              % (len(sessions), len(signals)))
        for s in np.unique(sessions):
            session_confounds = None
            if confounds is not None:
                session_confounds = confounds[sessions == s]
            signals[sessions == s, :] = \
                clean(signals[sessions == s],
                      detrend=detrend, standardize=standardize,
                      confounds=session_confounds, low_pass=low_pass,
                      high_pass=high_pass, t_r=t_r)

    signals = _ensure_float(signals)

    # Apply low- and high-pass filters
    if low_pass is not None or high_pass is not None:
        if t_r is None:
            raise ValueError("Repetition time (t_r) must be specified for "
                             "filtering. You specified None.")
    if detrend:
        mean_signals = signals.mean(axis=0)
        signals = _standardize(signals, standardize=False, detrend=detrend)

    if low_pass is not None or high_pass is not None:
        if t_r is None:
            raise ValueError("Repetition time (t_r) must be specified for "
                             "filtering")

        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)
    # Remove confounds
    if confounds is not None:
        confounds = _ensure_float(confounds)
        # Apply low- and high-pass filters to keep filters orthogonal
        # (according to Lindquist et al. (2018))
        if low_pass is not None or high_pass is not None:

            confounds = butterworth(confounds, sampling_rate=1. / t_r,
                                    low_pass=low_pass, high_pass=high_pass)

        confounds = _standardize(confounds, standardize=standardize,
                                 detrend=detrend)

        if not standardize:
            # Improve numerical stability by controlling the range of
            # confounds. We don't rely on _standardize as it removes any
            # constant contribution to confounds.
            confound_max = np.max(np.abs(confounds), axis=0)
            confound_max[confound_max == 0] = 1
            confounds /= confound_max

        # Pivoting in qr decomposition was added in scipy 0.10
        Q, R, _ = linalg.qr(confounds, mode='economic', pivoting=True)
        Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.]
        signals -= Q.dot(Q.T).dot(signals)

    # Standardize
    if detrend and (standardize == 'psc'):
        # If the signal is detrended, we have to know the original mean
        # signal to calculate the psc.
        signals = _standardize(signals + mean_signals, standardize=standardize,
                               detrend=False)
    else:
        signals = _standardize(signals, standardize=standardize,
                               detrend=False)

    return signals
