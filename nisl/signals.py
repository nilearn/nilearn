"""
Preprocessing functions for time series.
"""
# Authors: Alexandre Abraham, Gael Varoquaux
# License: simplified BSD

import numpy as np
from scipy import signal, stats, linalg
from sklearn.utils.fixes import qr_economic


def _standardize(signals, detrend=False, normalize=True):
    """ Center and norm a given signal (time is along first axis)

    Parameters
    ==========
    signals (numpy.ndarray)
        Timeseries to standardize

    detrend (boolean)
        if detrending of timeseries is requested

    normalize (boolean)
        if True, shift timeseries to zero mean value and scale
        to unit energy (sum of squares).

    Returns
    =======
    std_signals: copy of signals, normalized.
    """
    if detrend:
        signals = _detrend(signals, inplace=False)
    else:
        signals = signals.copy()

    if normalize:
        # remove mean if not already detrended
        if not detrend:
            signals -= signals.mean(axis=0)

        std = np.sqrt((signals ** 2).sum(axis=0))
        std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
        signals /= std
    return signals


def _detrend(signals, inplace=False, type="linear"):
    """Detrend timeseries in signals.

    Timeseries are supposed to be columns of `signals`.
    This function is significantly faster than scipy.signal.detrend.

    Parameters
    ==========
    signals (2D numpy array)
        timeseries to detrend. A timeseries is a column.

    inplace (boolean)
        tells if the computation must be made inplace or not (default
        False).

    type (string)
        detrending type ("linear" or "constant").
        See also scipy.signal.detrend.

    Returns
    =======
    detrended_signals (2D numpy array)
        detrended timeseries.
    """
    if not inplace:
        signals = signals.copy()

    signals -= np.mean(signals, axis=0)
    if type == "linear":
        regressor = np.arange(signals.shape[0]).astype(np.float)
        regressor -= regressor.mean()
        regressor /= np.sqrt((regressor ** 2).sum())
        signals -= np.dot(regressor, signals) * regressor[:, np.newaxis]
    return signals


def butterworth(signals, sampling_rate, low_pass=None, high_pass=None,
                order=5, copy=False, save_memory=False):
    """ Apply a low pass, high pass or band pass butterworth filter

    Apply a filter to remove signal below the `low` frequency and above the
    `high`frequency.

    Parameters
    ----------
    signals: numpy array (1D sequence or n_samples x n_sources)
        Timeseries to be filtered. A timeseries is assumed to be a
        column of `signals`.

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

    copy: boolean, optional
        If False, `signals` is modified inplace, and memory consumption is
        lower than for copy=True, though computation time is higher.

    Returns
    -------
    filtered_signals: numpy array
        Signals filtered according to the parameters
    """
    if low_pass is None and high_pass is None:
        if copy:
            return signal.copy()
        else:
            return signal

    if low_pass is not None and high_pass is not None \
                            and high_pass >= low_pass:
        raise ValueError(
            "High pass cutoff frequency (%f) is greater or equal"
            "to low pass filter frequency (%f). This case is not handled "
            "by this function."
            % (high_pass, low_pass))

    nyq = sampling_rate * 0.5

    wn = None
    if low_pass is not None:
        lf = low_pass / nyq
        btype = 'low'
        wn = lf

    if high_pass is not None:
        hf = high_pass / nyq
        btype = 'high'
        wn = hf

    if low_pass is not None and high_pass is not None:
        btype = 'band'
        wn = [hf, lf]

    b, a = signal.butter(order, wn, btype=btype)
    if signals.ndim == 1:
        # 1D case
        output = signal.lfilter(b, a, signals)
        if copy:  # lfilter does a copy in all cases.
            signals = output
        else:
            signals[...] = output
    else:
        # lfilter() leaks memory in scipy 0.7.0.
        if copy:
            # No way to save memory when a copy has been requested,
            # because lfilter does out-of-place processing
            signals = signal.lfilter(b, a, signals, axis=0)
        else:
            # Lesser memory consumption, slower.
            for timeseries in signals.T:
                timeseries[:] = signal.lfilter(b, a, timeseries)
    return signals


def high_variance_confounds(series, n_confounds=10, percentile=1.):
    """ Return confounds time series extracted from series with highest
        variance.

        Parameters
        ==========
        series (numpy.ndarray)
            Timeseries. A timeseries is a column in the "series" array.
            shape (sample number, feature number)

        n_confounds (int)
            Number of confounds to return

        percentile (float)
            Highest-variance series percentile to keep before computing the
            singular value decomposition.
            series.shape[0] * percentile must be greater than n_confounds.

        Returns
        =======
        v (numpy.ndarray)
            highest variance confounds. Shape: (samples, n_confounds)

        Notes
        ======
        This method is related to what has been published in the literature
        as 'CompCor' (Behzadi NeuroImage 2007).

        The implemented algorithm does the following:
        - compute sum of squares for each time series (no mean removal)
        - keep a given percentile of series with highest variances (percentile)
        - compute an svd of the extracted series
        - return a given number (n_confounds) of series from the svd with
          highest singular values.
    """
    # Retrieve the voxels|features with highest variance

    # Compute variance without mean removal.
    # The execution speed of these three lines is independent of array
    # ordering (C or F)
    var = np.copy(series)
    var **= 2
    var = var.mean(axis=0)

    var_thr = stats.scoreatpercentile(var, 100. - percentile)
    series = series[:, var > var_thr]  # extract columns (i.e. features)
    # Return the singular vectors with largest singular values
    u, _, _ = linalg.svd(series, full_matrices=False)
    u = u[:, :n_confounds].copy()
    return u


def clean(signals, detrend=True, standardize=True, confounds=None,
          low_pass=None, high_pass=None, t_r=2.5):
    """Improve SNR on masked fMRI signals.

       This function can do several things on the input signals, in
       the following order:
       - detrend
       - standardize
       - remove confounds
       - low- and high-pass filter

       Low-pass filtering improves specificity.

       High-pass filtering should be kept small, to keep some
       sensitivity.

       Filtering is only meaningful on evenly-sampled timeseries.

       Parameters
       ==========
       signals (numpy array)
           Timeseries. Must have shape (instant number, features number).
           This array is not modified.

       confounds (numpy array or file name)
           Confounds timeseries. Shape muse be
           (instant number, confound number). The number of time
           instants in signals and confounds must be identical
           (i.e. signals.shape[0] == confounds.shape[0])

       t_r (float)
           Repetition time, in second (sampling period).

       low_pass, high_pass (float)
           Respectively low and high cutoff frequencies, in Hertz.

       detrend (boolean)
           If detrending should be applied on timeseries (before
           confound removal)

       standardize (boolean)
           If variances should be set to one and mean to zero for
           all timeseries (before confound removal)

       Returns
       =======
       cleaned_signals (numpy array)
           Input signals, cleaned. Same shape as `signals`.

       Notes
       =====
       Confounds removal is based on a projection on the orthogonal
       of the signal space. See `Friston, K. J., A. P. Holmes,
       K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak.
       "Statistical Parametric Maps in Functional Imaging: A General
       Linear Approach". Human Brain Mapping 2, no 4 (1994): 189-210.
       <http://dx.doi.org/10.1002/hbm.460020402>`_
    """

    # Standardize / detrend
    signals = _standardize(signals, normalize=standardize, detrend=detrend)

    # Remove confounds
    if confounds is not None:
        if isinstance(confounds, basestring):
            filename = confounds
            confounds = np.genfromtxt(filename)
            if np.isnan(confounds.flat[0]):
                # There may be a header
                if np.version.short_version >= '1.4.0':
                    confounds = np.genfromtxt(filename, skip_header=1)
                else:
                    confounds = np.genfromtxt(filename, skiprows=1)
        # Restrict the signal to the orthogonal of the confounds
        confounds = np.atleast_2d(confounds)
        confounds = _standardize(confounds, normalize=True)
        Q = qr_economic(confounds)[0]
        signals -= np.dot(Q, np.dot(Q.T, signals))

    if low_pass is not None or high_pass is not None:
        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)

    return signals
