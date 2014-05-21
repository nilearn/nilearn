"""
Preprocessing functions for time series.

All functions in this module should take X matrices with samples x
features
"""
# Authors: Alexandre Abraham, Gael Varoquaux, Philippe Gervais
# License: simplified BSD

import distutils.version

import numpy as np
from scipy import signal, stats, linalg
from sklearn.utils import gen_even_slices

np_version = distutils.version.LooseVersion(np.version.short_version).version


def _standardize(signals, detrend=False, normalize=True):
    """ Center and norm a given signal (time is along first axis)

    Parameters
    ==========
    signals: numpy.ndarray
        Timeseries to standardize

    detrend: bool
        if detrending of timeseries is requested

    normalize: bool
        if True, shift timeseries to zero mean value and scale
        to unit energy (sum of squares).

    Returns
    =======
    std_signals: numpy.ndarray
        copy of signals, normalized.
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


def _mean_of_squares(signals, n_batches=20):
    """Compute mean of squares for each signal.
    This function is equivalent to

        var = np.copy(signals)
        var **= 2
        var = var.mean(axis=0)

    but uses a lot less memory.

    Parameters
    ==========
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
    ==========
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
    =======
    detrended_signals: numpy.ndarray
        Detrended signals. The shape is that of 'signals'.
    """
    if not inplace:
        signals = signals.copy()

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


def butterworth(signals, sampling_rate, low_pass=None, high_pass=None,
                order=5, copy=False, save_memory=False):
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
        if copy:
            # No way to save memory when a copy has been requested,
            # because lfilter does out-of-place processing
            signals = signal.lfilter(b, a, signals, axis=0)
        else:
            # Lesser memory consumption, slower.
            for timeseries in signals.T:
                timeseries[:] = signal.lfilter(b, a, timeseries)
    return signals


def high_variance_confounds(series, n_confounds=5, percentile=2.,
                            detrend=True):
    """ Return confounds time series extracted from series with highest
        variance.

        Parameters
        ==========
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
        =======
        v: numpy.ndarray
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

        See also
        ========
        nilearn.image.high_variance_confounds
    """

    if detrend:
        series = _detrend(series)  # copy

    # Retrieve the voxels|features with highest variance

    # Compute variance without mean removal.
    var = _mean_of_squares(series)

    var_thr = stats.scoreatpercentile(var, 100. - percentile)
    series = series[:, var > var_thr]  # extract columns (i.e. features)
    # Return the singular vectors with largest singular values
    # We solve the symmetric eigenvalue problem here, increasing stability
    s, u = linalg.eigh(series.dot(series.T) / series.shape[0])
    ix_ = np.argsort(s)[::-1]
    u = u[:, ix_[:n_confounds]].copy()
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

       Filtering is only meaningful on evenly-sampled signals.

       Parameters
       ==========
       signals: numpy.ndarray
           Timeseries. Must have shape (instant number, features number).
           This array is not modified.

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
           Repetition time, in second (sampling period).

       low_pass, high_pass: float
           Respectively low and high cutoff frequencies, in Hertz.

       detrend: bool
           If detrending should be applied on timeseries (before
           confound removal)

       standardize: bool
           If True, returned signals are set to unit variance.

       Returns
       =======
       cleaned_signals: numpy.ndarray
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

    if not isinstance(confounds,
                      (list, tuple, basestring, np.ndarray, type(None))):
        raise TypeError("confounds keyword has an unhandled type: %s"
                        % confounds.__class__)

    # Standardize / detrend
    normalize = False
    if confounds is not None:
        # If confounds are to be removed, then force normalization to improve
        # matrix conditioning.
        normalize = True
    signals = _standardize(signals, normalize=normalize, detrend=detrend)

    # Remove confounds
    if confounds is not None:
        if not isinstance(confounds, (list, tuple)):
            confounds = (confounds, )

        # Read confounds
        all_confounds = []
        for confound in confounds:
            if isinstance(confound, basestring):
                filename = confound
                confound = np.genfromtxt(filename)
                if np.isnan(confound.flat[0]):
                    # There may be a header
                    if np_version >= [1, 4, 0]:
                        confound = np.genfromtxt(filename, skip_header=1)
                    else:
                        confound = np.genfromtxt(filename, skiprows=1)
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
        confounds = _standardize(confounds, normalize=True, detrend=detrend)
        Q = linalg.qr(confounds, mode='economic')[0]
        signals -= np.dot(Q, np.dot(Q.T, signals))

    if low_pass is not None or high_pass is not None:
        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)

    if standardize:
        signals = _standardize(signals, normalize=True, detrend=False)
        signals *= np.sqrt(signals.shape[0])  # for unit variance

    return signals
