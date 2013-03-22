"""
Preprocessing functions for time series.
"""
# Authors: Alexandre Abraham, Gael Varoquaux
# License: simplified BSD

from scipy import signal
import numpy as np
from sklearn.utils.fixes import qr_economic


def _standardize(signals, copy=True, normalize=True):
    """ Center and norm a given signal (sample = axis -1)
    """
    signals = np.array(signals, copy=copy).astype(np.float)
    length = float(signals.shape[-1])
    buffer = signals.T
    buffer -= signals.mean(axis=-1)
    if normalize:
        std = np.sqrt(length) * signals.std(axis=-1).T
        std[std == 0] = 1
        buffer /= std
    return signals


def butterworth(signals, sampling_rate, low_pass=None, high_pass=None,
                order=5, copy=False):
    """ Apply a low pass, high pass or band pass butterworth filter

    Apply a filter to remove signal below the `low` frequency and above the
    `high`frequency.

    Parameters
    ----------
    signals: numpy array (1D sequence or n_sources x time_series)
        Signals to be filtered

    sampling_rate: float
        Number of samples per time unit (sample frequency)

    low_pass: float, optional
        If specified, signals above this frequency will be filtered (low pass)

    high_pass: float, optional
        If specified, signals below this frequency will be filtered (high pass)

    order: integer, optional
        Order of the butterworth filter. When filtering signals, the
        butterworth filter has a decay to avoid ringing. Increase the order
        sharpens the decay.

    copy: boolean, optional
        If false, apply filter inplace.

    Returns
    -------
    filtered_signals: numpy array
        Signals filtered according to the parameters
    """
    nyq = sampling_rate * 0.5

    if low_pass is None and high_pass is None:
        return signal

    if low_pass is not None and high_pass is not None \
                            and high_pass >= low_pass:
        raise ValueError("Your value for high pass filter (%f) is higher or"
                         " equal to the value for low pass filter (%f). This"
                         " would result in a blank signal"
                         % (high_pass, low_pass))

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
    if len(signals.shape) == 1:
        # 1D case
        if copy:
            signals = signals.copy()
        signals[:] = signal.lfilter(b, a, signals)
    else:
        if copy:
            # copy by chunks to avoid huge memory allocation
            signals_copy = []
            for i in range(signals.shape[0]):
                signals_copy.append(np.zeros(signals.shape[1:]))
            signals = np.asarray(signals_copy)
        for s in signals:
            s[:] = signal.lfilter(b, a, s)
    return signals


def clean(signals, confounds=None, t_r=2.5, low_pass=None,
          high_pass=None, detrend=False, standardize=True,
          shift_confounds=False):
    """ Normalize the signal, and if any confounds are given, project in
        the orthogonal space.

        Low pass filter improves specificity (more interesting arrows
        selected)

        High pass filter should be kept small, so as not to kill
        sensitivity
    """
    if standardize:
        signals = _standardize(signals, normalize=True)
    elif detrend:
        signals = _standardize(signals, normalize=False)
    signals = np.asarray(signals)

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
        if shift_confounds:
            confounds = np.r_[confounds[..., 1:-1],
                              confounds[..., 2:],
                              confounds[..., :-2]]
            signals = signals[..., 1:-1]
        confounds = _standardize(confounds, normalize=True)
        confounds = qr_economic(confounds)[0].T
        signals -= np.dot(np.dot(signals, confounds.T), confounds)

    if low_pass is not None or high_pass is not None:
        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)

    if detrend:
        # This is faster than scipy.detrend and equivalent
        regressor = np.arange(signals.shape[1]).astype(np.float)
        regressor -= regressor.mean()
        regressor /= np.sqrt((regressor ** 2).sum())

        signals -= np.dot(signals, regressor)[:, np.newaxis] * regressor

    if standardize:
        signals = _standardize(signals, normalize=True)
    elif detrend:
        signals = _standardize(signals, normalize=False)
    return signals
