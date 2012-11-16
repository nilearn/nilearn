"""
Preprocessing functions for time series.
"""
# Authors: Alexandre Abraham, Gael Varoquaux
# License: simplified BSD

from scipy import fftpack
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


def clean(signals, confounds=None, low_pass=0.2, t_r=2.5,
          high_pass=False, detrend=False, standardize=True,
          shift_confounds=False):
    """ Normalize the signal, and if any confounds are given, project in
        the orthogonal space.

        Low pass filter improves specificity (more interesting arrows
        selected)

        High pass filter should be kepts small, so as not to kill
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

    if low_pass and high_pass and high_pass >= low_pass:
        raise ValueError("Your value for high pass filter (%f) is higher or"
                         " equal to the value for low pass filter (%f). This"
                         " would result in a blank signal"
                         % (high_pass, low_pass))

    if low_pass or high_pass:
        n = signals.shape[-1]
        freq = fftpack.fftfreq(n, d=t_r)
        for s in signals:
            fft = fftpack.fft(s)
            if low_pass:
                fft[np.abs(freq) > low_pass] = 0
            if high_pass:
                fft[np.abs(freq) < high_pass] = 0
            s[:] = fftpack.ifft(fft)

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
