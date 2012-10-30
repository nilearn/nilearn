"""
Preprocessing functions for time series.
"""
# Authors: Alexandre Abraham, Gael Varoquaux
# License: simplified BSD

from scipy import signal, linalg, fftpack
import numpy as np


def standardize(signals, copy=True, normalize=True):
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
                         high_pass=False, detrend=False,
                         normalize=True,
                         shift_confounds=False):
    """ Normalize the signal, and if any confounds are given, project in
        the orthogonal space.

        Low pass filter improves specificity (more interesting arrows
        selected)

        High pass filter should be kepts small, so as not to kill
        sensitivity
    """
    signals = standardize(signals, normalize=normalize)

    if confounds is not None:
        if isinstance(confounds, basestring):
            filename = confounds
            confounds = np.genfromtxt(filename)
            if np.isnan(confounds.flat[0]):
                # There may be a header
                del confounds
                confounds = np.genfromtxt(filename, skip_header=1)
        # Restrict the signal to the orthogonal of the confounds
        confounds = np.atleast_2d(confounds)
        if shift_confounds:
            confounds = np.r_[confounds[..., 1:-1],
                              confounds[..., 2:],
                              confounds[..., :-2]]
            signals = signals[..., 1:-1]
        confounds = standardize(confounds, normalize=True)
        #confounds = linalg.svd(confounds, full_matrices=False)[-1]
        confounds = linalg.qr(confounds, mode='economic')[0].T
        #y = y - np.dot(np.dot(confounds, y), confounds)
        signals -= np.dot(np.dot(signals, confounds.T), confounds)

    if low_pass and high_pass and high_pass >= low_pass:
        raise ValueError("Your value for high pass filter (%f) is higher or"
        " equal to the value for low pass filter (%f). This would result in a"
        " blank signal" % (high_pass, low_pass))

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
        for s in signals:
            s[:] = signal.detrend(s)

    signals = standardize(signals, normalize=normalize)
    return signals
