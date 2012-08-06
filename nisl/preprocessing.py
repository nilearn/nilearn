"""
Preprocessing functions for time series.
"""
from scipy import signal, linalg, fftpack
import numpy as np


def ledoit_wolf(x, return_factor=False):
    """ Estimates the shrunk Ledoit-Wolf covariance matrix.

        Parameters
        ----------
        x: 2D ndarray, shape (p, n)
            The data matrix, with p features and n samples.
        return_factor: boolean, optional
            If return_factor is True, the regularisation_factor is
            returned.

        Returns
        -------
        regularised_cov: 2D ndarray
            Regularized covariance
        regularisation_factor: float
            Regularisation factor

        Notes
        -----
        The regularised covariance is::

            (1 - regularisation_factor)*cov
                    + regularisation_factor*np.identity(n_features)
    """
    n_features, n_samples = x.shape
    if n_features == 1:
        if return_factor:
            return np.atleast_2d(x.std()), 0
        return np.atleast_2d(x.std())
    cov = np.dot(x, x.T) / n_samples
    i = np.identity(n_features)
    mu = np.trace(cov) / n_features
    delta = ((cov - mu * i) ** 2).sum() / n_features
    #beta_ = 1./(n_features*n_samples**2) * sum([
    #        ((np.dot(this_x[:, np.newaxis],
    #            this_x[np.newaxis, :]) - cov)**2).sum()
    #        for this_x in x.T
    #    ])
    x2 = x ** 2
    beta_ = 1. / (n_features * n_samples) * np.sum(
                            np.dot(x2, x2.T) / n_samples - cov ** 2
                )

    beta = min(beta_, delta)
    alpha = delta - beta
    if not return_factor:
        return beta / delta * mu * i + alpha / delta * cov
    else:
        return beta / delta * mu * i + alpha / delta * cov, beta / delta


def standardize(signals, copy=True, normalize=True):
    """ Center and norm a given signal (sample = axis -1)
    """
    # FIXME: Should this be merged with canica.algorithms.center_and_norm
    signals = np.array(signals, copy=copy)
    length = float(signals.shape[-1])
    buffer = signals.T
    buffer -= signals.mean(axis=-1)
    if normalize:
        std = np.sqrt(length) * signals.std(axis=-1).T
        std[std == 0] = 1
        buffer /= std
    return signals


def clean_signals(signals, confounds=None, low_pass=0.2, t_r=2.5,
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
        # Restrict the signal to the orthogonal of the confounds
        confounds = np.atleast_2d(confounds)
        if shift_confounds:
            confounds = np.r_[confounds[..., 1:-1],
                              confounds[..., 2:],
                              confounds[..., :-2]]
            signals = signals[..., 1:-1]
        confounds = standardize(confounds, normalize=True)
        #confounds = linalg.svd(confounds, full_matrices=False)[-1]
        confounds = linalg.qr(confounds.T, econ=True)[0].T
        #y = y - np.dot(np.dot(confounds, y), confounds)
        signals -= np.dot(np.dot(signals, confounds.T), confounds)

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
