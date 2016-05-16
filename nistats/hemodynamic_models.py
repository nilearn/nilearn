"""
This module is for hemodynamic reponse function (hrf) specification.
Here we provide for SPM, Glover hrfs and finite timpulse response (FIR) models.
This module closely follows SPM implementation

Author: Bertrand Thirion, 2011--2015
"""

import warnings
import numpy as np
from scipy.stats import gamma


def _gamma_difference_hrf(tr, oversampling=16, time_length=32., onset=0.,
                          delay=6, undershoot=16., dispersion=1.,
                          u_dispersion=1., ratio=0.167):
    """ Compute an hrf as the difference of two gamma functions

    Parameters
    ----------

    tr : float
        scan repeat time, in seconds

    oversampling : int, optional (default=16)
        temporal oversampling factor

    time_length : float, optional (default=32)
        hrf kernel length, in seconds

    onset: float
        onset time of the hrf

    delay: float, optional
        delay parameter of the hrf (in s.)

    undershoot: float, optional
        undershoot parameter of the hrf (in s.)

    dispersion : float, optional
        dispersion parameter for the first gamma function

    u_dispersion : float, optional
        dispersion parameter for the second gamma function

    ratio : float, optional
        ratio of the two gamma components

    Returns
    -------
    hrf : array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, float(time_length) / dt)
    time_stamps -= onset
    hrf = gamma.pdf(time_stamps, delay / dispersion, dt / dispersion) -\
        ratio * gamma.pdf(
        time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the SPM hrf model

    Parameters
    ----------
    tr : float
        scan repeat time, in seconds

    oversampling : int, optional
        temporal oversampling factor

    time_length : float, optional
        hrf kernel length, in seconds

    onset : float, optional
        hrf onset time, in seconds

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset)


def glover_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the Glover hrf model

    Parameters
    ----------
    tr : float
        scan repeat time, in seconds

    oversampling : int, optional
        temporal oversampling factor

    time_length : float, optional
        hrf kernel length, in seconds

    onset : float, optional
        onset of the response

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset,
                                delay=6, undershoot=12., dispersion=.9,
                                u_dispersion=.9, ratio=.35)


def spm_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the SPM time derivative hrf (dhrf) model

    Parameters
    ----------
    tr: float
        scan repeat time, in seconds

    oversampling: int, optional
        temporal oversampling factor, optional

    time_length: float, optional
        hrf kernel length, in seconds

    onset: float, optional
        onset of the response in seconds

    Returns
    -------
    dhrf: array of shape(length / tr, dtype=float)
          dhrf sampling on the provided grid
    """
    do = .1
    dhrf = 1. / do * (spm_hrf(tr, oversampling, time_length, onset) -
                      spm_hrf(tr, oversampling, time_length, onset + do))
    return dhrf


def glover_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the Glover time derivative hrf (dhrf) model

    Parameters
    ----------
    tr: float
        scan repeat time, in seconds
    oversampling: int,
        temporal oversampling factor, optional
    time_length: float,
        hrf kernel length, in seconds
    onset: float,
        onset of the response

    Returns
    -------
    dhrf: array of shape(length / tr), dtype=float
          dhrf sampling on the provided grid
    """
    do = .1
    dhrf = 1. / do * (glover_hrf(tr, oversampling, time_length, onset) -
                      glover_hrf(tr, oversampling, time_length, onset + do))
    return dhrf


def spm_dispersion_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the SPM dispersion derivative hrf model

    Parameters
    ----------
    tr: float
        scan repeat time, in seconds

    oversampling: int, optional
        temporal oversampling factor in seconds

    time_length: float, optional
        hrf kernel length, in seconds

    onset : float, optional
        onset of the response in seconds

    Returns
    -------
    dhrf: array of shape(length / tr * oversampling), dtype=float
          dhrf sampling on the oversampled time grid
    """
    dd = .01
    dhrf = 1. / dd * (
        - _gamma_difference_hrf(tr, oversampling, time_length,
                                onset, dispersion=1. + dd)
        + _gamma_difference_hrf(tr, oversampling, time_length, onset))
    return dhrf


def glover_dispersion_derivative(tr, oversampling=16, time_length=32.,
                                 onset=0.):
    """Implementation of the Glover dispersion derivative hrf model

    Parameters
    ----------
    tr: float
        scan repeat time, in seconds

    oversampling: int, optional
        temporal oversampling factor in seconds

    time_length: float, optional
        hrf kernel length, in seconds

    onset : float, optional
        onset of the response in seconds

    Returns
    -------
    dhrf: array of shape(length / tr * oversampling), dtype=float
          dhrf sampling on the oversampled time grid
    """
    dd = .01
    dhrf = 1. / dd * (
        - _gamma_difference_hrf(
            tr, oversampling, time_length, onset,
            delay=6, undershoot=12., dispersion=.9 + dd, ratio=.35)
        + _gamma_difference_hrf(
            tr, oversampling, time_length, onset, delay=6, undershoot=12.,
            dispersion=.9, ratio=.35))
    return dhrf


def _sample_condition(exp_condition, frame_times, oversampling=16,
                     min_onset=-24):
    """Make a possibly oversampled event regressor from condition information.

    Parameters
    ----------
    exp_condition : arraylike of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet

    frame_times : array of shape(n_scans)
        sample time points

    over_sampling : int, optional
        factor for oversampling event regressor

    min_onset : float, optional
        minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered

    Returns
    -------
    regressor: array of shape(over_sampling * n_scans)
        possibly oversampled event regressor
    hr_frame_times : array of shape(over_sampling * n_scans)
        time points used for regressor sampling
    """
    # Find the high-resolution frame_times
    n = frame_times.size
    min_onset = float(min_onset)
    n_hr = ((n - 1) * 1. / (frame_times.max() - frame_times.min()) *
            (frame_times.max() * (1 + 1. / (n - 1)) - frame_times.min() -
             min_onset) * oversampling) + 1

    hr_frame_times = np.linspace(frame_times.min() + min_onset,
                                 frame_times.max() * (1 + 1. / (n - 1)), n_hr)

    # Get the condition information
    onsets, durations, values = tuple(map(np.asanyarray, exp_condition))
    if (onsets < frame_times[0] + min_onset).any():
        warnings.warn(('Some stimulus onsets are earlier than %d in the' +
                       ' experiment and are thus not considered in the model'
                % (frame_times[0] + min_onset)), UserWarning)

    # Set up the regressor timecourse
    tmax = len(hr_frame_times)
    regressor = np.zeros_like(hr_frame_times).astype(np.float)
    t_onset = np.minimum(np.searchsorted(hr_frame_times, onsets), tmax - 1)
    regressor[t_onset] += values
    t_offset = np.minimum(
        np.searchsorted(hr_frame_times, onsets + durations),
        tmax - 1)

    # Handle the case where duration is 0 by offsetting at t + 1
    for i, t in enumerate(t_offset):
        if t < (tmax - 1) and t == t_onset[i]:
            t_offset[i] += 1

    regressor[t_offset] -= values
    regressor = np.cumsum(regressor)

    return regressor, hr_frame_times


def _resample_regressor(hr_regressor, hr_frame_times, frame_times):
    """ this function sub-samples the regressors at frame times

    Parameters
    ----------
    hr_regressor : array of shape(n_samples),
        the regressor time course sampled at high temporal resolution

    hr_frame_times : array of shape(n_samples),
        the corresponding time stamps

    frame_times: array of shape(n_scans),
         the desired time stamps

    Returns
    -------
    regressor: array of shape(n_scans)
         the resampled regressor
    """
    from scipy.interpolate import interp1d
    f = interp1d(hr_frame_times, hr_regressor)
    return f(frame_times).T


def _orthogonalize(X):
    """ Orthogonalize every column of design `X` w.r.t preceding columns

    Parameters
    ----------
    X: array of shape(n, p)
       the data to be orthogonalized

    Returns
    -------
    X: array of shape(n, p)
       the data after orthogonalization

    Notes
    -----
    X is changed in place. The columns are not normalized
    """
    if X.size == X.shape[0]:
        return X
    from scipy.linalg import pinv, norm
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))
        # X[:, i] /= norm(X[:, i])
    return X


def _regressor_names(con_name, hrf_model, fir_delays=None):
    """ Returns a list of regressor names, computed from con-name and hrf type

    Parameters
    ----------
    con_name: string
        identifier of the condition

    hrf_model: string
       hrf model chosen

    fir_delays: 1D array_like, optional,
        Delays used in case of an FIR model

    Returns
    -------
    names: list of strings,
        regressor names
    """
    if hrf_model in ['glover', 'spm']:
        return [con_name]
    elif hrf_model in ["glover + derivative", 'spm + derivative']:
        return [con_name, con_name + "_derivative"]
    elif hrf_model in ['spm + derivative + dispersion',
                       'glover + derivative + dispersion']:
        return [con_name, con_name + "_derivative", con_name + "_dispersion"]
    elif hrf_model == 'fir':
        return [con_name + "_delay_%d" % i for i in fir_delays]


def _hrf_kernel(hrf_model, tr, oversampling=16, fir_delays=None):
    """ Given the specification of the hemodynamic model and time parameters,
    return the list of matching kernels

    Parameters
    ----------
    hrf_model : string
        identifier of the hrf model

    tr : float
        the repetition time in seconds

    oversampling : int, optional
        temporal oversampling factor to have a smooth hrf

    fir_delays : list of floats,
        list of delays for finite impulse response models

    Returns
    -------
    hkernel : list of arrays
        samples of the hrf (the number depends on the hrf_model used)
    """
    acceptable_hrfs = [
        'spm', 'spm + derivative', 'spm + derivative + dispersion', 'fir',
        'glover', 'glover + derivative', 'glover + derivative + dispersion']
    if hrf_model == 'spm':
        hkernel = [spm_hrf(tr, oversampling)]
    elif hrf_model == 'spm + derivative':
        hkernel = [spm_hrf(tr, oversampling),
                   spm_time_derivative(tr, oversampling)]
    elif hrf_model == 'spm + derivative + dispersion':
        hkernel = [spm_hrf(tr, oversampling),
                   spm_time_derivative(tr, oversampling),
                   spm_dispersion_derivative(tr, oversampling)]
    elif hrf_model == 'glover':
        hkernel = [glover_hrf(tr, oversampling)]
    elif hrf_model == 'glover + derivative':
        hkernel = [glover_hrf(tr, oversampling),
                   glover_time_derivative(tr, oversampling)]
    elif hrf_model == 'glover + derivative + dispersion':
        hkernel = [glover_hrf(tr, oversampling),
                   glover_time_derivative(tr, oversampling),
                   glover_dispersion_derivative(tr, oversampling)]
    elif hrf_model == 'fir':
        hkernel = [np.hstack((np.zeros(f * oversampling),
                              np.ones(oversampling)))
                   for f in fir_delays]
    else:
        raise ValueError('"{0}" is not a known hrf model. Use one of {1}'.
                         format(hrf_model, acceptable_hrfs))
    return hkernel


def compute_regressor(exp_condition, hrf_model, frame_times, con_id='cond',
                      oversampling=16, fir_delays=None, min_onset=-24):
    """ This is the main function to convolve regressors with hrf model

    Parameters
    ----------
    exp_condition : array-like of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet

    hrf_model : {'spm', 'spm + derivative', 'spm + derivative + dispersion',
        'glover', 'glover + derivative', 'fir'}
        Name of the hrf model to be used

    frame_times : array of shape (n_scans)
        the desired sampling times

    con_id : string
        optional identifier of the condition

    oversampling : int, optional
        oversampling factor to perform the convolution

    fir_delays : 1D-array-like, optional
        delays (in seconds) used in case of a finite impulse reponse model

    min_onset : float, optional
        minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered

    Returns
    -------
    computed_regressors: array of shape(n_scans, n_reg)
        computed regressors sampled at frame times

    reg_names: list of strings
        corresponding regressor names

    Notes
    -----
    The different hemodynamic models can be understood as follows:
    'spm': this is the hrf model used in SPM
    'spm + derivative': SPM model plus its time derivative (2 regressors)
    'spm + time + dispersion': idem, plus dispersion derivative (3 regressors)
    'glover': this one corresponds to the Glover hrf
    'glover + derivative': the Glover hrf + time derivative (2 regressors)
    'glover + derivative + dispersion': idem + dispersion derivative
                                        (3 regressors)
    'fir': finite impulse response basis, a set of delayed dirac models
           with arbitrary length. This one currently assumes regularly spaced
           frame times (i.e. fixed time of repetition).
    It is expected that spm standard and Glover model would not yield
    large differences in most cases.

    In case of glover and spm models, the derived regressors are
    orthogonalized wrt the main one.
    """
    # this is the average tr in this session, not necessarily the true tr
    tr = float(frame_times.max()) / (np.size(frame_times) - 1)

    # 1. create the high temporal resolution regressor
    hr_regressor, hr_frame_times = _sample_condition(
        exp_condition, frame_times, oversampling, min_onset)

    # 2. create the  hrf model(s)
    hkernel = _hrf_kernel(hrf_model, tr, oversampling, fir_delays)

    # 3. convolve the regressor and hrf, and downsample the regressor
    conv_reg = np.array([np.convolve(hr_regressor, h)[:hr_regressor.size]
                         for h in hkernel])

    # 4. temporally resample the regressors
    computed_regressors = _resample_regressor(
        conv_reg, hr_frame_times, frame_times)

    # 5. ortogonalize the regressors
    if hrf_model != 'fir':
        computed_regressors = _orthogonalize(computed_regressors)

    # 6 generate regressor names
    reg_names = _regressor_names(con_id, hrf_model, fir_delays=fir_delays)
    return computed_regressors, reg_names
