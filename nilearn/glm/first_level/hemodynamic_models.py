"""Hemodynamic response function (hrf) specification.

Here we provide for SPM, Glover hrfs and finite timpulse response (FIR) models.
This module closely follows SPM implementation

Author: Bertrand Thirion, 2011--2018
"""

import warnings
from collections.abc import Iterable

import numpy as np
from scipy.stats import gamma

from nilearn._utils import fill_doc


def _gamma_difference_hrf(
    tr,
    oversampling=50,
    time_length=32.0,
    onset=0.0,
    delay=6,
    undershoot=16.0,
    dispersion=1.0,
    u_dispersion=1.0,
    ratio=0.167,
):
    """Compute an hrf as the difference of two gamma functions.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor.
        Default=50.

    time_length : float, optional
        hrf kernel length, in seconds.
        Default=32.

    onset : float, optional
        Onset time of the hrf. Default=0.

    delay : float, optional
        Delay parameter of the hrf (in s.).
        Default=6s.

    undershoot : float, optional
        Undershoot parameter of the hrf (in s.).
        Default=16s.

    dispersion : float, optional
        Dispersion parameter for the first gamma function.
        Default=1.

    u_dispersion : float, optional
        Dispersion parameter for the second gamma function.
        Default=1.

    ratio : float, optional
        Ratio of the two gamma components. Default=0.167.

    Returns
    -------
    hrf : array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid

    """
    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(int)
    )
    time_stamps -= onset

    # define peak and undershoot gamma functions
    peak_gamma = gamma.pdf(
        time_stamps, delay / dispersion, loc=dt, scale=dispersion
    )
    undershoot_gamma = gamma.pdf(
        time_stamps, undershoot / u_dispersion, loc=dt, scale=u_dispersion
    )

    # calculate the hrf
    hrf = peak_gamma - ratio * undershoot_gamma
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=50, time_length=32.0, onset=0.0):
    """Implement the SPM hrf model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        hrf onset time, in seconds. Default=0.

    Returns
    -------
    hrf : array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid

    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset)


def glover_hrf(tr, oversampling=50, time_length=32.0, onset=0.0):
    """Implement the Glover hrf model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response. Default=0.

    Returns
    -------
    hrf : array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid.

    """
    return _gamma_difference_hrf(
        tr,
        oversampling,
        time_length,
        onset,
        delay=6,
        undershoot=12.0,
        dispersion=0.9,
        u_dispersion=0.9,
        ratio=0.35,
    )


def _compute_derivative_from_values(values, values_plus_dt, dt=0.1):
    """Return the time or dispersion derivative of an hrf."""
    return 1.0 / dt * (values - values_plus_dt)


def _generic_time_derivative(
    func, tr, oversampling=50, time_length=32.0, onset=0.0, dt=0.1
):
    """Return the time derivative of an hrf for a given function.

    Parameters
    ----------
    func : :obj:`function`
        spm_hrf or glover_hrf

    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response. Default=0.

    dt : float, optional
        Time step for the derivative. Default=0.1.
    """
    return _compute_derivative_from_values(
        func(tr, oversampling, time_length, onset),
        func(tr, oversampling, time_length, onset + dt),
        dt=dt,
    )


def spm_time_derivative(tr, oversampling=50, time_length=32.0, onset=0.0):
    """Implement the SPM time derivative hrf (dhrf) model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response in seconds. Default=0.

    Returns
    -------
    dhrf : array of shape(length / tr, dtype=float)
          dhrf sampling on the provided grid

    """
    return _generic_time_derivative(
        spm_hrf,
        tr=tr,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )


def glover_time_derivative(tr, oversampling=50, time_length=32.0, onset=0.0):
    """Implement the Glover time derivative hrf (dhrf) model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response. Default=0.

    Returns
    -------
    dhrf : array of shape(length / tr), dtype=float
          dhrf sampling on the provided grid

    """
    return _generic_time_derivative(
        glover_hrf,
        tr=tr,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )


def _generic_dispersion_derivative(
    tr,
    oversampling=50,
    time_length=32.0,
    onset=0.0,
    undershoot=16,
    ratio=0.167,
    dispersion=1.0,
    dt=0.01,
):
    """Return the dispersion derivative of an hrf.

    Parameters
    ----------
    dt : float, optional
        Dispersion step for the derivative. Default=0.01.

    See _gamma_difference_hrf for the other parameters description.
    """
    return _compute_derivative_from_values(
        _gamma_difference_hrf(
            tr,
            oversampling,
            time_length,
            onset,
            undershoot=undershoot,
            ratio=ratio,
            dispersion=dispersion,
        ),
        _gamma_difference_hrf(
            tr,
            oversampling,
            time_length,
            onset,
            undershoot=undershoot,
            ratio=ratio,
            dispersion=dispersion + dt,
        ),
        dt=dt,
    )


def spm_dispersion_derivative(
    tr, oversampling=50, time_length=32.0, onset=0.0
):
    """Implement the SPM dispersion derivative hrf model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor in seconds. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response in seconds. Default=0.

    Returns
    -------
    dhrf : array of shape(length / tr * oversampling), dtype=float
          dhrf sampling on the oversampled time grid

    """
    return _generic_dispersion_derivative(
        tr, oversampling=oversampling, time_length=time_length, onset=onset
    )


def glover_dispersion_derivative(
    tr, oversampling=50, time_length=32.0, onset=0.0
):
    """Implement the Glover dispersion derivative hrf model.

    Parameters
    ----------
    tr : float
        Scan repeat time, in seconds.

    oversampling : int, optional
        Temporal oversampling factor in seconds. Default=50.

    time_length : float, optional
        hrf kernel length, in seconds. Default=32.

    onset : float, optional
        Onset of the response in seconds. Default=0.

    Returns
    -------
    dhrf : array of shape(length / tr * oversampling), dtype=float
          dhrf sampling on the oversampled time grid

    """
    return _generic_dispersion_derivative(
        tr,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
        undershoot=12.0,
        ratio=0.35,
        dispersion=0.9,
    )


def _sample_condition(
    exp_condition, frame_times, oversampling=50, min_onset=-24
):
    """Make a possibly oversampled event regressor from condition information.

    Parameters
    ----------
    exp_condition : arraylike of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet

    frame_times : array of shape(n_scans)
        Sample time points.

    oversampling : int, optional
        Factor for oversampling event regressor. Default=50.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    Returns
    -------
    regressor : array of shape(over_sampling * n_scans)
        Possibly oversampled event regressor.

    frame_times_high_res : array of shape(over_sampling * n_scans)
        Time points used for regressor sampling.

    """
    # Find the high-resolution frame_times
    n_frames = frame_times.size
    min_onset = float(min_onset)
    n_frames_high_res = _compute_n_frames_high_res(
        frame_times, min_onset, oversampling
    )

    frame_times_high_res = np.linspace(
        frame_times.min() + min_onset,
        frame_times.max() * (1 + 1.0 / (n_frames - 1)),
        np.rint(n_frames_high_res).astype(int),
    )

    # Get the condition information
    onsets, durations, values = tuple(map(np.asanyarray, exp_condition))
    if (onsets < frame_times[0] + min_onset).any():
        warnings.warn(
            (
                "Some stimulus onsets are earlier "
                f"than {frame_times[0] + min_onset} in the"
                " experiment and are thus not considered in the model."
            ),
            UserWarning,
        )

    # Set up the regressor timecourse
    tmax = len(frame_times_high_res)
    regressor = np.zeros_like(frame_times_high_res).astype(np.float64)
    t_onset = np.minimum(
        np.searchsorted(frame_times_high_res, onsets), tmax - 1
    )
    for t, v in zip(t_onset, values):
        regressor[t] += v
    t_offset = np.minimum(
        np.searchsorted(frame_times_high_res, onsets + durations), tmax - 1
    )

    # Handle the case where duration is 0 by offsetting at t + 1
    for i, t in enumerate(t_offset):
        if t < (tmax - 1) and t == t_onset[i]:
            t_offset[i] += 1

    for t, v in zip(t_offset, values):
        regressor[t] -= v
    regressor = np.cumsum(regressor)

    return regressor, frame_times_high_res


def _compute_n_frames_high_res(frame_times, min_onset, oversampling):
    """Compute the number of frames after upsampling."""
    n_frames = frame_times.size
    mini, maxi = _extrema(frame_times)
    n_frames_high_res = (n_frames - 1) * 1.0 / (maxi - mini)
    n_frames_high_res *= (
        maxi * (1 + 1.0 / (n_frames - 1)) - mini - min_onset
    ) * oversampling
    return n_frames_high_res + 1


def _extrema(arr):
    """Return the min and max of an array."""
    return np.min(arr), np.max(arr)


def _resample_regressor(hr_regressor, frame_times_high_res, frame_times):
    """Sub-sample the regressors at frame times.

    Parameters
    ----------
    hr_regressor : array of shape(n_samples),
        the regressor time course sampled at high temporal resolution

    frame_times_high_res : array of shape(n_samples),
        the corresponding time stamps

    frame_times : array of shape(n_scans),
         the desired time stamps

    Returns
    -------
    regressor : array of shape(n_scans)
         The resampled regressor.

    """
    from scipy.interpolate import interp1d

    f = interp1d(frame_times_high_res, hr_regressor)
    return f(frame_times).T


def _orthogonalize(X):
    """Orthogonalize every column of design `X` w.r.t preceding columns.

    Parameters
    ----------
    X : array of shape(n, p)
       The data to be orthogonalized.

    Returns
    -------
    X : array of shape(n, p)
       The data after orthogonalization.

    Notes
    -----
    X is changed in place. The columns are not normalized.

    """
    if X.size == X.shape[0]:
        return X

    from scipy.linalg import pinv

    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))

    return X


@fill_doc
def _regressor_names(con_name, hrf_model, fir_delays=None):
    """Return a list of regressor names, \
    computed from con-name and hrf type \
    when this information is explicitly given.

    If hrf_model is a custom function or a list of custom functions,
    return their names.

    Parameters
    ----------
    con_name : string
        identifier of the condition
    %(hrf_model)s
    fir_delays : 1D array_like, optional
        Delays (in scans) used in case of an FIR model

    Returns
    -------
    names : list of strings,
        regressor names

    """
    # Default value
    names = [con_name]

    # Handle strings
    if hrf_model in ["glover", "spm"]:
        names = [con_name]
    elif hrf_model in ["glover + derivative", "spm + derivative"]:
        names = [con_name, f"{con_name}_derivative"]
    elif hrf_model in [
        "spm + derivative + dispersion",
        "glover + derivative + dispersion",
    ]:
        names = [con_name, f"{con_name}_derivative", f"{con_name}_dispersion"]
    elif hrf_model == "fir":
        names = [f"{con_name}_delay_{int(i)}" for i in fir_delays]
    # Handle callables
    elif callable(hrf_model):
        names = [f"{con_name}_{hrf_model.__name__}"]
    elif isinstance(hrf_model, Iterable) and all(
        [callable(_) for _ in hrf_model]
    ):
        names = [f"{con_name}_{model.__name__}" for model in hrf_model]
    # Handle some default cases
    elif isinstance(hrf_model, Iterable) and not isinstance(hrf_model, str):
        names = [f"{con_name}_{i}" for i in range(len(hrf_model))]

    # Check that all names within the list are different
    if len(np.unique(names)) != len(names):
        raise ValueError(f"Computed regressor names are not unique: {names}")

    return names


def _hrf_kernel(hrf_model, tr, oversampling=50, fir_delays=None):
    """Return the list of matching kernels \
    given the specification of the hemodynamic model and time parameters.

    Parameters
    ----------
    hrf_model : string, function, list of functions, or None,
        HRF model to be used.

    tr : float
        the repetition time in seconds

    oversampling : int, optional
        Temporal oversampling factor to have a smooth hrf.
        Default=50.

    fir_delays : 1D-array-like, optional
        List of delays (in scans) for finite impulse response models.

    Returns
    -------
    hkernel : list of arrays
        Samples of the hrf (the number depends on the hrf_model used).

    """
    acceptable_hrfs = [
        "spm",
        "spm + derivative",
        "spm + derivative + dispersion",
        "fir",
        "glover",
        "glover + derivative",
        "glover + derivative + dispersion",
        None,
    ]
    error_msg = (
        "Could not process custom HRF model provided. "
        "Please refer to the related documentation."
    )
    if hrf_model == "spm":
        hkernel = [spm_hrf(tr, oversampling)]
    elif hrf_model == "spm + derivative":
        hkernel = [
            spm_hrf(tr, oversampling),
            spm_time_derivative(tr, oversampling),
        ]
    elif hrf_model == "spm + derivative + dispersion":
        hkernel = [
            spm_hrf(tr, oversampling),
            spm_time_derivative(tr, oversampling),
            spm_dispersion_derivative(tr, oversampling),
        ]
    elif hrf_model == "glover":
        hkernel = [glover_hrf(tr, oversampling)]
    elif hrf_model == "glover + derivative":
        hkernel = [
            glover_hrf(tr, oversampling),
            glover_time_derivative(tr, oversampling),
        ]
    elif hrf_model == "glover + derivative + dispersion":
        hkernel = [
            glover_hrf(tr, oversampling),
            glover_time_derivative(tr, oversampling),
            glover_dispersion_derivative(tr, oversampling),
        ]
    elif hrf_model == "fir":
        hkernel = [
            np.hstack(
                (
                    np.zeros((f) * oversampling),
                    np.ones(oversampling) * 1.0 / oversampling,
                )
            )
            for f in fir_delays
        ]
    elif callable(hrf_model):
        try:
            hkernel = [hrf_model(tr, oversampling)]
        except TypeError:
            raise ValueError(error_msg)
    elif isinstance(hrf_model, Iterable) and all(
        [callable(_) for _ in hrf_model]
    ):
        try:
            hkernel = [model(tr, oversampling) for model in hrf_model]
        except TypeError:
            raise ValueError(error_msg)
    elif hrf_model is None:
        hkernel = [np.hstack((1, np.zeros(oversampling - 1)))]
    else:
        raise ValueError(
            f'"{hrf_model}" is not a known hrf model. '
            "Use either a custom model or "
            f"one of {acceptable_hrfs}"
        )
    return hkernel


@fill_doc
def compute_regressor(
    exp_condition,
    hrf_model,
    frame_times,
    con_id="cond",
    oversampling=50,
    fir_delays=None,
    min_onset=-24,
):
    """Convolve regressors with hrf model.

    Parameters
    ----------
    exp_condition : array-like of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet
    %(hrf_model)s
    frame_times : array of shape (n_scans)
        the desired sampling times

    con_id : string, optional, default is 'cond'.
        Identifier of the condition

    oversampling : int, optional
        Oversampling factor to perform the convolution. Default=50.

    fir_delays : [int] 1D-array-like, optional
        Delays (in scans) used in case of a finite impulse response model.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    Returns
    -------
    computed_regressors : array of shape(n_scans, n_reg)
        Computed regressors sampled at frame times.

    reg_names : list of strings
        Corresponding regressor names.

    """
    # fir_delays should be integers
    if fir_delays is not None:
        fir_delays = [int(x) for x in fir_delays]
    oversampling = int(oversampling)

    # this is the minimal tr in this session, not necessarily the true tr
    tr = _calculate_tr(frame_times)
    # 1. create the high temporal resolution regressor
    hr_regressor, frame_times_high_res = _sample_condition(
        exp_condition, frame_times, oversampling, min_onset
    )

    # 2. create the  hrf model(s)
    hkernel = _hrf_kernel(hrf_model, tr, oversampling, fir_delays)

    # 3. convolve the regressor and hrf, and downsample the regressor
    conv_reg = np.array(
        [np.convolve(hr_regressor, h)[: hr_regressor.size] for h in hkernel]
    )

    # 4. temporally resample the regressors
    if hrf_model == "fir" and oversampling > 1:
        computed_regressors = _resample_regressor(
            conv_reg[:, oversampling - 1 :],
            frame_times_high_res[: 1 - oversampling],
            frame_times,
        )
    else:
        computed_regressors = _resample_regressor(
            conv_reg, frame_times_high_res, frame_times
        )

    # 5. ortogonalize the regressors
    if hrf_model != "fir":
        computed_regressors = _orthogonalize(computed_regressors)

    # 6 generate regressor names
    reg_names = _regressor_names(con_id, hrf_model, fir_delays=fir_delays)
    return computed_regressors, reg_names


def _calculate_tr(frame_times):
    """Calculate TR from differences in frame_times.

    Parameters
    ----------
    frame_times : array of shape (n_scans)
        the desired sampling times
    Returns
    -------
    float
        repetition time
    """
    return np.min(np.diff(frame_times))
