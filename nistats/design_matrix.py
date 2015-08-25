# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import with_statement
"""
This module implements fMRI Design Matrix creation.

The DesignMatrix object is just a container that represents the design matrix.
Computations of the different parts of the design matrix are confined
to the make_dmtx() function, that instantiates the DesignMatrix object.
All the remainder are just ancillary functions.

Design matrices contain three different types of regressors:

1. Task-related regressors, that result from the convolution
   of the experimental paradigm regressors with hemodynamic models
   A hemodynamic model is one of:
   'spm' : linear filter used in the SPM software
   'glover' : linear filter estimated by G.Glover
   'fir' : finite impulse response model, generic linear filter

2. User-specified regressors, that represent information available on
   the data, e.g. motion parameters, physiological data resampled at
   the acquisition rate, or sinusoidal regressors that model the
   signal at a frequency of interest.

3. Drift regressors, that represent low_frequency phenomena of no
   interest in the data; they need to be included to reduce variance
   estimates.

Author: Bertrand Thirion, 2009-2015
"""
from warnings import warn
import numpy as np
import scipy.linalg as spl
from pandas import DataFrame

from .hemodynamic_models import compute_regressor, _orthogonalize
from .experimental_paradigm import check_paradigm



######################################################################
# Ancillary functions
######################################################################


def _poly_drift(order, frame_times):
    """Create a polynomial drift matrix

    Parameters
    ----------
    order : int,
        number of polynomials in the drift model

    frame_times : array of shape(n_scans),
        time stamps used to sample polynomials

    Returns
    -------
    pol : ndarray, shape(n_scans, order + 1)
         estimated polynomial drifts plus a constant regressor
    """
    order = int(order)
    pol = np.zeros((np.size(frame_times), order + 1))
    tmax = float(frame_times.max())
    for k in range(order + 1):
        pol[:, k] = (frame_times / tmax) ** k
    pol = _orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol


def _cosine_drift(period_cut, frame_times):
    """Create a cosine drift matrix with periods greater or equal to
    period_cut.

    Parameters
    ----------
    period_cut : float
        Cut period of the low-pass filter (in sec)

    frame_times : array of shape(n_scans)
        The sampling times (in sec)

    Returns
    -------
    cosine_drift : array of shape(n_scans, n_drifts)
        Cosine drifts plus a constant regressor at cosine_drift[:, -1]

    Ref: http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II
    """
    n_frames = len(frame_times)
    n_times = np.arange(n_frames)
    hfcut = 1. / period_cut  # input parameter is the period
    dt = frame_times[1] - frame_times[0]
    order = int(np.floor(2 * n_frames * hfcut * dt))
    # s.t. hfcut = 1 / (2 * dt) yields n_frames
    cosine_drift = np.zeros((n_frames, order))
    normalizer = np.sqrt(2.0 / n_frames)

    for k in range(1, order):
        cosine_drift[:, k - 1] = normalizer * np.cos(
            (np.pi / n_frames) * (n_times + .5) * k)

    cosine_drift[:, order - 1] = 1.
    return cosine_drift


def _blank_drift(frame_times):
    """ Create the blank drift matrix

    Returns
    -------
    np.ones_like(frame_times)
    """
    return np.reshape(np.ones_like(frame_times), (np.size(frame_times), 1))


def _make_drift(drift_model, frame_times, order=1, period_cut=128.):
    """Create the drift matrix

    Parameters
    ----------
    drift_model : {'polynomial', 'cosine', 'blank'},
        string that specifies the desired drift model

    frame_times : array of shape(n_scans),
        list of values representing the desired TRs

    order : int, optional,
        order of the drift model (in case it is polynomial)

    period_cut : float, optional (defaults to 128),
        period cut in case of a cosine model (in seconds)

    Returns
    -------
    drift : array of shape(n_scans, n_drifts),
        the drift matrix

    names : list of length(n_drifts),
        the associated names
    """
    drift_model = drift_model.lower()   # for robust comparisons
    if drift_model == 'polynomial':
        drift = _poly_drift(order, frame_times)
    elif drift_model == 'cosine':
        drift = _cosine_drift(period_cut, frame_times)
    elif drift_model == 'blank':
        drift = _blank_drift(frame_times)
    else:
        raise NotImplementedError("Unknown drift model %r" % (drift_model))
    names = []
    for k in range(1, drift.shape[1]):
        names.append('drift_%d' % k)
    names.append('constant')
    return drift, names


def _convolve_regressors(paradigm, hrf_model, frame_times, fir_delays=[0],
                         min_onset=-24):
    """ Creation of  a matrix that comprises
    the convolution of the conditions onset with a certain hrf model

    Parameters
    ----------
    paradigm : DataFrame instance,
        Descriptor of an experimental paradigm
        see nistats.experimental_paradigm to check the specification
        for these to be valid paradigm descriptors

    hrf_model : {'canonical', 'canonical with derivative', 'fir'}
        string that specifies the hemodynamic response function

    frame_times : array of shape(n_scans)
        the targeted timing for the design matrix

    fir_delays : array-like of shape(nb_onsets), optional,
        in case of FIR design, yields the array of delays
        used in the FIR model

    min_onset : float, optional (default: -24),
        minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered

    Returns
    -------
    regressor_matrix : array of shape(n_scans, n_regressors),
        contains the convolved regressors
        associated with the experimental conditions

    regressor_names : list of strings,
        the regressor names, that depend on the hrf model used
        if 'canonical' then this is identical to the input names
        if 'canonical with derivative', then two names are produced for
        input name 'name': 'name' and 'name_derivative'
    """
    regressor_names = []
    regressor_matrix = None
    if hrf_model == 'fir':
        oversampling = 1
    else:
        oversampling = 16

    name, onset, duration, modulation = check_paradigm(paradigm)
    for condition in np.unique(paradigm.name):
        condition_mask = (name == condition)
        exp_condition = (onset[condition_mask],
                         duration[condition_mask],
                         modulation[condition_mask])
        reg, names = compute_regressor(
            exp_condition, hrf_model, frame_times, con_id=condition,
            fir_delays=fir_delays, oversampling=oversampling,
            min_onset=min_onset)
        regressor_names += names
        if regressor_matrix is None:
            regressor_matrix = reg
        else:
            regressor_matrix = np.hstack((regressor_matrix, reg))
    return regressor_matrix, regressor_names


def _full_rank(X, cmax=1e15):
    """ Computes the condition number of X and if it is larger than cmax,
    returns a matrix with a condition number smaller than cmax.

    Parameters
    ----------
    X : array of shape(nrows, ncols)
        input array

    cmax : float, optional (default:1.e-15),
        tolerance for condition number

    Returns
    -------
    X : array of shape(nrows, ncols)
        output array

    cond : float,
        actual condition number
    """
    U, s, V = spl.svd(X, 0)
    smax, smin = s.max(), s.min()
    cond = smax / smin
    if cond < cmax:
        return X, cond

    warn('Matrix is singular at working precision, regularizing...')
    lda = (smax - cmax * smin) / (cmax - 1)
    X = np.dot(U, np.dot(np.diag(s + lda), V))
    return X, cmax


######################################################################
# Design matrix creation
######################################################################


def make_design_matrix(
    frame_times, paradigm=None, hrf_model='canonical',
    drift_model='cosine', period_cut=128, drift_order=1, fir_delays=[0],
    add_regs=None, add_reg_names=None, min_onset=-24):
    """ Generate a design matrix from the input parameters

    Parameters
    ----------
    frame_times : array of shape(n_frames),
        the timing of the scans

    paradigm : DataFrame instance, optional
        description of the experimental paradigm

    hrf_model : string, optional,
        that specifies the hemodynamic response function
        it can be 'canonical', 'canonical with derivative' or 'fir'

    drift_model : string, optional
        specifies the desired drift model,
        to be chosen among 'polynomial', 'cosine', 'blank'

    period_cut : float, optional
        cut period of the low-pass filter in seconds

    drift_order : int, optional
        order of the drift model (in case it is polynomial)

    fir_delays : array of shape(n_onsets) or list, optional,
        in case of FIR design, yields the array of delays used in the FIR model

    add_regs : array of shape(n_frames, n_add_reg), optional
        additional user-supplied regressors

    add_reg_names : list of (n_add_reg) strings, optional
        if None, while n_add_reg > 0, these will be termed
        'reg_%i', i = 0..n_add_reg - 1

    min_onset : float, optional
        minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered

    Returns
    -------
    design_matrix : DataFrame instance,
        holding the computed design matrix
    """
    # check arguments
    # check that additional regressor specification is correct
    n_add_regs = 0
    if add_regs is not None:
        if add_regs.shape[0] == np.size(add_regs):
            add_regs = np.reshape(add_regs, (np.size(add_regs), 1))
        n_add_regs = add_regs.shape[1]
        assert add_regs.shape[0] == np.size(frame_times), ValueError(
            'incorrect specification of additional regressors: '
            'length of regressors provided: %s, number of ' +
            'time-frames: %s' % (add_regs.shape[0], np.size(frame_times)))

    # check that additional regressor names are well specified
    if add_reg_names is None:
        add_reg_names = ['reg%d' % k for k in range(n_add_regs)]
    elif len(add_reg_names) != n_add_regs:
        raise ValueError(
            'Incorrect number of additional regressor names was provided'
            '(%s provided, %s expected) % (len(add_reg_names),'
            'n_add_regs)')

    # computation of the matrix
    names = []
    matrix = np.zeros((frame_times.size, 0))

    # step 1: paradigm-related regressors
    if paradigm is not None:
        # create the condition-related regressors
        matrix, names = _convolve_regressors(
            paradigm, hrf_model.lower(), frame_times, fir_delays, min_onset)

    # step 2: additional regressors
    if add_regs is not None:
        # add user-supplied regressors and corresponding names
        matrix = np.hstack((matrix, add_regs))
        names += add_reg_names

    # step 3: drifts
    drift, dnames = _make_drift(drift_model.lower(), frame_times, drift_order,
                                period_cut)
    matrix = np.hstack((matrix, drift))
    names += dnames

    # step 4: Force the design matrix to be full rank at working precision
    matrix, _ = _full_rank(matrix)

    design_matrix = DataFrame(
        np.hstack((frame_times[:, np.newaxis], matrix)),
        columns=['frame_times'] + names)
    return design_matrix


def check_design_matrix(design_matrix):
    """ Check that the provided DataFrame is indeed a valid design matrix
    descriptor, and returns a triplet of fields

    Parameters
    ----------
    design matrix : pandas DataFrame,
        describes a design matrix

    Returns
    -------
    frame_times : array of shape (n_frames),
        sampling times of the design matrix

    matrix : array of shape (n_frames, n_regressors), dtype='f'
        numerical values for the design matrix

    names : array of shape (n_events), dtype='f'
           per-event onset time (in seconds)
    """
    names = design_matrix.keys()
    if 'frame_times' not in names:
        raise ValueError('The provided DataFrame does not contain the'
                         'mandatory frame_times field')
    frame_times = design_matrix['frame_times']
    names = list(names.drop('frame_times'))
    matrix = design_matrix[names].values
    return frame_times, matrix, names


def plot_design_matrix(design_matrix, rescale=True, ax=None):
    """ Plot a design matrix provided as a DataFrame

    Parameters
    ----------
    design matrix : pandas DataFrame,
        describes a design matrix

    rescale : bool, optional
        rescale columns magnitude for visualization or not

    ax : axis handle, optional
        Handle to axis onto which we will draw design matrix

    Returns
    -------
    ax: axis handle
    """
    # We import _set_mpl_backend because just the fact that we are
    # importing it sets the backend
    from nilearn.plotting import _set_mpl_backend
    # avoid unhappy pyflackes
    _set_mpl_backend
    import matplotlib.pyplot as plt

    # normalize the values per column for better visualization
    _, X, names = check_design_matrix(design_matrix)
    if rescale:
        X = X / np.maximum(1.e-12, np.sqrt(np.sum(X ** 2, 0)))
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    ax.imshow(X, interpolation='Nearest', aspect='auto')
    ax.set_label('conditions')
    ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right')
    return ax
