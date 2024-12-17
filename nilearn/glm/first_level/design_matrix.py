"""Implement fMRI Design Matrix creation.

Design matrices are represented by Pandas DataFrames
Computations of the different parts of the design matrix are confined
to the make_first_level_design_matrix function, that create a DataFrame
All the others are ancillary functions.

Design matrices contain three different types of regressors:

1. Task-related regressors, that result from the convolution
   of the experimental paradigm regressors with hemodynamic models
   A hemodynamic model is one of:

         - 'spm' : linear filter used in the SPM software
         - 'glover' : linear filter estimated by G.Glover
         - 'spm + derivative', 'glover + derivative': the same linear models,
            plus their time derivative (2 regressors per condition)
         - 'spm + derivative + dispersion', 'glover + derivative + dispersion':
            idem plus the derivative wrt the dispersion parameter of the hrf
            (3 regressors per condition)
         - 'fir' : finite impulse response model, generic linear filter

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
import pandas as pd

from nilearn._utils import fill_doc
from nilearn._utils.glm import check_and_load_tables
from nilearn.glm._utils import full_rank
from nilearn.glm.first_level.experimental_paradigm import (
    check_events,
    handle_modulation_of_duplicate_events,
)
from nilearn.glm.first_level.hemodynamic_models import (
    compute_regressor,
    orthogonalize,
)

######################################################################
# Ancillary functions
######################################################################


def _poly_drift(order, frame_times):
    """Create a polynomial drift matrix.

    Parameters
    ----------
    order : :obj:`int`,
        Number of polynomials in the drift model.

    frame_times : array of shape(n_scans),
        Time stamps used to sample polynomials.

    Returns
    -------
    pol : ndarray, shape(n_scans, order + 1)
         Estimated polynomial drifts plus a constant regressor.

    """
    order = int(order)
    pol = np.zeros((np.size(frame_times), order + 1))
    tmax = float(frame_times.max())
    for k in range(order + 1):
        pol[:, k] = (frame_times / tmax) ** k
    pol = orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol


def create_cosine_drift(high_pass, frame_times):
    """Create a cosine drift matrix with frequencies or equal to high_pass.

    Parameters
    ----------
    high_pass : :obj:`float`
        Cut frequency of the high-pass filter in Hz

    frame_times : array of shape (n_scans,)
        The sampling times in seconds

    Returns
    -------
    cosine_drift : array of shape(n_scans, n_drifts)
        Cosine drifts plus a constant regressor at cosine_drift[:, -1]

    References
    ----------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II

    """
    n_frames = len(frame_times)
    n_times = np.arange(n_frames)
    dt = (frame_times[-1] - frame_times[0]) / (n_frames - 1)
    if high_pass * dt >= 0.5:
        warn(
            "High-pass filter will span all accessible frequencies "
            "and saturate the design matrix. "
            "You may want to reduce the high_pass value."
            f"The provided value is {high_pass} Hz"
        )
    order = np.minimum(
        n_frames - 1, int(np.floor(2 * n_frames * high_pass * dt))
    )
    cosine_drift = np.zeros((n_frames, order + 1))
    normalizer = np.sqrt(2.0 / n_frames)

    for k in range(1, order + 1):
        cosine_drift[:, k - 1] = normalizer * np.cos(
            (np.pi / n_frames) * (n_times + 0.5) * k
        )

    cosine_drift[:, -1] = 1.0
    return cosine_drift


def _none_drift(frame_times):
    """Create an intercept vector.

    Returns
    -------
    np.ones_like(frame_times)

    """
    return np.reshape(np.ones_like(frame_times), (np.size(frame_times), 1))


def _make_drift(drift_model, frame_times, order, high_pass):
    """Create the drift matrix.

    Parameters
    ----------
    drift_model : {'polynomial', 'cosine', None},
        string that specifies the desired drift model

    frame_times : array of shape(n_scans),
        list of values representing the desired TRs

    order : :obj:`int`, optional,
        order of the drift model (in case it is polynomial)

    high_pass : :obj:`float`, optional,
        high-pass frequency in case of a cosine model (in Hz)

    Returns
    -------
    drift : array of shape(n_scans, n_drifts),
        the drift matrix

    names : :obj:`list` of length(n_drifts),
        the associated names

    """
    if isinstance(drift_model, str):
        drift_model = drift_model.lower()  # for robust comparisons
    if drift_model == "polynomial":
        drift = _poly_drift(order, frame_times)
    elif drift_model == "cosine":
        drift = create_cosine_drift(high_pass, frame_times)
    elif drift_model is None:
        drift = _none_drift(frame_times)
    else:
        raise NotImplementedError(f"Unknown drift model {drift_model!r}")
    names = [f"drift_{int(k)}" for k in range(1, drift.shape[1])]
    names.append("constant")
    return drift, names


def _convolve_regressors(
    events,
    hrf_model,
    frame_times,
    fir_delays=None,
    min_onset=-24,
    oversampling=50,
):
    """Creation of a matrix that comprises \
    the convolution of the conditions onset with a certain hrf model.

    Parameters
    ----------
    events : DataFrame instance,
        Events data describing the experimental paradigm
        see nilearn.glm.first_level.experimental_paradigm to check the
        specification for these to be valid paradigm descriptors

    hrf_model : {'spm', 'spm + derivative', 'spm + derivative + dispersion',
        'glover', 'glover + derivative', 'glover + derivative + dispersion',
        'fir', None}
        String that specifies the hemodynamic response function

    frame_times : array of shape (n_scans,)
        The targeted timing for the design matrix.

    fir_delays : array-like of shape (n_onsets,), default=None
        In case of FIR design, yields the array of delays
        used in the FIR model (in scans).
        Will default to ``[0]`` if ``None`` is passed.

    min_onset : :obj:`float`, default=-24
        Minimal onset relative to frame_times[0] (in seconds) events
        that start before frame_times[0] + min_onset are not considered.

    oversampling : :obj:`int`, default=50
        Oversampling factor used in temporal convolutions.

    Returns
    -------
    regressor_matrix : array of shape (n_scans, n_regressors),
        Contains the convolved regressors associated with the
        experimental conditions.

    regressor_names : :obj:`list` of strings,
        The regressor names, that depend on the hrf model used
        if 'glover' or 'spm' then this is identical to the input names
        if 'glover + derivative' or 'spm + derivative', a second name is output
            i.e. '#name_derivative'
        if 'spm + derivative + dispersion' or
            'glover + derivative + dispersion',
            a third name is used, i.e. '#name_dispersion'
        if 'fir', the regressos are numbered according to '#name_#delay'

    """
    if fir_delays is None:
        fir_delays = [0]
    regressor_names = []
    regressor_matrix = None

    events_copy = check_events(events)
    cleaned_events = handle_modulation_of_duplicate_events(events_copy)

    trial_type = cleaned_events["trial_type"].to_numpy()
    onset = cleaned_events["onset"].to_numpy()
    duration = cleaned_events["duration"].to_numpy()
    modulation = cleaned_events["modulation"].to_numpy()

    for condition in np.unique(trial_type):
        condition_mask = trial_type == condition
        exp_condition = (
            onset[condition_mask],
            duration[condition_mask],
            modulation[condition_mask],
        )
        reg, names = compute_regressor(
            exp_condition,
            hrf_model,
            frame_times,
            con_id=condition,
            fir_delays=fir_delays,
            oversampling=oversampling,
            min_onset=min_onset,
        )

        regressor_names += names
        if regressor_matrix is None:
            regressor_matrix = reg
        else:
            regressor_matrix = np.hstack((regressor_matrix, reg))
    return regressor_matrix, regressor_names


######################################################################
# Design matrix creation
######################################################################


@fill_doc
def make_first_level_design_matrix(
    frame_times,
    events=None,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=None,
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
    oversampling=50,
):
    """Generate a design matrix from the input parameters.

    Parameters
    ----------
    frame_times : array of shape (n_frames,)
        The timing of acquisition of the scans in seconds.

    events : :obj:`pandas.DataFrame` instance, \
             or :obj:`str` or :obj:`pathlib.Path` to a CSV or TSV file, \
             or None, default=None
        Events data that describes the experimental paradigm.
        The resulting DataFrame instance must/may have these keys:

        - ``'onset'``: REQUIRED
            Column to specify the start time of each events in seconds.
            An error is raised if this key is missing.

        - ``'duration'``: REQUIRED
            Column to specify the duration of each events in seconds.

            .. warning::

                Events with a duration of 0 seconds will be modeled
                using a 'delta function'.

        - ``'trial_type'``: OPTIONAL
            Column to specify per-event experimental conditions identifier.
            If missing each event are labeled 'dummy'
            and considered to form a unique condition.

        - ``'modulation'``: OPTIONAL
            Column to specify the amplitude of each events.
            If missing the default is set to ones(n_events).

        An experimental paradigm is valid if it has an ``'onset'`` key
        and a ``'duration'`` key.
        If these keys are missing an error will be raised.
        For the others keys a warning will be displayed.
        Particular attention should be given to the ``'trial_type'`` key
        which defines the different conditions in the experimental paradigm.

    %(hrf_model)s

    drift_model : {'cosine', 'polynomial', None}, default='cosine'
        Specifies the desired drift model.

    high_pass : :obj:`float`, default=0.01
        High-pass frequency in case of a cosine model (in Hz).

    drift_order : :obj:`int`, default=1
        Order of the drift model (in case it is polynomial).

    fir_delays : array of shape(n_onsets), :obj:`list` or None, default=None
        Will be set to ``[0]`` if ``None`` is passed.
        In case of :term:`FIR` design,
        yields the array of delays used in the :term:`FIR`
        model (in scans).

    add_regs : array of shape(n_frames, n_add_reg) or \
            pandas DataFrame or None, default=None
        additional user-supplied regressors, e.g. data driven noise regressors
        or seed based regressors.

    add_reg_names : :obj:`list` of (n_add_reg,) :obj:`str`, or \
        None, default=None
        If None, while add_regs was provided, these will be termed
        'reg_i', i = 0..n_add_reg - 1
        If add_regs is a DataFrame, the corresponding column names are used
        and add_reg_names is ignored.

    min_onset : :obj:`float`, default=-24
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.

    oversampling : :obj:`int`, default=50
        Oversampling factor used in temporal convolutions.

    Returns
    -------
    design_matrix : DataFrame instance,
        holding the computed design matrix, the index being the frames_times
        and each column a regressor.

    """
    if fir_delays is None:
        fir_delays = [0]
    # check arguments
    # check that additional regressor specification is correct
    n_add_regs = 0
    if add_regs is not None:
        if isinstance(add_regs, pd.DataFrame):
            add_regs_ = add_regs.to_numpy()
            add_reg_names = add_regs.columns.tolist()
        else:
            add_regs_ = np.atleast_2d(add_regs)
        n_add_regs = add_regs_.shape[1]
        assert add_regs_.shape[0] == np.size(frame_times), (
            "Incorrect specification of additional regressors: "
            f"length of regressors provided: {add_regs_.shape[0]}, number of "
            f"time-frames: {np.size(frame_times)}."
        )

    # check that additional regressor names are well specified
    if add_reg_names is None:
        add_reg_names = [f"reg{int(k)}" for k in range(n_add_regs)]
    elif len(add_reg_names) != n_add_regs:
        raise ValueError(
            "Incorrect number of additional regressor names was provided"
            f"({len(add_reg_names)} provided, {n_add_regs} expected."
        )

    # computation of the matrix
    names = []
    matrix = None

    # step 1: events-related regressors
    if events is not None:
        events = check_and_load_tables(events, "events")[0]
        # create the condition-related regressors
        if isinstance(hrf_model, str):
            hrf_model = hrf_model.lower()
        matrix, names = _convolve_regressors(
            events, hrf_model, frame_times, fir_delays, min_onset, oversampling
        )

    # step 2: additional regressors
    if add_regs is not None:
        # add user-supplied regressors and corresponding names
        matrix = (
            np.hstack((matrix, add_regs)) if matrix is not None else add_regs
        )
        names += add_reg_names

    # step 3: drifts
    drift, dnames = _make_drift(
        drift_model, frame_times, drift_order, high_pass
    )

    matrix = np.hstack((matrix, drift)) if matrix is not None else drift

    names += dnames
    # check column names are all unique
    if len(np.unique(names)) != len(names):
        raise ValueError("Design matrix columns do not have unique names")

    # step 4: Force the design matrix to be full rank at working precision
    matrix, _ = full_rank(matrix)

    design_matrix = pd.DataFrame(matrix, columns=names, index=frame_times)
    return design_matrix


def check_design_matrix(design_matrix):
    """Check that the provided DataFrame is indeed a valid design matrix \
    descriptor, and returns a triplet of fields.

    Parameters
    ----------
    design matrix : :obj:`pandas.DataFrame`
        Describes a design matrix.

    Returns
    -------
    frame_times : array of shape (n_frames,),
        Sampling times of the design matrix in seconds.

    matrix : array of shape (n_frames, n_regressors), dtype='f'
        Numerical values for the design matrix.

    names : array of shape (n_events,), dtype='f'
        Per-event onset time (in seconds)

    """
    if len(design_matrix.columns) == 0:
        raise ValueError("The design_matrix dataframe cannot be empty.")
    names = list(design_matrix.keys())
    frame_times = design_matrix.index
    matrix = design_matrix.to_numpy()
    return frame_times, matrix, names


def make_second_level_design_matrix(subjects_label, confounds=None):
    """Set up a second level design.

    Construct a design matrix with an intercept and subject specific confounds.

    Parameters
    ----------
    subjects_label : :obj:`list` of :obj:`str`
        Contain subject labels to extract confounders in the right order,
        corresponding with the images, to create the design matrix.

    confounds : :class:`pandas.DataFrame` or ``None``, default=None
        If given, contains at least two columns, ``subject_label`` and one
        confound. The subjects list determines the rows to extract from
        confounds thanks to its ``subject_label`` column. All subjects must
        have confounds specified. There should be only one row per subject.

    Returns
    -------
    design_matrix : :class:`pandas.DataFrame`
        The second level design matrix.

    """
    confounds_name = []
    if confounds is not None:
        confounds_name = confounds.columns.tolist()
        confounds_name.remove("subject_label")

    design_columns = [*confounds_name, "intercept"]
    # check column names are unique
    if len(np.unique(design_columns)) != len(design_columns):
        raise ValueError("Design matrix columns do not have unique names")

    # float dtype necessary for linalg
    design_matrix = pd.DataFrame(columns=design_columns, dtype="float64")
    for ridx, subject_label in enumerate(subjects_label):
        design_matrix.loc[ridx] = [0.0] * len(design_columns)
        design_matrix.loc[ridx, "intercept"] = 1.0
        if confounds is not None:
            conrow = confounds["subject_label"] == subject_label
            if np.sum(conrow) > 1:
                raise ValueError(
                    "confounds contain more than one row "
                    f"for subject {subject_label}"
                )
            elif np.sum(conrow) == 0:
                raise ValueError(
                    f"confounds not specified for subject {subject_label}"
                )
            for conf_name in confounds_name:
                confounds_value = confounds[conrow][conf_name].to_numpy()[0]
                design_matrix.loc[ridx, conf_name] = confounds_value

    # check design matrix is not singular
    if np.linalg.cond(design_matrix.values) > design_matrix.size:
        warn(
            "Attention: Design matrix is singular. Aberrant estimates "
            "are expected."
        )
    return design_matrix
