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
2. User-specified regressors, that represent information available on
   the data, e.g. motion parameters, physiological data resampled at
   the acquisition rate, or sinusoidal regressors that model the
   signal at a frequency of interest.
3. Drift regressors, that represent low_frequency phenomena of no
   interest in the data; they need to be included to reduce variance
   estimates.

Author: Bertrand Thirion, 2009-2011
"""
import numpy as np
from warnings import warn

from .hemodynamic_models import compute_regressor, _orthogonalize
from .utils  import open4csv


######################################################################
# Ancillary functions
######################################################################


def _poly_drift(order, frametimes):
    """Create a polynomial drift matrix

    Parameters
    ----------
    order, int, number of polynomials in the drift model
    tmax, float maximal time value used in the sequence
          this is used to normalize properly the columns

    Returns
    -------
    pol, array of shape(n_scans, order + 1)
         all the polynomial drift plus a constant regressor
    """
    order = int(order)
    pol = np.zeros((np.size(frametimes), order + 1))
    tmax = float(frametimes.max())
    for k in range(order + 1):
        pol[:, k] = (frametimes / tmax) ** k
    pol = _orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol


def _cosine_drift(period_cut, frametimes):
    """Create a cosine drift matrix with periods greater or equals to period_cut

    Parameters
    ----------
    period_cut: float 
         Cut period of the low-pass filter (in sec)
    frametimes: array of shape(nscans)
         The sampling times (in sec)

    Returns
    -------
    cdrift:  array of shape(n_scans, n_drifts)
             cosin drifts plus a constant regressor at cdrift[:,0]

    Ref: http://en.wikipedia.org/wiki/Discrete_cosine_transform DCT-II
    """
    len_tim = len(frametimes)
    n_times = np.arange(len_tim)
    hfcut = 1. / period_cut  # input parameter is the period
    dt = frametimes[1] - frametimes[0]
    order = int(np.floor(2 * len_tim * hfcut * dt))
    # s.t. hfcut = 1/(2*dt) yields len_tim
    cdrift = np.zeros((len_tim, order))
    nfct = np.sqrt(2.0 / len_tim)

    for k in range(1, order):
        cdrift[:, k - 1] = nfct * np.cos(
            (np.pi / len_tim) * (n_times + .5) * k)

    cdrift[:, order - 1] = 1.  # or 1./sqrt(len_tim) to normalize
    return cdrift


def _blank_drift(frametimes):
    """ Create the blank drift matrix

    Returns
    -------
    np.ones_like(frametimes)
    """
    return np.reshape(np.ones_like(frametimes), (np.size(frametimes), 1))


def _make_drift(drift_model, frametimes, order=1, hfcut=128.):
    """Create the drift matrix

    Parameters
    ----------
    drift_model: string,
                to be chosen among 'polynomial', 'cosine', 'blank'
                that specifies the desired drift model
    frametimes: array of shape(n_scans),
                list of values representing the desired TRs
    order: int, optional,
           order of the drift model (in case it is polynomial)
    hfcut: float, optional,
           frequency cut in case of a cosine model

    Returns
    -------
    drift: array of shape(n_scans, n_drifts), the drift matrix
    names: list of length(ndrifts), the associated names
    """
    drift_model = drift_model.lower()   # for robust comparisons
    if drift_model == 'polynomial':
        drift = _poly_drift(order, frametimes)
    elif drift_model == 'cosine':
        drift = _cosine_drift(hfcut, frametimes)
    elif drift_model == 'blank':
        drift = _blank_drift(frametimes)
    else:
        raise NotImplementedError("Unknown drift model %r" % (drift_model))
    names = []
    for k in range(drift.shape[1] - 1):
        names.append('drift_%d' % (k + 1))
    names.append('constant')
    return drift, names


def _convolve_regressors(paradigm, hrf_model, frametimes, fir_delays=[0],
                         min_onset=-24):
    """ Creation of  a matrix that comprises
    the convolution of the conditions onset with a certain hrf model

    Parameters
    ----------
    paradigm: paradigm instance
    hrf_model: string that can be 'canonical',
               'canonical with derivative' or 'fir'
               that specifies the hemodynamic response function
    frametimes: array of shape(n_scans)
                the targeted timing for the design matrix
    fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays
                    used in the FIR model
    min_onset: float, optional
        minimal onset relative to frametimes[0] (in seconds)
        events that start before frametimes[0] + min_onset are not considered

    Returns
    -------
    rmatrix: array of shape(n_scans, n_regressors),
             contains the convolved regressors
             associated with the experimental condition
    names: list of strings,
           the condition names, that depend on the hrf model used
           if 'canonical' then this is identical to the input names
           if 'canonical with derivative', then two names are produced for
             input name 'name': 'name' and 'name_derivative'
    """
    hnames = []
    rmatrix = None
    if hrf_model == 'fir':
        oversampling = 1
    else:
        oversampling = 16

    for nc in np.unique(paradigm.con_id):
        onsets = paradigm.onset[paradigm.con_id == nc]
        nos = np.size(onsets)
        if paradigm.amplitude is not None:
            values = paradigm.amplitude[paradigm.con_id == nc]
        else:
            values = np.ones(nos)
        if nos < 1:
            continue
        if paradigm.type == 'event':
            duration = np.zeros_like(onsets)
        else:
            duration = paradigm.duration[paradigm.con_id == nc]
        exp_condition = (onsets, duration, values)
        reg, names = compute_regressor(
            exp_condition, hrf_model, frametimes, con_id=nc,
            fir_delays=fir_delays, oversampling=oversampling,
            min_onset=min_onset)
        hnames += names
        if rmatrix is None:
            rmatrix = reg
        else:
            rmatrix = np.hstack((rmatrix, reg))
    return rmatrix, hnames


def _full_rank(X, cmax=1e15):
    """
    This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold.

    Parameters
    ----------
    X: array of shape(nrows, ncols)
    cmax=1.e-15, float tolerance for condition number

    Returns
    -------
    X: array of shape(nrows, ncols) after regularization
    cmax=1.e-15, float tolerance for condition number
    """
    U, s, V = np.linalg.svd(X, 0)
    smax, smin = s.max(), s.min()
    c = smax / smin
    if c < cmax:
        return X, c
    warn('Matrix is singular at working precision, regularizing...')
    lda = (smax - cmax * smin) / (cmax - 1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return X, cmax


######################################################################
# Design matrix
######################################################################


class DesignMatrix():
    """ This is a container for a light-weight class for design matrices
    This class is only used to make IO and visualization

    Class members
    -------------
    matrix: array of shape(n_scans, n_regressors),
            the numerical specification of the matrix
    names: list of len (n_regressors);
           the names associated with the columns
    frametimes: array of shape(n_scans), optional,
                the occurrence time of the matrix rows
    """

    def __init__(self, matrix, names, frametimes=None):
        """
        """
        matrix_ = np.atleast_2d(matrix)
        if matrix_.shape[1] != len(names):
            raise ValueError(
                'The number of names should equate the number of columns')
        if frametimes is not None:
            if frametimes.size != matrix.shape[0]:
                raise ValueError(
                    'The number %d of frametimes is different from the' +
                    'number %d of rows' % (frametimes.size, matrix.shape[0]))

        self.frametimes = frametimes
        self.matrix = matrix_
        self.names = names

    def write_csv(self, path):
        """ write self.matrix as a csv file with appropriate column names

        Parameters
        ----------
        path: string, path of the resulting csv file

        Notes
        -----
        The frametimes are not written
        """
        import csv
        with open4csv(path, "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(self.names)
            writer.writerows(self.matrix)

    def show(self, rescale=True, ax=None):
        """Visualization of a design matrix

        Parameters
        ----------
        rescale: bool, optional
                 rescale columns magnitude for visualization or not
        ax: axis handle, optional
            Handle to axis onto which we will draw design matrix

        Returns
        -------
        ax: axis handle
        """
        import matplotlib.pyplot as plt

        # normalize the values per column for better visualization
        x = self.matrix.copy()
        if rescale:
            x = x / np.sqrt(np.sum(x ** 2, 0))
        if ax is None:
            plt.figure()
            ax = plt.subplot(1, 1, 1)

        ax.imshow(x, interpolation='Nearest', aspect='auto')
        ax.set_label('conditions')
        ax.set_ylabel('scan number')

        if self.names is not None:
            ax.set_xticks(range(len(self.names)))
            ax.set_xticklabels(self.names, rotation=60, ha='right')
        return ax


def make_dmtx(frametimes, paradigm=None, hrf_model='canonical',
              drift_model='cosine', hfcut=128, drift_order=1, fir_delays=[0],
              add_regs=None, add_reg_names=None, min_onset=-24):
    """ Generate a design matrix from the input parameters

    Parameters
    ----------
    frametimes: array of shape(nbframes), the timing of the scans
    paradigm: Paradigm instance, optional
              description of the experimental paradigm
    hrf_model: string, optional,
               that specifies the hemodynamic response function
               it can be 'canonical', 'canonical with derivative' or 'fir'
    drift_model: string, optional
                 specifies the desired drift model,
                 to be chosen among 'polynomial', 'cosine', 'blank'
    hfcut: float, optional
           cut period of the low-pass filter
    drift_order: int, optional
                 order of the drift model (in case it is polynomial)
    fir_delays: array of shape(nb_onsets) or list, optional,
                in case of FIR design, yields the array of delays
                used in the FIR model
    add_regs: array of shape(nbframes, naddreg), optional
              additional user-supplied regressors
    add_reg_names: list of (naddreg) regressor names, optional
                   if None, while naddreg>0, these will be termed
                   'reg_%i',i=0..naddreg-1
    min_onset: float, optional
        minimal onset relative to frametimes[0] (in seconds)
        events that start before frametimes[0] + min_onset are not considered

    Returns
    -------
    DesignMatrix instance
    """
    # check arguments
    # check that additional regressor specification is correct
    n_add_regs = 0
    if add_regs is not None:
        if add_regs.shape[0] == np.size(add_regs):
            add_regs = np.reshape(add_regs, (np.size(add_regs), 1))
        n_add_regs = add_regs.shape[1]
        assert add_regs.shape[0] == np.size(frametimes), ValueError(
            'incorrect specification of additional regressors: '
            'length of regressors provided: %s, number of ' +
            'time-frames: %s' % (add_regs.shape[0], np.size(frametimes)))

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
    matrix = np.zeros((frametimes.size, 0))

    # step 1: paradigm-related regressors
    if paradigm is not None:
        # create the condition-related regressors
        matrix, names = _convolve_regressors(
            paradigm, hrf_model.lower(), frametimes, fir_delays, min_onset)

    # step 2: additional regressors
    if add_regs is not None:
        # add user-supplied regressors and corresponding names
        matrix = np.hstack((matrix, add_regs))
        names += add_reg_names

    # setp 3: drifts
    drift, dnames = _make_drift(drift_model.lower(), frametimes, drift_order,
                                hfcut)
    matrix = np.hstack((matrix, drift))
    names += dnames

    # step 4: Force the design matrix to be full rank at working precision
    matrix, _ = _full_rank(matrix)

    # complete the names with the drift terms
    return DesignMatrix(matrix, names, frametimes)


def dmtx_from_csv(path, frametimes=None):
    """ Return a DesignMatrix instance from  a csv file

    Parameters
    ----------
    path: string, path of the .csv file

    Returns
    -------
    A DesignMatrix instance
    """
    import csv
    with open4csv(path, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        boolfirst = True
        design = []
        for row in reader:
            if boolfirst:
                names = [row[j] for j in range(len(row))]
                boolfirst = False
            else:
                design.append([row[j] for j in range(len(row))])
    x = np.array([[float(t) for t in xr] for xr in design])
    return(DesignMatrix(x, names, frametimes))


def dmtx_light(frametimes, paradigm=None, hrf_model='canonical',
               drift_model='cosine', hfcut=128, drift_order=1, fir_delays=[0],
               add_regs=None, add_reg_names=None, min_onset=-24, path=None):
    """Make a design matrix while avoiding framework

    Parameters
    ----------
    see make_dmtx, plus
    path: string, optional: a path to write the output

    Returns
    -------
    dmtx array of shape(nreg, nbframes):
        the sampled design matrix
    names list of strings of len (nreg)
        the names of the columns of the design matrix
    """
    dmtx_ = make_dmtx(frametimes, paradigm, hrf_model, drift_model, hfcut,
                      drift_order, fir_delays, add_regs, add_reg_names,
                      min_onset)
    if path is not None:
        dmtx_.write_csv(path)
    return dmtx_.matrix, dmtx_.names
