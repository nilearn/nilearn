"""
Implementation of algorithm for sparse multi-subjects learning of Gaussian
graphical models.
"""
# Authors: Philippe Gervais
# License: simplified BSD

import warnings
import collections
import operator
import itertools

import numpy as np
import scipy.linalg

import sklearn.cross_validation
import sklearn.covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.covariance import empirical_covariance
from sklearn.base import BaseEstimator

from sklearn.externals.joblib import Memory, delayed, Parallel

from ._utils import CacheMixin, LogMixin
from ._utils.testing import is_spd


def compute_alpha_max(emp_covs, n_samples):
    """Compute the critical value of the regularization parameter.

    Above this value, the precisions matrices computed by
    group_sparse_covariance are diagonal (complete sparsity)

    This function also returns the value below which the precision
    matrices are fully dense (i.e. minimal number of zero coefficients).

    Parameters
    ----------
    emp_covs : array-like, shape (n_features, n_features, n_subjects)
        covariance matrix for each subject.

    n_samples : array-like, shape (n_subjects,)
        number of samples used in the computation of every covariance matrix.
        n_samples.sum() can be arbitrary.

    Returns
    -------
    alpha_max : float
        minimal value for the regularization parameter that gives a
        full-sparse matrix.

    alpha_min : float
        maximal value for the regularization parameter that gives a fully
        dense matrix.
    """
    A = np.copy(emp_covs)
    n_samples = np.asarray(n_samples).copy()
    n_samples /= n_samples.sum()

    for k in range(emp_covs.shape[-1]):
        # Set diagonal to zero
        A[..., k].flat[::A.shape[0] + 1] = 0
        A[..., k] *= n_samples[k]

    norms = np.sqrt((A ** 2).sum(axis=-1))

    return np.max(norms), np.min(norms[norms > 0])


def _group_sparse_covariance_costs(n_samples, alpha, omega, emp_covs,
                                   verbose=0, debug=False):
    """Compute group sparse covariance costs during computation.

    Returns
    -------
    primal_cost : float
        value of primal cost at current point. This value is minimized by the
        algorithm.

    duality_gap : float
        value of duality gap at current point, with a feasible dual point. This
        value is supposed to always be negative, and vanishing for the optimal
        point.
    """
    # Signs for primal and dual costs are inverted compared to the
    # Honorio & Samaras paper,
    # to match scikit-learn's usage of *minimizing* the primal problem.

    n_features, _, n_subjects = emp_covs.shape

    ## Primal cost
    log_likelihood = 0
    sps = 0  # scalar products
    for k in range(n_subjects):
        t = fast_logdet(omega[..., k])
        sp = (omega[..., k] * emp_covs[..., k]).sum()
        log_likelihood += n_samples[k] * (t - sp)
        sps += n_samples[k] * sp

    # L(1,2)-norm
    l2 = np.sqrt((omega ** 2).sum(axis=-1))
    l12 = l2.sum() - np.diag(l2).sum()  # Do not count diagonal terms
    cost = - (log_likelihood - alpha * l12)

    ## Dual cost: rather heavy computation.
    # Compute A(k)
    A = np.empty(omega.shape, dtype=omega.dtype, order="F")
    for k in range(n_subjects):
        # TODO: can be computed more efficiently using W_inv
        # (see Friedman 2008)
        omega_inv = scipy.linalg.inv(omega[..., k])
        if debug:
            assert is_spd(omega_inv)
        A[..., k] = n_samples[k] * (omega_inv - emp_covs[..., k])
        if debug:
            np.testing.assert_almost_equal(A[..., k], A[..., k].T)

    # Project A on the set of feasible points
    alpha_max = np.sqrt((A ** 2).sum(axis=-1))
    mask = alpha_max > alpha
    for k in range(A.shape[-1]):
        A[mask, k] *= alpha / alpha_max[mask]
        A[..., k].flat[::A.shape[0] + 1] = 0  # essential

    alpha_max = np.sqrt((A ** 2).sum(axis=-1)).max()
    dual_cost = 0
    for k in xrange(n_subjects):
        B = emp_covs[..., k] + A[..., k] / n_samples[k]
        dual_cost += n_samples[k] * (n_features + fast_logdet(B))

    # The previous computation can lead to a non-feasible point, because
    # one of the Bs are not positive definite.
    # Use another value in this case, that ensure positive definiteness of B.
    # The upper bound on the duality gap is not tight in the following, but
    # is smaller than infinity, which is better in any case.
    if not np.isfinite(dual_cost):
        for k in range(n_subjects):
            A[..., k] = - n_samples[k] * emp_covs[..., k]
            A[..., k].flat[::A.shape[0] + 1] = 0
        alpha_max = np.sqrt((A ** 2).sum(axis=-1)).max()
        # the second value (0.05 is arbitrary: positive in ]0,1[)
        gamma = min((alpha / alpha_max, 0.05))
        dual_cost = 0
        for k in range(n_subjects):
            # add gamma on the diagonal
            B = ((1. - gamma) * emp_covs[..., k]
                 + gamma * np.eye(emp_covs.shape[0]))
            dual_cost += n_samples[k] * (n_features + fast_logdet(B))

    gap = cost - dual_cost

    if verbose > 0:
        print("primal cost / duality gap: {cost: .8f} / {gap:.8f}".format(
            gap=gap, cost=cost))

    return (cost, gap)


def _update_submatrix(full, sub, sub_inv, p, h, v):
    """Update submatrix and its inverse.

    sub_inv is the inverse of the submatrix of "full" obtained by removing
    the p-th row and column.

    sub_inv is modified in-place. After execution of this function, it contains
    the inverse of the submatrix of "full" obtained by removing the n+1-th row
    and column.

    This computation is based on the Sherman-Woodbury-Morrison identity.
    """

    n = p - 1
    v[:n + 1] = full[:n + 1, n]
    v[n + 1:] = full[n + 2:, n]
    h[:n + 1] = full[n, :n + 1]
    h[n + 1:] = full[n, n + 2:]

    # change row: first usage of SWM identity
    coln = sub_inv[:, n:n + 1]  # 2d array, useful for sub_inv below
    V = h - sub[n, :]
    coln = coln / (1. + np.dot(V, coln))
    # The following line is equivalent to
    # sub_inv -= np.outer(coln, np.dot(V, sub_inv))
    sub_inv -= np.dot(coln, np.dot(V, sub_inv)[np.newaxis, :])
    sub[n, :] = h

    # change column: second usage of SWM identity
    rown = sub_inv[n:n + 1, :]  # 2d array, useful for sub_inv below
    U = v - sub[:, n]
    rown = rown / (1. + np.dot(rown, U))
    # The following line is equivalent to (but faster)
    # sub_inv -= np.outer(np.dot(sub_inv, U), rown)
    sub_inv -= np.dot(np.dot(sub_inv, U)[:, np.newaxis], rown)
    sub[:, n] = v   # equivalent to sub[n, :] += U

    # Make sub_inv symmetric (overcome some numerical limitations)
    sub_inv += sub_inv.T.copy()
    sub_inv /= 2.


def _assert_submatrix(full, sub, n):
    """Check that "sub" is the matrix obtained by removing the p-th col and row
    in "full". Used only for debugging.
    """
    true_sub = np.empty_like(sub)
    true_sub[:n, :n] = full[:n, :n]
    true_sub[n:, n:] = full[n + 1:, n + 1:]
    true_sub[:n, n:] = full[:n, n + 1:]
    true_sub[n:, :n] = full[n + 1:, :n]

    np.testing.assert_almost_equal(true_sub, sub)


def group_sparse_covariance(subjects, alpha, max_iter=50, tol=1e-3,
                            assume_centered=False, verbose=0,
                            probe_function=None, precisions_init=None,
                            debug=False):
    """Compute sparse precision matrices and covariance matrices.

    The precision matrices returned by this function are sparse, and share a
    common sparsity pattern: all have zeros at the same location. This is
    achieved by simultaneous computation of all precision matrices at the
    same time.

    Running time is linear on max_iter, and number of subjects (len(subjects)),
    but cubic on number of features (subjects[0].shape[1]).

    Parameters
    ==========
    subjects : list of numpy.ndarray
        input subjects. Each subject is a 2D array, whose columns contain
        signals. Each array shape must be (sample number, feature number).
        The sample number can vary from subject to subject, but all subjects
        must have the same number of features (i.e. of columns).

    alpha : float
        regularization parameter. With normalized covariances matrices and
        number of samples, sensible values lie in the [0, 1] range(zero is
        no regularization: output is not sparse)

    max_iter : int, optional
        maximum number of iterations.

    tol : positive float or None, optional
        The tolerance to declare convergence: if the duality gap goes below
        this value, optimization is stopped. If None, no check is performed.

    verbose : int, optional
        verbosity level. Zero means "no message".

    probe_function : callable or None
        This value is called before the first iteration and after each
        iteration. If it returns True, then optimization is stopped
        prematurely.
        The function is given as arguments (in that order):

        - empirical covariances (ndarray),
        - number of samples for each subject (ndarray),
        - regularization parameter (float)
        - maximum iteration number (integer)
        - tolerance (float)
        - current iteration number (integer). -1 means "before first iteration"
        - current value of precisions (ndarray).
        - previous value of precisions (ndarray). None before first iteration.

    precisions_init: numpy.ndarray
        initial value of the precision matrices. If not provided, a diagonal
        matrix with the variances of each input signal is used.

    debug : bool, optional
        if True, perform checks during computation. It can help find
        numerical problems, but increases computation time a lot.

    Returns
    =======
    emp_covs : numpy.ndarray, shape (n_features, n_features, n_subjects)
        empirical covariances matrices

    precisions : numpy.ndarray, shape (n_features, n_features, n_subjects)
        estimated precision matrices

    Notes
    =====
    The present algorithm is based on:

    Jean Honorio and Dimitris Samaras.
    "Simultaneous and Group-Sparse Multi-Task Learning of Gaussian Graphical
    Models". arXiv:1207.4255 (17 July 2012). http://arxiv.org/abs/1207.4255.
    """

    emp_covs, n_samples = empirical_covariances(
        subjects, assume_centered=assume_centered)

    precisions = _group_sparse_covariance(
        emp_covs, n_samples, alpha, max_iter=max_iter, tol=tol,
        assume_centered=assume_centered, verbose=verbose,
        precisions_init=precisions_init, probe_function=probe_function,
        debug=debug)

    return emp_covs, precisions


def _group_sparse_covariance(emp_covs, n_samples, alpha, max_iter=10, tol=1e-3,
                             assume_centered=False, precisions_init=None,
                             probe_function=None, verbose=0, debug=False):
    """Internal version of group_sparse_covariance.
    See its docstring for details.
    """
    if tol == -1:
        tol = None
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("Regularization parameter alpha must be a "
                         "positive number.\n"
                         "You provided: {0}".format(str(alpha)))
    n_subjects = emp_covs.shape[-1]
    n_features = emp_covs[0].shape[0]
    n_samples = np.asarray(n_samples)
    n_samples /= n_samples.sum()  # essential for numerical stability

    if precisions_init is None:
        omega = np.ndarray(shape=emp_covs.shape, dtype=emp_covs.dtype,
                           order="F")
        for k in range(n_subjects):
            # Values on main diagonals are far from zero, because they
            # are timeseries energy.
            omega[..., k] = np.diag(1. / np.diag(emp_covs[..., k]))
    else:
        omega = precisions_init.copy()

    # Preallocate arrays
    y = np.ndarray(shape=(n_subjects, n_features - 1), dtype=emp_covs.dtype)
    u = np.ndarray(shape=(n_subjects, n_features - 1), dtype=emp_covs.dtype)
    y_1 = np.ndarray(shape=(n_subjects, n_features - 2), dtype=emp_covs.dtype)
    h_12 = np.ndarray(shape=(n_subjects, n_features - 2), dtype=emp_covs.dtype)
    q = np.ndarray(shape=(n_subjects,), dtype=emp_covs.dtype)
    aq = np.ndarray(shape=(n_subjects,), dtype=emp_covs.dtype)  # temp. array
    c = np.ndarray(shape=(n_subjects,), dtype=emp_covs.dtype)
    W = np.ndarray(shape=(omega.shape[0] - 1, omega.shape[1] - 1,
                          omega.shape[2]),
                   dtype=emp_covs.dtype, order="F")
    W_inv = np.ndarray(shape=W.shape, dtype=emp_covs.dtype, order="F")

    # Auxilliary arrays.
    v = np.ndarray((omega.shape[0] - 1,), dtype=omega.dtype)
    h = np.ndarray((omega.shape[1] - 1,), dtype=omega.dtype)

    # Optional.
    tolerance_reached = False
    max_norm = None

    omega_old = np.empty(omega.shape, dtype=omega.dtype)
    if probe_function is not None:
        # iteration number -1 means called before iteration loop.
        probe_function(emp_covs, n_samples, alpha, max_iter, tol,
                       -1, omega, None)
    probe_interrupted = False

    # Start optimization loop. Variables are named following (mostly) the
    # Honorio-Samaras paper notations.

    # Used in the innermost loop. Computed here to save some computation.
    alpha2 = alpha ** 2

    for n in xrange(max_iter):
        if verbose >= 1:
            if max_norm is not None:
                suffix = (" variation (max norm): {max_norm:.3e} ".format(
                    max_norm=max_norm))
            else:
                suffix = ""
            print("* iteration {iter_n:d} ({percentage:.0f} %){suffix} ..."
                  "".format(iter_n=n, percentage=100. * n / max_iter,
                            suffix=suffix))

        omega_old[...] = omega
        for p in xrange(n_features):

            if p == 0:
                # Initial state: remove first col/row
                W = omega[1:, 1:, :].copy()   # stack of W(k)
                W_inv = np.ndarray(shape=W.shape, dtype=np.float)
                for k in xrange(W.shape[2]):
                    # stack of W^-1(k)
                    W_inv[..., k] = scipy.linalg.inv(W[..., k])
                    if debug:
                        np.testing.assert_almost_equal(
                            np.dot(W_inv[..., k], W[..., k]),
                            np.eye(W_inv[..., k].shape[0]), decimal=10)
                        _assert_submatrix(omega[..., k], W[..., k], p)
                        assert(is_spd(W_inv[..., k]))
            else:
                # Update W and W_inv
                if debug:
                    omega_orig = omega.copy()

                for k in range(n_subjects):
                    _update_submatrix(omega[..., k],
                                      W[..., k], W_inv[..., k], p, h, v)

                    if debug:
                        _assert_submatrix(omega[..., k], W[..., k], p)
                        assert(is_spd(W_inv[..., k], decimal=14))
                        np.testing.assert_almost_equal(
                            np.dot(W[..., k], W_inv[..., k]),
                            np.eye(W_inv[..., k].shape[0]), decimal=10)
                if debug:
                    # Check that omega has not been modified.
                    np.testing.assert_almost_equal(omega_orig, omega)

            # In the following lines, implicit loop on k (subjects)
            # Extract y and u
            y[:, :p] = omega[:p, p, :].T
            y[:, p:] = omega[p + 1:, p, :].T

            u[:, :p] = emp_covs[:p, p, :].T
            u[:, p:] = emp_covs[p + 1:, p, :].T

            for m in xrange(n_features - 1):
                # Coordinate descent on y

                # T(k) -> n_samples[k]
                # v(k) -> emp_covs[p, p, k]
                # h_22(k) -> W_inv[m, m, k]
                # h_12(k) -> W_inv[:m, m, k],  W_inv[m+1:, m, k]
                # y_1(k) -> y[k, :m], y[k, m+1:]
                # u_2(k) -> u[k, m]
                h_12[:, :m] = W_inv[:m, m, :].T
                h_12[:, m:] = W_inv[m + 1:, m, :].T
                y_1[:, :m] = y[:, :m]
                y_1[:, m:] = y[:, m + 1:]

                c[:] = - n_samples * (
                    emp_covs[p, p, :] * (h_12 * y_1).sum(axis=1) + u[:, m]
                    )
                c2 = np.sqrt(np.dot(c, c))

                # x -> y[:][m]
                if c2 <= alpha:
                    y[:, m] = 0  # x* = 0
                else:
                    # q(k) -> T(k) * v(k) * h_22(k)
                    # \lambda -> gamma   (lambda is a Python keyword)
                    q[:] = n_samples * emp_covs[p, p, :] * W_inv[m, m, :]
                    if debug:
                        assert(np.all(q > 0))
                    # x* = \lambda* diag(1 + \lambda q)^{-1} c

                    # Newton-Raphson loop. Loosely based on Scipy's.
                    # Tolerance does not seem to be important for numerical
                    # stability (tolerance of 1e-2 works) but has an effect on
                    # overall convergence rate (the tighter the better.)

                    gamma = 0.  # initial value
                    # Precompute some quantities
                    cc = c * c
                    two_ccq = 2. * cc * q
                    for _ in itertools.repeat(None, 100):
                        # Function whose zero must be determined (fval) and
                        # its derivative (fder).
                        # Written inplace to save some function calls.
                        aq = 1. + gamma * q
                        aq2 = aq * aq
                        fder = (two_ccq / (aq2 * aq)).sum()

                        if fder == 0:
                            msg = "derivative was zero."
                            warnings.warn(msg, RuntimeWarning)
                            break
                        fval = - (alpha2 - (cc / aq2).sum()) / fder
                        gamma = fval + gamma
                        if abs(fval) < 1.5e-8:
                            break

                    if abs(fval) > 0.1:
                        warnings.warn("Newton-Raphson step did not converge.\n"
                                      "This may indicate a badly conditioned "
                                      "system.")

                    if debug:
                        assert gamma >= 0., gamma
                    y[:, m] = (gamma * c) / aq  # x*

            # Copy back y in omega (column and row)
            omega[:p, p, :] = y[:, :p].T
            omega[p + 1:, p, :] = y[:, p:].T
            omega[p, :p, :] = y[:, :p].T
            omega[p, p + 1:, :] = y[:, p:].T

            for k in xrange(n_subjects):
                omega[p, p, k] = 1. / emp_covs[p, p, k] + np.dot(
                    np.dot(y[k, :], W_inv[..., k]), y[k, :])

                if debug:
                    assert(is_spd(omega[..., k]))

        if probe_function is not None:
            if probe_function(emp_covs, n_samples, alpha, max_iter, tol,
                              n, omega, omega_old) is True:
                probe_interrupted = True
                print("probe_function interrupted loop")
                break

        # Compute max of variation
        omega_old -= omega
        omega_old = abs(omega_old)
        max_norm = omega_old.max()

        if tol is not None and max_norm < tol:
            if verbose >= 1:
                print("tolerance reached at iteration number {0:d}: {1:.3e}"
                      "".format(n + 1, max_norm))
            tolerance_reached = True
            break

    if tol is not None and not tolerance_reached and not probe_interrupted:
        warnings.warn("Maximum number of iterations reached without getting "
                      "to the requested tolerance level.")

    return omega


class GroupSparseCovariance(BaseEstimator, CacheMixin, LogMixin):
    """Covariance and precision matrix estimator.

    The algorithm used is based on what is described in:

    Jean Honorio and Dimitris Samaras.
    "Simultaneous and Group-Sparse Multi-Task Learning of Gaussian Graphical
    Models". arXiv:1207.4255 (17 July 2012). http://arxiv.org/abs/1207.4255.

    Parameters
    ----------
    alpha : float
        regularization parameter. With normalized covariances matrices and
        number of samples, sensible values lie in the [0, 1] range(zero is
        no regularization: output is not sparse)

    tol : positive float, optional
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped

    max_iter : int, optional
        maximum number of iterations. The default value (10) is rather
        conservative.

    verbose : int, optional
        verbosity level. Zero means "no message".

    assume_centered : bool
        if True, assume that all signals passed to fit() are centered.

    memory : instance of joblib.Memory or string, optional
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : int, optional
        Caching aggressiveness. Higher values mean more caching.

    Attributes
    ----------
    `covariances_` : numpy.ndarray, shape (n_features, n_features, n_subjects)
        maximum likelihood covariance estimations.

    `precisions_` : numpy.ndarraye, shape (n_features, n_features, n_subjects)
        precisions matrices estimated using Antonio & Samaras algorithm.
    """

    def __init__(self, alpha=0.1, tol=1e-3, max_iter=10, verbose=1,
                 assume_centered=False,
                 memory=Memory(cachedir=None), memory_level=0):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.assume_centered = assume_centered

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, subjects, y=None):
        """Fits the group sparse precision model according to the given
        training data and parameters.

        Parameters
        ----------
        subjects : list of numpy.ndarray, shape for each (n_samples, n_features)
            input subjects. Each subject is a 2D array, whose columns contain
            signals. Sample number can vary from subject to subject, but all
            subjects must have the same number of features (i.e. of columns).

        Attributes
        ----------
        `covariances_` : numpy.ndarray
            empirical covariances

        `precisions_` : numpy.ndarray
            precision matrices

        Returns
        -------
        self : GroupSparseCovariance instance
            the object itself. Useful for chaining operations.
        """

        self.log("Computing covariance matrices")
        self.covariances_, n_samples = empirical_covariances(
                subjects, assume_centered=self.assume_centered)

        self.log("Computing precision matrices")
        ret = self._cache(
            _group_sparse_covariance, memory_level=1)(
                self.covariances_, n_samples, self.alpha,
                tol=self.tol, max_iter=self.max_iter,
                assume_centered=self.assume_centered,
                verbose=self.verbose - 1, debug=False)

        self.precisions_ = ret
        return self


def empirical_covariances(subjects, assume_centered=False, dtype=np.float64):
    """Compute empirical covariances for several signals.

    Parameters
    ----------
    subjects : list of numpy.ndarray, shape for each (n_samples, n_features)
        input subjects. Each subject is a 2D array, whose columns contain
        signals. Sample number can vary from subject to subject, but all
        subjects must have the same number of features (i.e. of columns).

    assume_centered : bool, optional
        if True, assume that all input signals are centered. This slightly
        decreases computation time by avoiding useless computation.

    dtype : numpy dtype
        dtype of output array. Default: numpy.float64

    Returns
    -------
    emp_covs : numpy.ndarray
        empirical covariances.
        shape : (feature number, feature number, subject number)

    n_samples : numpy.ndarray
        number of samples for each subject. shape: (subject number,)
    """
    if not hasattr(subjects, "__iter__"):
        raise ValueError("'subjects' input argument must be an iterable. "
                         "You provided {0}".format(subjects.__class__))

    n_subjects = [s.shape[1] for s in subjects]
    if len(set(n_subjects)) > 1:
        raise ValueError("All subjects must have the same number of "
                         "features.\nYou provided: {0}".format(str(n_subjects))
                         )
    n_subjects = len(subjects)
    n_features = subjects[0].shape[1]

    # Enable to change dtype here because depending on user conversion from
    # single precision to double will be required or not.
    emp_covs = np.empty((n_features, n_features, n_subjects),
                        dtype=dtype, order="F")
    for k, s in enumerate(subjects):
        M = empirical_covariance(s, assume_centered=assume_centered)

        emp_covs[..., k] = M + M.T
    emp_covs /= 2

    n_samples = np.asarray([s.shape[0] for s in subjects], dtype=np.float64)

    return emp_covs, n_samples


def group_sparse_scores(precisions, n_samples, emp_covs, alpha):
    """Compute scores used by group_sparse_covariance.

    The log-likelihood of a given list of empirical covariances /
    precisions.

    Parameters
    ----------
    precisions : numpy.ndarray, shape (n_features, n_features, n_subjects)
        estimated precisions.

    n_samples : array-like, shape: (n_subjects,)
        number of samples used in estimating each subject in "precisions".
        n_samples.sum() must be equal to 1.

    Returns
    -------
    log_lik : float
        log-likelihood of precisions on the given covariances. This is the
        opposite of the loss function, without the regularization term.

    objective : float
        value of objective function. This is the value minimized by
        group_sparse_covariance().
    """
    log_lik = 0
    for k in range(precisions.shape[2]):
        log_lik += n_samples[k] * sklearn.covariance.log_likelihood(
            emp_covs[..., k], precisions[..., k])

    l2 = np.sqrt((precisions ** 2).sum(axis=-1))
    l12 = l2.sum() - np.diag(l2).sum()  # Do not count diagonal terms

    return (log_lik, alpha * l12 - log_lik)


def group_sparse_covariance_path(train_subjs, alphas, test_subjs=None,
                                 tol=1e-3, max_iter=10, assume_centered=False,
                                 precisions_init=None, verbose=0, debug=False,
                                 probe_function=None):
    """Get estimated precision matrices for different values of alpha.

    Calling this function is faster than calling group_sparse_covariance()
    repeatedly, because it makes use of the first result to initialize the
    next computation.

    Parameters
    ----------
    train_subjs : list of numpy.ndarray
        list of signals.

    alphas : list of float
         values of alpha to use. Best results for sorted values (decreasing)

    test_subjs : list of numpy.ndarray
        list of signals, independent from those in train_subjs, on which to
        compute a score. If None, no score is computed.

    verbose : int
        verbosity level

    tol, max_iter, assume_centered, debug, precisions_init :
        Passed to group_sparse_covariance(). See the corresponding docstring
        for details.

    probe_function : callable
        This value is called before the first iteration and after each
        iteration. If it returns True, then optimization is stopped
        prematurely.
        The function is given as arguments (in that order):

        - empirical covariances (ndarray),
        - number of samples for each subject (ndarray),
        - regularization parameter (float)
        - maximum iteration number (integer)
        - tolerance (float)
        - current iteration number (integer). -1 means "before first iteration"
        - current value of precisions (ndarray).
        - previous value of precisions (ndarray). None before first iteration.

    Returns
    -------
    precisions_list : list of numpy.ndarray
        estimated precisions for each value of alpha provided. The length of
        this list is the same as that of parameter "alphas".

    scores : list of float
        for each estimated precision, score obtained on the test set. Output
        only if test_subjs is not None.
    """
    train_covs, train_n_samples = empirical_covariances(
        train_subjs, assume_centered=assume_centered)
    test_covs, _ = empirical_covariances(
        test_subjs, assume_centered=assume_centered)

    scores = []
    precisions_list = []
    for alpha in alphas:
        precisions = _group_sparse_covariance(
            train_covs, train_n_samples, alpha, tol=tol, max_iter=max_iter,
            assume_centered=assume_centered, precisions_init=precisions_init,
            verbose=verbose, debug=debug, probe_function=probe_function)

        # Compute log-likelihood
        if test_subjs is not None:
            scores.append(group_sparse_scores(precisions, train_n_samples,
                                              test_covs, 0)[0])
        precisions_list.append(precisions)
        precisions_init = precisions

    if test_subjs is not None:
        return precisions_list, scores
    else:
        return precisions_list


class EarlyStopProbe(object):
    """Callable probe for early stopping in GroupSparseCovarianceCV.

    Stop optimizing as soon as the score on the test set starts decreasing.
    An instance of this class is supposed to be passed in the probe_function
    argument of group_sparse_covariance().
    """
    def __init__(self, test_subjs):
        self.test_emp_covs, _ = empirical_covariances(test_subjs)

    def __call__(self, emp_covs, n_samples, alpha, max_iter, tol,
                 iter_n, omega, prev_omega):
        score = group_sparse_scores(
            omega, n_samples, self.test_emp_covs, alpha)[0]
        if iter_n > -1 and self.last_score > score:
            print("test score is decreasing. Stopping at iteration %d"
                  % iter_n)
            return True
        self.last_score = score


class GroupSparseCovarianceCV(BaseEstimator):
    # See also GraphLasso in scikit-learn.
    """
    Parameters
    ----------
    alphas : integer
        initial number of points in the grid of regularization parameter
        values. Each step of grid refinement adds that many points as well.

    n_refinements : integer
        number of times the initial grid should be refined.

    cv : integer
        number of folds in a K-fold cross-validation scheme.

    tol : float
        tolerance used for every optimization.

    max_iter : integer
        maximum number of iterations for each optimization.

    assume_centered : bool
        if True, assume that every signal passed to fit() has zero mean. This
        can avoid useless computation.

    verbose : integer
        verbosity level. 0 means nothing is printed to the user.

    memory : joblib.Memory instance.
        joblib object used for caching.

    memory_level : integer
        caching aggressiveness. The larger, the more things are cached. Zero
        means no caching.

    n_jobs : integer
        maximum number of cpu cores to use. The number of cores actually used
        at the same time cannot exceed the number of folds in folding strategy
        (that is, the value of cv).

    debug : bool
        if True, activates some internal checks for consistency. Only useful
        for nilearn developers, not users.

    early_stopping : bool
        if True, reduce computation time by using a heuristic to reduce the
        number of iterations required to get the optimal value for alpha. Be
        aware that this can lead to slightly different values for the optimal
        alpha compared to early_stopping=False.
    """
    def __init__(self, alphas=4, n_refinements=4, cv=None,
                 tol=1e-3, max_iter=50, assume_centered=False, verbose=1,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, debug=False, early_stopping=True):
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.assume_centered = assume_centered

        self.verbose = verbose
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.debug = debug
        self.early_stopping = early_stopping

    def fit(self, subjects, y=None):
        """Compute cross-validated group-sparse precisions.

        Parameters
        ----------
        subjects : list of numpy.ndarray, shape for each (n_samples, n_features)
            input subjects. Each subject is a 2D array, whose columns contain
            signals. Sample number can vary from subject to subject, but all
            subjects must have the same number of features (i.e. of columns.)

        Attributes
        ----------
        covariances_ : numpy.ndarray, shape (n_features, n_features, n_subjects)
            covariance matrices, one per subject.

        precisions_ : numpy.ndarray, shape (n_features, n_features, n_subjects)
            precision matrices, one per subject. All matrices have the same
            sparsity pattern (if a coefficient is zero for a given matrix, it
            is also zero for every other.)

        alpha_ : float
            selected value for penalization parameter.

        cv_alphas_ : list of float
            all values explored for the penalization parameter.

        cv_scores_ : numpy.ndarray with shape (n_alphas, n_folds)
            scores obtained on test set for each value of the penalization
            parameter explored.

        Returns
        =======
        self: GroupSparseCovarianceCV
            the object instance itself.
        """
        # Empirical covariances
        emp_covs, n_samples = \
                  empirical_covariances(subjects,
                                        assume_centered=self.assume_centered)
        n_subjects = emp_covs.shape[2]

        # One cv generator per subject must be created, because each subject
        # can have a different number of samples from the others.
        cv = []
        for k in range(n_subjects):
            cv.append(sklearn.cross_validation.check_cv(
                self.cv, subjects[k], None, classifier=False))

        path = list()  # List of (alpha, scores, covs)
        n_alphas = self.alphas

        if isinstance(n_alphas, collections.Sequence):
            alphas = list(self.alphas)
            n_alphas = len(alphas)
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            alpha_1, _ = compute_alpha_max(emp_covs, n_samples)
            alpha_0 = 1e-2 * alpha_1
            alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1),
                               n_alphas)[::-1]

        covs_init = itertools.repeat(None)
        for i in range(n_refinements):
            # Compute the cross-validated loss on the current grid
            train_test_subjs = []
            for train_test in zip(*cv):
                assert(len(train_test) == n_subjects)
                train_test_subjs.append(zip(*[(subject[train, :],
                                               subject[test, :])
                                            for subject, (train, test)
                                            in zip(subjects, train_test)]))
            if self.early_stopping:
                probes = [EarlyStopProbe(test_subjs)
                          for _, test_subjs in train_test_subjs]
            else:
                probes = itertools.repeat(None)

            this_path = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(group_sparse_covariance_path)(
                    train_subjs, alphas, test_subjs=test_subjs,
                    max_iter=self.max_iter, tol=self.tol,
                    assume_centered=self.assume_centered,
                    verbose=self.verbose, debug=self.debug,
                    # Warm restart is only useful without early stopping.
                    precisions_init=None if self.early_stopping else prec_init,
                    probe_function=probe)
                for (train_subjs, test_subjs), prec_init, probe
                in zip(train_test_subjs, covs_init, probes))

            # this_path[i] is a tuple (precisions_list, scores)
            # - scores: scores obtained with the i-th folding, for each value
            #   of alpha.
            # - precisions_list: corresponding precisions matrices, for each
            #   value of alpha.
            precisions_list, scores = zip(*this_path)
            # now scores[i][j] is the score for the i-th folding, j-th value of
            # alpha (analoguous for precisions_list)
            precisions_list = zip(*precisions_list)
            scores = [np.mean(sc) for sc in zip(*scores)]
            # scores[i] is the mean score obtained for the i-th value of alpha.

            path.extend(zip(alphas, scores, precisions_list))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum score (avoid using the built-in 'max' function
            # to have a fully-reproducible selection of the smallest alpha in
            # case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (alpha, this_score, _) in enumerate(path):
                if this_score >= .1 / np.finfo(np.float).eps:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # Refine the grid
            if best_index == 0:
                # We do not need to go back: we have chosen
                # the highest value of alpha for which there are
                # non-zero coefficients
                alpha_1 = path[0][0]
                alpha_0 = path[1][0]
                covs_init = path[0][2]
            elif (best_index == last_finite_idx
                    and not best_index == len(path) - 1):
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                alpha_1 = path[best_index][0]
                alpha_0 = path[best_index + 1][0]
                covs_init = path[best_index][2]
            elif best_index == len(path) - 1:
                alpha_1 = path[best_index][0]
                alpha_0 = 0.01 * path[best_index][0]
                covs_init = path[best_index][2]
            else:
                alpha_1 = path[best_index - 1][0]
                alpha_0 = path[best_index + 1][0]
                covs_init = path[best_index - 1][2]
            alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0),
                                 len(alphas) + 2)
            alphas = alphas[1:-1]
            if self.verbose and n_refinements > 1:
                print("[GroupSparseCovarianceCV] Done refinement "
                      "% 2i out of %i" % (i + 1, n_refinements))

        path = list(zip(*path))
        cv_scores_ = list(path[1])
        alphas = list(path[0])

        self.cv_scores_ = np.array(cv_scores_)
        self.alpha_ = alphas[best_index]
        self.cv_alphas_ = alphas

        # Finally fit the model with the selected alpha
        self.covariances_ = emp_covs
        self.precisions_ = _group_sparse_covariance(
            emp_covs, n_samples, self.alpha_, tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose - 1, debug=self.debug)
        return self
