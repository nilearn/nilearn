"""
Implementation of algorithm for sparse multi-task learning of gaussian
graphical models.
"""
# Authors: Philippe Gervais
# License: simplified BSD

import warnings
import collections
import operator
import itertools
import time

import numpy as np

import sklearn.cross_validation
import sklearn.covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.covariance import empirical_covariance
from sklearn.base import BaseEstimator

from sklearn.externals.joblib import Memory, delayed, Parallel

from ._utils import CacheMixin, LogMixin
from .testing import is_spd


def rho_max(emp_covs, n_samples):
    """
    Parameters
    ----------
    emp_covs: numpy.ndarray
        covariance matrix for each task.
        shape (variable number, variable number, covariance matrix number)

    n_samples: array-like
        number of samples used in the computation of every covariance matrix.

    Returns
    -------
    rho_max: minimal value for regularization parameter that gives a
        full-sparse matrix.

    rho_min: maximal value for rho that gives a fully dense matrix
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


def _group_sparse_covariance_costs(n_tasks, n_var, n_samples, rho, omega,
                                   emp_covs, display=False, debug=False):
    """Compute group sparse covariance costs during computation.

    Returns
    -------
    primal_cost: float
        value of primal cost at current point. This value is minimized by the
        algorithm.

    duality_gap: float
        value of duality gap at current point, with a feasible dual point. This
        value is supposed to always be negative, and vanishing for the optimal
        point.
    """
    # Signs for primal and dual costs are inverted compared to the H&S paper,
    # to match scikit-learn's usage of *minimizing* the primal problem.

    ## Primal cost
    ll = 0  # log-likelihood
    sps = 0  # scalar products
    for k in xrange(n_tasks):
        t = fast_logdet(omega[..., k])
        sp = (omega[..., k] * emp_covs[..., k]).sum()
        ll += n_samples[k] * (t - sp)
        sps += n_samples[k] * sp

    # L(1,2)-norm
    l2 = np.sqrt((omega ** 2).sum(axis=-1))
    l12 = l2.sum() - np.diag(l2).sum()  # Do not count diagonal terms
    cost = - (ll - rho * l12)

    ## Dual cost: rather heavy computation.
    # Compute A(k)
    A = np.empty(omega.shape, dtype=omega.dtype, order="F")
    for k in xrange(n_tasks):
        # TODO: can be computed more efficiently using Winv (see Friedman 2008)
        omega_inv = np.linalg.inv(omega[..., k])
        if debug:
            assert is_spd(omega_inv)
        A[..., k] = n_samples[k] * (omega_inv - emp_covs[..., k])
        if debug:
            np.testing.assert_almost_equal(A[..., k], A[..., k].T)

    # Project A on the set of feasible points
    rho_max = np.sqrt((A ** 2).sum(axis=-1))
    mask = rho_max > rho
    for k in range(A.shape[-1]):
        A[mask, k] *= rho / rho_max[mask]
        A[..., k].flat[::A.shape[0] + 1] = 0  # essential

    rho_max = np.sqrt((A ** 2).sum(axis=-1)).max()
    dual_cost = 0
    for k in xrange(n_tasks):
        B = emp_covs[..., k] + A[..., k] / n_samples[k]
        dual_cost += n_samples[k] * (n_var + fast_logdet(B))

    # The previous computation can lead to a non-feasible point, because
    # one of the Bs are not positive definite.
    # Use another value in this case, that ensure positive definiteness of B.
    # The upper bound on the duality gap is not tight in the following, but
    # is smaller than infinity, which is better in any case.
    if not np.isfinite(dual_cost):
        for k in range(n_tasks):
            A[..., k] = - n_samples[k] * emp_covs[..., k]
            A[..., k].flat[::A.shape[0] + 1] = 0
        rho_max = np.sqrt((A ** 2).sum(axis=-1)).max()
        # the second value (0.05 is arbitrary: positive in ]0,1[)
        alpha = min((rho / rho_max, 0.05))
        dual_cost = 0
        for k in range(n_tasks):
            # add alpha on the diagonal
            B = ((1. - alpha) * emp_covs[..., k]
                 + alpha * np.eye(emp_covs.shape[0]))
            dual_cost += n_samples[k] * (n_var + fast_logdet(B))

    gap = cost - dual_cost

    other = (omega.copy(), time.time())

    if display:
        print("primal cost / duality gap: {cost: .8f} / {gap:.8f}".format(
            gap=gap, cost=cost))

    return (cost, gap, other)


# The signatures of quad_trust_region and quad_trust_region_deriv are
# complicated, but this allows for some interesting optimizations.

# inplace operation, merge with quad_trust_region_deriv
def quad_trust_region(alpha, q, two_ccq, cc, rho2):
    """This value is optimized to zero by the Newton-Raphson step."""
    return rho2 - (cc / ((1. + alpha * q) ** 2)).sum()


def quad_trust_region_deriv(alpha, q, two_ccq, cc, rho2):
    """Derivative of quad_trust_region."""
    return (two_ccq / (1. + alpha * q) ** 3).sum()


def update_submatrix(full, sub, sub_inv, p, h, v):
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

    # change row
    coln = sub_inv[:, n]
    V = h - sub[n, :]
    coln = coln / (1. + np.dot(V, coln))
    sub_inv -= np.outer(coln, np.dot(V, sub_inv))
    sub[n, :] = h

    # change column
    rown = sub_inv[n, :]
    U = v - sub[:, n]
    rown = rown / (1. + np.dot(rown, U))
    sub_inv -= np.outer(np.dot(sub_inv, U), rown)
    sub[:, n] = v   # equivalent to sub[n, :] += U


def assert_submatrix(full, sub, n):
    """Check that "sub" is the matrix obtained by removing the p-th col and row
    in "full". Used only for debugging.
    """
    true_sub = np.ndarray(shape=sub.shape, dtype=sub.dtype)
    true_sub[:n, :n] = full[:n, :n]
    true_sub[n:, n:] = full[n + 1:, n + 1:]
    true_sub[:n, n:] = full[:n, n + 1:]
    true_sub[n:, :n] = full[n + 1:, :n]

    np.testing.assert_almost_equal(true_sub, sub)


def group_sparse_covariance(tasks, rho, max_iter=50, tol=1e-3,
                            assume_centered=False, verbose=0,
                            probe_function=None, precisions_init=None,
                            debug=False):
    """Compute sparse precision matrices and covariance matrices.

    The precision matrices returned by this function are sparse, and share a
    common sparsity pattern: all have zeros at the same location. This is
    achieved by simultaneous computation of all precision matrices at the
    same time.

    Running time is linear on max_iter, and number of tasks (len(tasks)), but
    cubic on number of signals (tasks[0].shape[1]).

    Parameters
    ==========
    tasks: list of numpy.ndarray
        list of signals to process. Each element of the list must be a
        2D array, with shape (sample number, feature number). The sample number
        can vary from task to task, but not the feature number.

    rho: float
        regularization parameter. With normalized covariances matrices and
        number of samples, sensible values lie in the [0, 1] range(zero is
        no regularization: output is not sparse)

    tol: positive float or None, optional
        The tolerance to declare convergence: if the maximum change in
        estimated precision matrices goes below this value, optimization is
        stopped. If None, no check is performed.

    max_iter: int, optional
        maximum number of iterations.

    verbose: int, optional
        verbosity level. Zero means "no message".

    debug: bool, optional
        if True, perform checks during computation. It can help find
        numerical problems, but increases computation time a lot.

    Returns
    =======
    emp_covs: numpy.ndarray
        empirical covariances matrices, as a 3D array (last index is task)

    precisions: numpy.ndarray
        estimated precision matrices, as a 3D array (last index is task)

    Notes
    =====
    The present algorithm is based on:

    Jean Honorio and Dimitris Samaras.
    "Simultaneous and Group-Sparse Multi-Task Learning of Gaussian Graphical
    Models". arXiv:1207.4255 (17 July 2012). http://arxiv.org/abs/1207.4255.
    """

    emp_covs, n_samples, _, _ = empirical_covariances(
        tasks, assume_centered=assume_centered, debug=debug)

    precisions = _group_sparse_covariance(
        emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
        assume_centered=assume_centered, verbose=verbose,
        precisions_init=precisions_init, probe_function=probe_function,
        debug=debug)

    return emp_covs, precisions


def _group_sparse_covariance(emp_covs, n_samples, rho, max_iter=10, tol=1e-3,
                             assume_centered=False, precisions_init=None,
                             probe_function=None, verbose=0, debug=False):
    """Internal version of group_sparse_covariance. See its doctype for details

    Parameters
    ----------
    probe_function: callable
        called at the end of each iteration. If the function returns
        True, then optimization is stopped prematurely. It is also called
        before the first iteration.

    precisions_init: numpy.ndarray
        initial value of the precision matrices. Used by the cross-validation
        function.
    """
    ## initial value of the precision matrices, or way of computing it.
    ## Possible values: "identity" (default), "ledoit-wolf" or numpy.ndarray
    ## giving actual value.

    if tol == -1:
        tol = None
    if not isinstance(rho, (int, float)) or rho < 0:
        raise ValueError("Regularization parameter rho must be a "
                         "positive number.\n"
                         "You provided: {0}".format(str(rho)))
    n_tasks = emp_covs.shape[-1]
    n_var = emp_covs[0].shape[0]
    n_samples = np.asarray(n_samples)
    n_samples /= n_samples.sum()  # essential for numerical stability

    if precisions_init is None:
        omega = np.ndarray(shape=emp_covs.shape, dtype=emp_covs.dtype,
                           order="F")
        for k in xrange(n_tasks):
            # Values on main diagonals are far from zero, because they
            # are timeseries energy.
            omega[..., k] = np.diag(1. / np.diag(emp_covs[..., k]))
    else:
        omega = precisions_init.copy()

    # Preallocate arrays
    y = np.ndarray(shape=(n_tasks, n_var - 1), dtype=emp_covs.dtype)
    u = np.ndarray(shape=(n_tasks, n_var - 1), dtype=emp_covs.dtype)
    y_1 = np.ndarray(shape=(n_tasks, n_var - 2), dtype=emp_covs.dtype)
    h_12 = np.ndarray(shape=(n_tasks, n_var - 2), dtype=emp_covs.dtype)
    q = np.ndarray(shape=(n_tasks,), dtype=emp_covs.dtype)
    c = np.ndarray(shape=(n_tasks,), dtype=emp_covs.dtype)
    W = np.ndarray(shape=(omega.shape[0] - 1, omega.shape[1] - 1,
                          omega.shape[2]),
                   dtype=emp_covs.dtype, order="F")
    Winv = np.ndarray(shape=W.shape, dtype=emp_covs.dtype, order="F")

    # Auxilliary arrays.
    v = np.ndarray((omega.shape[0] - 1,), dtype=omega.dtype)
    h = np.ndarray((omega.shape[1] - 1,), dtype=omega.dtype)

    # Optional.
    tolerance_reached = False
    max_norm = None

    # Start optimization loop. Variables are named following (mostly) the
    # Honorio-Samaras paper notations.
    omega_old = np.empty(omega.shape, dtype=omega.dtype)
    if probe_function is not None:
        # iteration number -1 means called before iteration loop.
        probe_function(emp_covs, n_samples, rho, max_iter, tol,
                       -1, omega, None)

    for n in xrange(max_iter):
        if verbose >= 1:
            if max_norm is not None:
                suffix = (" max norm: {max_norm:.3e} ".format(
                    max_norm=max_norm))
            else:
                suffix = ""
            print("* iteration {iter_n:d} ({percentage:.0f} %){suffix} ..."
                  "".format(iter_n=n, percentage=100. * n / max_iter,
                            suffix=suffix))

        omega_old[...] = omega
        for p in xrange(n_var):

            if p == 0:
                # Initial state: remove first col/row
                W = omega[1:, 1:, :].copy()   # stack of W(k)
                Winv = np.ndarray(shape=W.shape, dtype=np.float)
                for k in xrange(W.shape[2]):
                    # stack of W^-1(k)
                    Winv[..., k] = np.linalg.inv(W[..., k])
                    if debug:
                        np.testing.assert_almost_equal(
                            np.dot(Winv[..., k], W[..., k]),
                            np.eye(Winv[..., k].shape[0]), decimal=12)
                        assert_submatrix(omega[..., k], W[..., k], p)
            else:
                # Update W and Winv
                if debug:
                    omega_orig = omega.copy()

                for k in xrange(n_tasks):
                    update_submatrix(omega[..., k],
                                     W[..., k], Winv[..., k], p, h, v)

                    if debug:
                        assert_submatrix(omega[..., k], W[..., k], p)
                        np.testing.assert_almost_equal(
                            np.dot(Winv[..., k], W[..., k]),
                            np.eye(Winv[..., k].shape[0]), decimal=12)
                        assert(is_spd(W[..., k]))
                        assert(is_spd(Winv[..., k], decimal=14))
                if debug:
                    np.testing.assert_almost_equal(omega_orig, omega)

            # In the following lines, implicit loop on k (tasks)
            # Extract y and u
            y[:, :p] = omega[:p, p, :].T
            y[:, p:] = omega[p + 1:, p, :].T

            u[:, :p] = emp_covs[:p, p, :].T
            u[:, p:] = emp_covs[p + 1:, p, :].T

            for m in xrange(n_var - 1):
                # Coordinate descent on y

                # T(k) -> n_samples[k]
                # v(k) -> emp_covs[p, p, k]
                # h_22(k) -> Winv[m, m, k]
                # h_12(k) -> Winv[:m, m, k],  Winv[m+1:, m, k]
                # y_1(k) -> y[k, :m], y[k, m+1:]
                # u_2(k) -> u[k, m]
                h_12[:, :m] = Winv[:m, m, :].T
                h_12[:, m:] = Winv[m + 1:, m, :].T
                y_1[:, :m] = y[:, :m]
                y_1[:, m:] = y[:, m + 1:]

                c[:] = - n_samples * (
                    emp_covs[p, p, :] * (h_12 * y_1).sum(axis=1) + u[:, m]
                    )
                c2 = np.sqrt(np.dot(c, c))

                # x -> y[:][m]
                if c2 <= rho:
                    y[:, m] = 0  # x* = 0
                else:
                    # q(k) -> T(k) * v(k) * h_22(k)
                    # \lambda -> alpha   (lambda is a Python keyword)
                    q = n_samples * emp_covs[p, p, :] * Winv[m, m, :]
                    if debug:
                        assert(np.all(q > 0))
                    # x* = \lambda* diag(1 + \lambda q)^{-1} c
                    # Precompute some quantities
                    cc = c * c
                    two_ccq = 2. * cc * q
                    # tolerance does not seem to be important for
                    # numerical stability (tol=1e-2 works) but has an
                    # effect on overall convergence rate.
                    # (often the tighter the better)

                    # Newton-Raphson loop
                    alpha = 0.
                    for _ in xrange(100):
                        fder = quad_trust_region_deriv(alpha, q, two_ccq,
                                                       cc, rho ** 2)
                        if fder == 0:
                            msg = "derivative was zero."
                            warnings.warn(msg, RuntimeWarning)
                            break
                        fval = quad_trust_region(alpha, q, two_ccq,
                                                 cc, rho ** 2)
                        p1 = alpha - fval / fder
                        remainder = abs(p1 - alpha)
                        alpha = p1
                        if remainder < 1.5e-8:
                            break

                    if remainder > 0.1:
                        warnings.warn("Newton-Raphson step did not converge.\n"
                                      "This may indicate a badly conditioned "
                                      "system.")

                    if debug:
                        assert alpha >= 0, alpha
                    y[:, m] = (alpha * c) / (1. + alpha * q)  # x*

            # Copy back y in omega (column and row)
            omega[:p, p, :] = y[:, :p].T
            omega[p + 1:, p, :] = y[:, p:].T
            omega[p, :p, :] = y[:, :p].T
            omega[p, p + 1:, :] = y[:, p:].T

            for k in xrange(n_tasks):
                omega[p, p, k] = 1. / emp_covs[p, p, k] + np.dot(
                    np.dot(y[k, :], Winv[..., k]), y[k, :])

                if debug:
                    assert(is_spd(omega[..., k]))

        # Compute max of variation
        omega_old -= omega
        max_norm = abs(omega_old).max()

        if probe_function is not None:
            if probe_function(emp_covs, n_samples, rho, max_iter, tol,
                              n, omega, omega_old) is True:
                print("probe_function interrupted loop")
                break

        if tol is not None and max_norm < tol:
            if verbose >= 1:
                print("tolerance reached at iteration number {0:d}: {1:.3e}"
                      "".format(n + 1, max_norm))
            tolerance_reached = True
            break

    if tol is not None and not tolerance_reached:
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
    rho: float
        regularization parameter. With normalized covariances matrices and
        number of samples, sensible values lie in the [0, 1] range(zero is
        no regularization: output is not sparse)

    tol: positive float, optional
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped

    max_iter: int
        maximum number of iterations. The default value (10) is rather
        conservative.

    verbose: int
        verbosity level. Zero means "no message".

    assume_centered: bool
        if True, assume that all signals passed to fit() are centered.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Caching aggressiveness. Higher values mean more caching.

    Attributes
    ----------
    `covariances_`: 3D numpy.ndarray
        maximum likelihood covariance estimations.
        Shape: (n_features, n_features, n_tasks)

    `precisions_`: 3D numpy.ndarray
        precisions matrices estimated using Antonio & Samaras algorithm.
        Shape: (n_features, n_features, n_tasks)
    """

    def __init__(self, rho=0.1, tol=1e-3, max_iter=10, verbose=1,
                 assume_centered=False,
                 memory=Memory(cachedir=None), memory_level=0):
        self.rho = rho
        self.tol = tol
        self.max_iter = max_iter
        self.assume_centered = assume_centered

        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

    def fit(self, tasks, y=None):
        """Fits the group sparse precision model according to the given
        training data and parameters.

        Parameters
        ----------
        tasks: list of numpy.ndarray
            input tasks. Each task is a 2D array, whose columns contain
            signals. Each array shape must be (sample number, feature number).
            The sample number can vary from task to task, but all tasks must
            have the same number of features (i.e. of columns).

        Attributes
        ----------
        `covariances_`: numpy.ndarray
            empirical covariances

        `precisions_`: numpy.ndarray
            precision matrices

        Returns
        -------
        self: object
            the object itself. Useful for chaining operations.
        """

        self.log("Computing covariance matrices")
        self.covariances_, n_samples, _, _ = empirical_covariances(
                tasks, assume_centered=self.assume_centered, debug=False)

        self.log("Computing precision matrices")
        ret = self._cache(
            _group_sparse_covariance, memory_level=1)(
                self.covariances_, n_samples, self.rho,
                tol=self.tol, max_iter=self.max_iter,
                assume_centered=self.assume_centered,
                verbose=self.verbose - 1, debug=False)

        self.precisions_ = ret
        return self


def empirical_covariances(tasks, assume_centered=False, dtype=np.float64,
                          debug=False):
    """Compute empirical covariances for several signals.

    Parameters
    ----------
    tasks: list of numpy.ndarray
        input tasks. Each task is a 2D array, whose columns contain signals.
        Each array shape must be (sample number, feature number). The sample
        number can vary from task to task, but all tasks must have the same
        number of features (i.e. of columns).

    assume_centered: bool, optional
        if True, assume that all input signals are centered. This slightly
        decreases computation time by avoiding useless computation.

    dtype: numpy dtype
        dtype of output array. Default: numpy.float64

    debug: bool, optional
        if True, perform checks during computation. It can help find
        numerical problems, but increases computation time a lot.

    Returns
    -------
    emp_covs: numpy.ndarray
        empirical covariances.
        shape: (feature number, feature number, task number)

    n_samples: numpy.ndarray
        number of samples for each task. shape: (task number,)
    """
    if not hasattr(tasks, "__iter__"):
        raise ValueError("'tasks' input argument must be an iterable. "
                         "You provided {0}".format(tasks.__class__))

    n_tasks = [s.shape[1] for s in tasks]
    if len(set(n_tasks)) > 1:
        raise ValueError("All tasks must have the same number of features.\n"
                         "You provided: {0}".format(str(n_tasks)))
    n_tasks = len(tasks)
    n_var = tasks[0].shape[1]

    # Enable to change dtype here because depending on user conversion from
    # single precision to double will be required or not.
    emp_covs = np.empty((n_var, n_var, n_tasks), dtype=dtype, order="F")
    for k, s in enumerate(tasks):
        M = empirical_covariance(
            s, assume_centered=assume_centered)

        emp_covs[..., k] = M + M.T
        if debug:
            assert(is_spd(emp_covs[..., k]))
    emp_covs /= 2

    n_samples = np.asarray([s.shape[0] for s in tasks], dtype=np.float64)

    return emp_covs, n_samples, n_tasks, n_var


def group_sparse_score(precisions, n_samples, emp_covs, rho):
    """Compute the log-likelihood of a given list of empirical covariances /
    precisions.

    This is the loss function used by the group_sparse_covariance function,
    without the regularization term.

    Parameters
    ----------
    precisions: numpy.ndarray
        estimated precisions. shape (n_var, n_var, n_tasks)

    n_samples: array-like
        number of samples used in estimating each task in "precisions".
        shape: (n_tasks,)

    Returns
    -------
    score: float
        value of loss function.
    """
    ll = 0
    for k in range(precisions.shape[2]):
        ll += n_samples[k] * sklearn.covariance.log_likelihood(
            emp_covs[..., k], precisions[..., k])

    l2 = np.sqrt((precisions ** 2).sum(axis=-1))
    l12 = l2.sum() - np.diag(l2).sum()  # Do not count diagonal terms

    return (-ll, rho * l12 - ll)


def group_sparse_covariance_path(train_tasks, rhos, test_tasks=None, tol=1e-3,
                                 max_iter=10, assume_centered=False,
                                 precisions_init=None, verbose=0, debug=False):
    """Get estimated precision matrices for different values of rho.

    Calling this function is faster than calling group_sparse_covariance()
    repeatedly, because it makes use of the first result to initialize the
    next computation.

    Parameters
    ----------
    train_tasks : list of numpy.ndarray
        list of signals.

    rhos : list of float
         values of rho to use. Best results for sorted values (decreasing)

    test_tasks : list of numpy.ndarray
        list of signals, independent from those in train_tasks, on which to
        compute a score. If None, no score is computed.

    verbose : int
        verbosity level

    tol, max_iter, assume_centered, debug, precisions_init :
        Passed to group_sparse_covariance(). See the corresponding docstring
        for details.

    Returns
    -------
    precisions_list: list of numpy.ndarray
        estimated precisions for each value of rho provided. The length on this
        list is the same as that of parameter rhos.

    scores: list of float
        for each estimated precision, score obtained on the test set. Output
        only if test_tasks is not None.
    """
    train_covs, train_n_samples, _, _ = empirical_covariances(
        train_tasks, assume_centered=assume_centered, debug=debug)
    test_covs, _, _, _ = empirical_covariances(
        test_tasks, assume_centered=assume_centered, debug=debug)

    scores = []
    precisions_list = []
    for rho in rhos:
        precisions = _group_sparse_covariance(
            train_covs, train_n_samples, rho, tol=tol, max_iter=max_iter,
            assume_centered=assume_centered, precisions_init=precisions_init,
            verbose=verbose, debug=debug)

        # Compute log-likelihood
        if test_tasks is not None:
            scores.append(group_sparse_score(precisions, train_n_samples,
                                             test_covs, 0))
        precisions_list.append(precisions)
        precisions_init = precisions

    if test_tasks is not None:
        return precisions_list, scores
    else:
        return precisions_list


class GroupSparseCovarianceCV(object):
    # See also GraphLasso in scikit-learn.
    """
    Parameters
    ----------
    cv: integer
        number of folds in a K-fold cross-validation scheme.
    """
    def __init__(self, rhos=4, n_refinements=4, cv=None,
                 tol=1e-3, max_iter=10, assume_centered=False, verbose=1,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, debug=False):
        self.rhos = rhos
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

    def fit(self, tasks, y=None):
        """Compute cross-validated group-sparse precision.

        Parameters
        ----------
        tasks: list of numpy.ndarray
            input tasks. Each task is a 2D array, whose columns contain
            signals. Each array shape must be (sample number, feature number).
            The sample number can vary from task to task, but all tasks must
            have the same number of features (i.e. of columns).

        Attributes
        ----------
        `covariances_`: numpy.ndarray
        `precision_`: numpy.ndarray
        `rho_`: selected value for penalization parameter
        `cv_rhos`: list of float
            All penalization values explored.
        `cv_scores`: numpy.ndarray with shape (n_rhos, n_folds)
        """

        # Empirical covariances
        emp_covs, n_samples, n_tasks, _ = \
                  empirical_covariances(tasks,
                                        assume_centered=self.assume_centered,
                                        debug=self.debug)

        # One cv generator per task must be created, because each task can
        # have a different number of samples.
        cv = []
        for k in range(n_tasks):
            cv.append(sklearn.cross_validation.check_cv(
                self.cv, tasks[k], None, classifier=False))

        path = list()  # List of (rho, scores, covs)
        n_rhos = self.rhos

        if isinstance(n_rhos, collections.Sequence):
            rhos = list(self.rhos)
            n_rhos = len(rhos)
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            rho_1, _ = rho_max(emp_covs, n_samples)
            rho_0 = 1e-2 * rho_1
            rhos = np.logspace(np.log10(rho_0), np.log10(rho_1),
                               n_rhos)[::-1]

        covs_init = itertools.repeat(None)
        for i in range(n_refinements):
            # Compute the cross-validated loss on the current grid
            train_test_tasks = []
            for train_test in zip(*cv):
                assert(len(train_test) == n_tasks)
                train_test_tasks.append(zip(*[(task[train, :], task[test, :])
                                              for task, (train, test)
                                              in zip(tasks, train_test)]))

            this_path = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(group_sparse_covariance_path)(
                    train_tasks, rhos, test_tasks=test_tasks,
                    max_iter=self.max_iter, tol=self.tol,
                    assume_centered=self.assume_centered,
                    verbose=self.verbose, debug=self.debug,
                    precisions_init=prec_init)
                for (train_tasks, test_tasks), prec_init
                in zip(train_test_tasks, covs_init))

            # this_path[i] is a tuple (precisions_list, scores)
            # - scores: scores obtained with the i-th folding, for varying rho.
            # - precisions_list: corresponding precisions matrices, for each
            #   value of rho.
            precisions_list, scores = zip(*this_path)
            precisions_list = zip(*precisions_list)
            scores = [np.mean(sc) for sc in zip(*scores)]

            # scores is the mean score obtained for a given value of rho.
            path.extend(zip(rhos, scores, precisions_list))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using the built-in 'max' function to
            # have a fully-reproducible selection of the smallest rho
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (rho, this_score, _) in enumerate(path):
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
                # the highest value of rho for which there are
                # non-zero coefficients
                rho_1 = path[0][0]
                rho_0 = path[1][0]
                covs_init = path[0][2]
            elif (best_index == last_finite_idx
                    and not best_index == len(path) - 1):
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                rho_1 = path[best_index][0]
                rho_0 = path[best_index + 1][0]
                covs_init = path[best_index][2]
            elif best_index == len(path) - 1:
                rho_1 = path[best_index][0]
                rho_0 = 0.01 * path[best_index][0]
                covs_init = path[best_index][2]
            else:
                rho_1 = path[best_index - 1][0]
                rho_0 = path[best_index + 1][0]
                covs_init = path[best_index - 1][2]
            rhos = np.logspace(np.log10(rho_1), np.log10(rho_0), len(rhos) + 2)
            rhos = rhos[1:-1]
            if self.verbose and n_refinements > 1:
                print("[GroupSparseCovarianceCV] Done refinement "
                      "% 2i out of %i" % (i + 1, n_refinements))

        path = list(zip(*path))
        cv_scores = list(path[1])
        rhos = list(path[0])

        self.cv_scores = np.array(cv_scores)
        self.rho_ = rhos[best_index]
        self.cv_rhos = rhos

        # Finally fit the model with the selected rho
        self.covariances_ = emp_covs
        self.precisions_ = _group_sparse_covariance(
            emp_covs, n_samples, self.rho_, tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose - 1, debug=self.debug)
        return self
