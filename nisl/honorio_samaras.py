"""
Implementation of algorithm for sparse multi-task learning of gaussian
graphical models.

Honorio, Jean, and Dimitris Samaras.
"Simultaneous and Group-Sparse Multi-Task Learning of Gaussian Graphical
Models". arXiv:1207.4255 (17 July 2012). http://arxiv.org/abs/1207.4255.

"""
# Authors: Philippe Gervais
# License: simplified BSD

import warnings

import numpy as np
import scipy
import scipy.optimize

from sklearn.utils.extmath import fast_logdet


def symmetrize(M):
    M[...] = M + M.T
    M[...] /= 2.

#@profile
def honorio_samaras(emp_covs, rho, n_samples=None, n_iter=10, verbose=0,
                    debug=False, normalize_n_samples=True):
    """
    Parameters
    ==========
    rho: float
        regularization parameter. With normalized covariances matrices and
        number of samples, sensible values lie in the [0, 1] range.

    covariances: list of numpy.ndarray
        covariance estimates. All shapes must be identical. Must be positive
        semidefinite. Normalizing these matrices (e.g. having ones on the
        diagonal) is not required, but recommended.

    n_samples: array-like or None
        number of samples for each task. len(n_samples) == len(emp_covs)
        if n_samples is None, then the number of samples is assumed to be
        identical for each task.

    n_iter: int
        number of iteration

    debug: bool
        if True, perform checks during computation. It can help finding
        numerical instabilities.

    normalize_n_samples: bool
        if True, divides n_samples by the maximum value. This improves
        numerical stability a lot.

    Returns
    =======
    omega:
        estimated precision matrices
    """

    # Test input arguments
    # FIXME: Check that all covariances have the same size.
    # FIXME: check consistency between matrix sizes and task number.
    if rho < 0:
        raise ValueError("Regularization parameter rho must be positive.\n"
                         "You provided: {0}".format(str(rho)))

    # allow passing covariances as a 3D array, not a list.
    emp_covs = np.dstack(emp_covs)
    n_var = emp_covs.shape[0]

    assert(emp_covs.shape[0] == emp_covs.shape[1])

    if n_samples is not None:
        n_samples = np.asarray(n_samples, dtype=np.float)
    else:
        n_samples = np.ones(emp_covs.shape[-1])
    if normalize_n_samples:
        n_samples /= n_samples.max()

    n_tasks = emp_covs.shape[2]
    for k in xrange(n_tasks):
        symmetrize(emp_covs[..., k])
        assert_spd(emp_covs[..., k])

    omega = np.ndarray(shape=emp_covs.shape, dtype=np.float)
    for k in xrange(n_tasks):
        # Values on main diagonals should be far from zero, because they
        # are timeseries energy.
        omega[..., k] = np.diag(1. / np.diag(emp_covs[..., k]))

    # debugging
    all_crit = []

    # Preallocate arrays
    y = np.ndarray(shape=(n_tasks, n_var - 1), dtype=np.float)
    u = np.ndarray(shape=(n_tasks, n_var - 1))
    y_1 = np.ndarray(shape=(n_tasks, n_var - 2))
    h_12 = np.ndarray(shape=(n_tasks, n_var - 2))
    q = np.ndarray(shape=(n_tasks,))
    c = np.ndarray(shape=(n_tasks,))
    W = np.ndarray(shape=(omega.shape[0] - 1, omega.shape[1] - 1,
                          omega.shape[2]),
                   dtype=np.float)
    Winv = np.ndarray(shape=W.shape, dtype=np.float)

    for n in xrange(n_iter):
        if verbose >= 1:
            print("\n-- Starting {iter_n:d}-th iteration...".format(iter_n=n))

        for p in xrange(n_var):

            if p == 0:
                # Initial state: remove first col/row
                W = omega[1:, 1:, :].copy()   # stack of W(k)
                Winv = np.ndarray(shape=W.shape, dtype=np.float)
                for k in xrange(W.shape[2]):
                    # stack of W^-1(k)

# scipy.linalg.pinvh does no exist in scipy 0.7.0.
#                    Winv[..., k], rank = \
#                        scipy.linalg.pinvh(W[..., k], return_rank=True)
                    Winv[..., k] = np.linalg.inv(W[..., k])
                    if debug:
#                        assert(rank == Winv[..., k].shape[0])
                        np.testing.assert_almost_equal(
                            np.dot(Winv[..., k], W[..., k]),
                            np.eye(Winv[..., k].shape[0]), decimal=12)
                        assert_submatrix(omega[..., k], W[..., k], p)
            else:
                # Update W and Winv
                if debug:
                    omega_orig = omega.copy()

                for k in xrange(n_tasks):
                    update_submatrix(omega[..., k], W[..., k], Winv[..., k], p)
                    if debug:
#                        assert_submatrix(omega[..., k], W[..., k], p)
                        np.testing.assert_almost_equal(
                            np.dot(Winv[..., k], W[..., k]),
                            np.eye(Winv[..., k].shape[0]), decimal=12)
                    assert_spd(W[..., k], debug=debug)
                    assert_spd(Winv[..., k], debug=debug)
                if debug:
                    np.testing.assert_almost_equal(omega_orig, omega)

            # In the following lines, implicit loop on k (tasks)
            # Extract y
            y[:, :p] = omega[:p, p, :].T
            y[:, p:] = omega[p + 1:, p, :].T
            # Extract u (TODO: precompute? emp_covs does not change.)
            u[:, :p] = emp_covs[:p, p, :].T
            u[:, p:] = emp_covs[p + 1:, p, :].T

            if verbose >= 2:
                print("\n-- entering coordinate descent loop (%d)" % p)

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

                # Compute c
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
                    if True:
                        # Precompute some quantities
                        cc = c * c
                        two_ccq = 2. * cc * q
                        # tolerance does not seem to be important for
                        # numerical stability (tol=1e-2 works)
                        alpha = scipy.optimize.newton(
                            optimized_quadratic_trust_region, 0,
                            fprime=optimized_quadratic_trust_region_deriv,
                            args=(q, two_ccq, cc, rho ** 2),
                            maxiter=50)

                        remainder = optimized_quadratic_trust_region(
                            alpha, q, two_ccq, cc, rho ** 2)
                    else:
                        alpha = scipy.optimize.newton(
                            quadratic_trust_region, 0,
                            fprime=quadratic_trust_region_deriv,
                            args=(c, q, rho),
                            maxiter=50)

                        remainder = quadratic_trust_region(
                            alpha, c, q, rho ** 2)

                    if abs(remainder) > 0.1:
                        warnings.warn("Newton-Raphson step did not converge.\n"
                                      "This indicates a badly conditioned "
                                      "system.\n"
                                      "Try option normalize_n_samples=True or "
                                      "ensure that values on the main diagonal"
                                      "of \nempirical covariances are equal "
                                      "to one.")

                    if debug:
                        assert alpha >= 0, alpha
                    y[:, m] = (alpha * c) / (1. + alpha * q)  # x*

            # These lines can be put out of this loop. Used only to
            # compute criterion for debugging.
            # Copy back y in omega (column and row)
            omega[:p, p, :] = y[:, :p].T
            omega[p + 1:, p, :] = y[:, p:].T
            omega[p, :p, :] = y[:, :p].T
            omega[p, p + 1:, :] = y[:, p:].T

            for k in xrange(n_tasks):
                omega[p, p, k] = 1. / emp_covs[p, p, k] + np.dot(
                    np.dot(y[k, :], Winv[..., k]), y[k, :])

                # Check that all omega are symmetric positive definite
                if debug:
                    assert_spd(omega[..., k], debug=debug)

            criterion = display_criterion(n_tasks, n_samples, rho,
                                          omega, emp_covs,
                                          display=verbose >= 1)
            all_crit.append(criterion)

    return omega, all_crit


def display_criterion(n_tasks, n_samples, rho, omega, emp_covs, display=True):
    # Compute optimization criterion (for display)
    ll = 0  # log-likelihood
    for k in xrange(n_tasks):
        t = fast_logdet(omega[..., k])
        t -= (omega[..., k] * emp_covs[..., k]).sum()
        ll += n_samples[k] * t

    # L(1,2)-norm
    l2 = np.sqrt((omega ** 2).sum(axis=-1))
    # Do not count diagonal terms
    l12 = l2.sum() - np.diag(l2).sum()
    criterion = - (ll - rho * l12)
    if display:
        print("Criterion to minimize: {criterion:.8f}".format(
                criterion=criterion))
    return criterion


def display_criterion_2(n_tasks, n_samples, rho, omega, emp_covs,
                        y, u, Winv, p, display=True):
    criterion = 0
    for k in xrange(n_tasks):
        t = 0
        t += np.log(omega[p, p, k]
                    - np.dot(np.dot(y[k, :], Winv[..., k]), y[k, :]))
        t += - 2 * np.dot(u[k, :], y[k, :]) - emp_covs[p, p, k] * omega[p, p, k]
        criterion += n_samples[k] * t
    criterion += - 2. * rho * np.sqrt((y * y).sum(axis=0)).sum()
    criterion = -criterion
    if display:
        print("Criterion(2) to minimize: {criterion:.8f}".format(
                criterion=criterion))
    return criterion


def quadratic_trust_region_criterion(l, c, q, rho):
    if l < 0:
        ret = ((c * c) / q).sum()  # value at zero. Just for display
    else:
        ret = ((c * c) / (q * (1 + l * q))).sum() + (rho ** 2) * l
    return ret


#@profile
def optimized_quadratic_trust_region(alpha, q, two_ccq, cc, rho2):
    return rho2 - (cc / ((1. + alpha * q) ** 2)).sum()


#@profile
def optimized_quadratic_trust_region_deriv(alpha, q, two_ccq, cc, rho2):
    return (two_ccq / (1. + alpha * q) ** 3).sum()

#@profile
def quadratic_trust_region(l, c, q, rho):
    if l < 0:
        slope = 2 * (c * c * q).sum()
        return rho ** 2 - (c * c).sum() + l * slope

    return rho ** 2 - ((c * c) / ((1. + l * q) ** 2)).sum()

#@profile
def quadratic_trust_region_deriv(l, c, q, rho):
    if l < 0:
        return 2 * ((c * c) * q).sum()
    return 2 * ((c * c) * q / ((1. + l * q) ** 3)).sum()


def test_plot_quadratic_trust_region():
    import pylab as pl

    rand_gen = np.random.RandomState(0)
    size = 5

    rho = 0.5
    c = 2. * rand_gen.randn(size)
    q = abs(20. * rand_gen.randn(size))

    c2 = np.sqrt((c * c).sum())
    t = pl.linspace(-10., c2 / rho, 500)
    criterion = [quadratic_trust_region_criterion(l, c, q, rho) for
                 l in t]
    criterion_deriv = [quadratic_trust_region(l, c, q, rho) for
                       l in t]
    criterion_dderiv = [quadratic_trust_region_deriv(l, c, q, rho) for
                       l in t]

    pl.plot(t, criterion, label="criterion")
    pl.plot(t, criterion_deriv, label="deriv 1")
    pl.plot(t, criterion_dderiv, label="deriv 2")
    pl.grid()
    pl.show()


def test_quadratic_trust_region():
    import pylab as pl

    rand_gen = np.random.RandomState(0)
    size = 5

    rho = 0.01
    c = 2. * rand_gen.randn(size)
    q = abs(20. * rand_gen.randn(size))

    c2 = np.sqrt((c * c).sum())
    print ("c2: %.2f" % c2)
    if c2 <= rho:
        print("bad input values")

    l = 0.

    print("criterion before: %.2f"
          % quadratic_trust_region_criterion(l, c, q, rho))

    t = pl.linspace(-10., 2. * c2 / rho, 1000)
    criterion = [quadratic_trust_region_criterion(l, c, q, rho) for
                 l in t]
    deriv = [quadratic_trust_region(l, c, q, rho) for l in t]
    pl.plot(t, criterion, label="criterion")
    pl.plot(t, deriv, label="deriv")
    pl.legend()
    pl.grid()

    l = scipy.optimize.newton(
        quadratic_trust_region, l,
        fprime=quadratic_trust_region_deriv,
        args=(c, q, rho))

    print("criterion after: %.2f. l = %.2f"
          % (quadratic_trust_region_criterion(l, c, q, rho), l))
    print("deriv after: %.2f. l = %.2f"
          % (quadratic_trust_region(l, c, q, rho), l))
    pl.show()


def update_submatrix(full, sub, sub_inv, n):
    sub[:n, :n] = full[:n, :n]
    sub[n:, n:] = full[n + 1:, n + 1:]
    sub[:n, n:] = full[:n, n + 1:]
    sub[n:, :n] = full[n + 1:, :n]
    sub_inv[...] = np.linalg.inv(sub)
    symmetrize(sub_inv)


def update_submatrix2(full, sub, sub_inv, p):
    """Update submatrix and its inverse.

    sub_inv is the inverse of the submatrix of "full" obtained by removing
    the p-th row and column.

    sub_inv is modified in-place. After execution of this function, it contains
    the inverse of the submatrix of "full" obtained by removing the n+1-th row
    and column.

    This computation is based on Sherman-Woodbury-Morrison identity.
    """

    n = p - 1
    h, v = update_vectors(full, n)

    # change row
    coln = sub_inv[:, n]  # A^(-1)*U
    V = h - sub[n, :]
    coln = coln / (1. + np.dot(V, coln))
    sub_inv -= np.outer(coln, np.dot(V, sub_inv))
    sub[n, :] = h

    # change column
    rown = sub_inv[n, :]  # V*A^(-1)
    U = v - sub[:, n]
    rown = rown / (1. + np.dot(rown, U))
    sub_inv -= np.outer(np.dot(sub_inv, U), rown)
    sub[:, n] = v   # equivalent to sub[n, :] += U


def assert_submatrix(full, sub, n):
    """Check that "sub" is the matrix obtained by removing the p-th col and row
    in "full".
    """
    true_sub = np.ndarray(shape=sub.shape, dtype=sub.dtype)
    true_sub[:n, :n] = full[:n, :n]
    true_sub[n:, n:] = full[n + 1:, n + 1:]
    true_sub[:n, n:] = full[:n, n + 1:]
    true_sub[n:, :n] = full[n + 1:, :n]

    np.testing.assert_almost_equal(true_sub, sub)


def test_update_submatrix():
    N = 5
    rand_gen = np.random.RandomState(0)
    M = rand_gen.randn(N, N)
    sub = M[1:, 1:].copy()
    sub_inv = np.linalg.inv(M[1:, 1:])
    true_sub = np.zeros(sub.shape, dtype=sub.dtype)
    true_inv = sub_inv.copy()

    for n in xrange(1, N):
        update_submatrix(M, sub, sub_inv, n)

        true_sub[n:, n:] = M[n + 1:, n + 1:]
        true_sub[:n, :n] = M[:n, :n]
        true_sub[n:, :n] = M[n + 1:, :n]
        true_sub[:n, n:] = M[:n, n + 1:]
        true_inv = np.linalg.inv(true_sub)
        np.testing.assert_almost_equal(true_sub, sub)
        np.testing.assert_almost_equal(true_inv, sub_inv)


def update_vectors(full, n):
    """full is a (N, N) matrix.

    This function is a helper function for updating the submatrix equals to
    "full" with row n + 1 and column n + 1 removed. The initial state of the
    submatrix is supposed to be "full" with row and column n removed.

    This functions returns the new value of row and column n in the submatrix.
    Thus, if h, v are the return values of this function, the submatrix must
    be updated this way: sub[n, :] = h ; sub[:, n] = v
    """
    v = np.ndarray((full.shape[0] - 1,), dtype=full.dtype)
    v[:n + 1] = full[:n + 1, n]
    v[n + 1:] = full[n + 2:, n]

    h = np.ndarray((full.shape[1] - 1,), dtype=full.dtype)
    h[:n + 1] = full[n, :n + 1]
    h[n + 1:] = full[n, n + 2:]

    return h, v


def generate_multi_task_gg_model(n_tasks=5, n_var=10, density=0.2,
                                 min_eigenvalue=0.1,
                                 min_samples=30, max_samples=50,
                                 rand_gen=np.random.RandomState(0)):
    """Generate signals drawn from a sparse gaussian graphical models.

    Parameters
    ==========
    n_tasks: int
        number of tasks

    n_var: int
        number of signals per task to generate

    density: float
        density of edges in graph topology

    min_eigenvalue: float
        To ensure positive definiteness of covariance matrices, make sure that
        the smallest eigenvalue is greater than this number.

    min_samples, max_samples: int
        Each task have a different number of samples, between these two
        numbers. All signals in a given task have the same number of samples.

    Returns
    =======
    tasks: list of signals
        tasks[n] is the signals for task n. They are provided as a numpy array
        with shape (sample number, n_var). len(tasks) == n_tasks

    topology: numpy.array
        binary array giving the graph topology used for generating covariances
        and signals.
    """

    # Generate topology (upper triangular binary matrix, with ones on the
    # diagonal)
    topology = np.ndarray(shape=(n_var, n_var))
    topology[:, :] = np.triu((
        rand_gen.randint(0, high=int(1. / density), size=n_var * n_var)
        ).reshape(n_var, n_var) == 0, k=1)

    # Generate edges weights on precision matrices
    precisions = []
    for _ in xrange(n_tasks):

        while True:
            weights = rand_gen.uniform(low=-.1, high=1., size=n_var * n_var
                                       ).reshape((n_var, n_var))
            prec = topology * weights
            prec += np.triu(prec, k=1).T
            prec += np.eye(*prec.shape)
            # Loop until a positive definite matrix is obtained.
            if np.linalg.eigvals(prec).min() > min_eigenvalue:
                precisions.append(prec)
                break

    for prec in precisions:
        np.testing.assert_almost_equal(prec, prec.T)

    # Returns a symmetric topology matrix
    topology = topology + np.triu(topology, k=1).T
    topology += np.eye(*topology.shape)

    assert(np.all(topology == topology.T))

    signals = []
    mean = np.zeros(topology.shape[0])
    n_samples = rand_gen.randint(min_samples, high=max_samples,
                                 size=len(precisions))

    for n, prec in zip(n_samples, precisions):
        signals.append(rand_gen.multivariate_normal(mean, -np.linalg.inv(prec),
                                                    (n,)))

    return signals, precisions, topology


def assert_spd(M, debug=True):
    if debug:
        np.testing.assert_almost_equal(M, M.T, decimal=15)
        eigvalsh = np.linalg.eigvalsh(M)
        assert eigvalsh.min() > 0, eigvalsh


if __name__ == "__main__":
    display = True

    signals, precisions, topology = generate_multi_task_gg_model(density=0.3,
                                                           n_tasks=5, n_var=10,
                                             min_samples=100, max_samples=101)
    emp_covs = [np.dot(s.T, s) / len(s) for s in signals]

    if display:
        import pylab as pl

    rho = 70.
    n_samples = [len(signal) for signal in signals]

    omega, all_crit = honorio_samaras(rho, emp_covs, n_samples, n_iter=15,
                                      verbose=1, debug=False)

    if display:

        for n, value in enumerate(zip(signals, precisions)):
            s, prec = value
            pl.figure()
            pl.subplot(1, 2, 1)
            pl.imshow(omega[..., n] != 0, interpolation="nearest", cmap="gray",
                      vmin=0, vmax=1)
            pl.colorbar()
            pl.title("sparsity {n:d}".format(n=n))

            pl.subplot(1, 2, 2)
            pl.imshow(omega[..., n], interpolation="nearest", cmap="gray")
            pl.colorbar()
            pl.title("precision {n:d}".format(n=n))

    if display:
        pl.figure()
        last = all_crit[-1]
        pl.loglog(np.asarray(all_crit - last
                             + 4 * (all_crit[-2] - all_crit[-1])))
        pl.ylabel("criterion")
        pl.xlabel("iteration")

        pl.figure()
        pl.imshow(topology, interpolation="nearest", cmap="gray")
        pl.colorbar()
        pl.title("true sparsity")

    if display:
        pl.show()
