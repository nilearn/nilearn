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
import sys

import numpy as np
import scipy.optimize

# Test:
# - watch criterion
# - omega matrices always positive definite


def honorio_samaras(rho, covariances, n_samples, n_iter=10):
    """
    Parameters
    ==========
    rho: float
        regularization parameter

    covariances: list of numpy.ndarray
        covariance estimates. All shapes must be identical. Must be positive
        semidefinite.

    n_samples: list of int
        number of samples for each task. len(n_samples) == len(sigma)

    n_iter: int
        number of iteration

    Returns
    =======
    omega:
        estimated precision matrices
    """

    # Check that all covariances have the same size.
    n_samples = np.asarray(n_samples)

    # TODO: low robustness with inverse computation.
    covar = np.dstack(covariances)
    n_var = covar.shape[0]
    assert(covar.shape[0] == covar.shape[1])
    n_task = covar.shape[2]
    omega = np.ndarray(shape=covar.shape, dtype=np.float)
    for k in xrange(n_task):
        omega[..., k] = np.diag(1. / np.diag(covar[..., k]))

    y = np.ndarray(shape=(n_task, n_var - 1), dtype=np.float)
    ## y = [np.ndarray(shape=n_var - 1) for _ in xrange(n_task)]
    u = np.ndarray(shape=(n_task, n_var - 1))
    ## u = [np.ndarray(shape=n_var - 1) for _ in xrange(n_task)]
    y_1 = np.ndarray(shape=(n_task, n_var - 2))
    ## y_1 = [np.ndarray(shape=n_var - 2) for _ in xrange(n_task)]
    h_12 = np.ndarray(shape=(n_task, n_var - 2))
    ## h_12 = [np.ndarray(shape=n_var - 2) for _ in xrange(n_task)]
    q = np.ndarray(shape=(n_task,))
    c = np.ndarray(shape=(n_task,))

    for n in xrange(n_iter):
        print("Starting {iter_n:d}-th iteration...".format(iter_n=n))
        for p in xrange(n_var):

            if p == 0:
                # Initial state: remove first col/row
                W = omega[1:, 1:, :]   # stack of W(k)
                Winv = np.ndarray(shape=W.shape, dtype=np.float)
                for k in xrange(W.shape[2]):
                    # stack of W^-1(k)
                    Winv[..., k] = np.linalg.inv(W[..., k])
            else:
                # Update W and Winv (use update_submatrix() )
                for k in xrange(n_task):
                    update_submatrix(omega[..., k],
                                     W[..., k], Winv[..., k], p - 1)
                pass

            # In the following lines, implicit loop on k (tasks)
            # Extract y
            y[:, :p] = omega[:p, p, :].T
            y[:, p:] = omega[p + 1:, p, :].T
            # Extract u
            u[:, :p] = covar[:p, p, :].T
            u[:, p:] = covar[p + 1:, p, :].T

            for m in xrange(n_var - 1):
                # Compute  c
                # T(k) -> n_samples[k]
                # v(k) -> covar[p, p, k]
                # h_22(k) -> Winv[m, m, k]
                # h_12(k) -> Winv[:m, m, k],  Winv[m+1:, m, k]
                # y_1(k) -> y[k, :m], y[k, m+1:]
                # u_2(k) -> u[k, m]
                h_12[:, :m] = Winv[:m, m, :].T
                h_12[:, m:] = Winv[m + 1:, m, :].T
                y_1[:, :m] = y[:, :m]
                y_1[:, m:] = y[:, m + 1:]

                for k in xrange(n_task):
                    c[k] = - n_samples[k] * (
                        covar[p, p, k] * np.dot(h_12[k, :], y_1[k, :])
                        + u[k, m]
                        )

                c2 = np.sqrt((c * c).sum())

                # x -> y[:][m]
                if c2 <= rho:
                    y[:, m] = 0
                else:
                    # q(k) -> T(k) * v(k) * h_22(k)
                    # \lambda -> l   (lambda is a python keyword)
                    q = n_samples * covar[p, p, :] * Winv[m, m, :]
                     # x* = \lambda* diag(1 + \lambda q)^{-1} c
                    l = scipy.optimize.newton(
                        quadratic_trust_region, 0,
                        fprime=quadratic_trust_region_deriv,
                        args=(c, q, rho))
                    remainder = quadratic_trust_region(l, c, q, rho)
                    if abs(remainder) > 1e-4:
                        warnings.warn("Newton-Raphson step did not converge. "
                                      "remainder: %f" % remainder)
                    y[:, m] = (l * c) / (1. + l * q)

            # Copy back y in omega
            omega[:p, p, :] = y[:, :p].T
            omega[p + 1:, p, :] = y[:, p:].T

            for k in xrange(n_task):
                omega[p, p, k] = 1. / covar[p, p, k] + np.dot(
                    np.dot(y[k, :], Winv[..., k]), y[k, :])

            # Compute optimization criterion (for display)
            ll = 0  # log-likelihood
            for k in xrange(n_task):
                t = np.log(np.linalg.det(omega[..., k]))
                t -= (omega[..., k] * covar[..., k]).sum()
                ll += n_samples[k] * t

            # L(1,2)-norm
            l12 = np.sqrt((omega ** 2).sum(axis=-1)).sum()
            criterion = ll - rho * l12
            print("Criterion to maximize: {criterion:.3f}".format(
                    criterion=criterion))

    return omega


def quadratic_trust_region_criterion(l, c, q, rho):
    if l < 0:
        ret = ((c * c) / q).sum()  # value at zero. Just for display
    else:
        ret = ((c * c) / (q + l * q * q)).sum() + rho ** 2 * l
    return ret


def quadratic_trust_region(l, c, q, rho):
    if l < 0:
        slope = 2 * ((c * c) * q).sum()
        return rho ** 2 - (c * c).sum() + l * slope

    return rho ** 2 - ((c * c) / ((1. + l * q) ** 2)).sum()


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
    """Update submatrix and its inverse.

    sub_inv is the inverse of the submatrix of "full" obtained by removing
    the n-th row and column.

    sub_inv is modified in-place. After execution of this function, it contains
    the inverse of the submatrix of "full" obtained by removing the n+1-th row
    and column.

    This computation is based on Sherman-Woodbury-Morrison identity.
    """

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


def test_update_submatrix():
    N = 5
    rand_gen = np.random.RandomState(0)
    M = rand_gen.randn(N, N)
    sub = M[1:, 1:].copy()
    sub_inv = np.linalg.inv(M[1:, 1:])
    true_sub = np.zeros(sub.shape, dtype=sub.dtype)
    true_inv = sub_inv.copy()

    for n in xrange(1, N):
        update_submatrix(M, sub, sub_inv, n - 1)

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


def generate_multi_task_gg_model(n_task=5, n_var=10, density=0.2,
                                 min_eigenvalue=0.1,
                                 min_samples=30, max_samples=50,
                                 rand_gen=np.random.RandomState(0)):
    """Generate signals drawn from a sparse gaussian graphical models.

    Parameters
    ==========
    n_task: int
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
        with shape (sample number, n_var). len(tasks) == n_task

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

    # Generate edges weights
    covariances = []
    for _ in xrange(n_task):

        while True:
            weights = rand_gen.uniform(low=-.7, high=1., size=n_var * n_var
                                       ).reshape((n_var, n_var))
            covar = topology * weights
            covar += np.triu(covar, k=1).T
            covar += np.eye(*covar.shape)
            # Loop until a positive definite matrix is obtained.
            if np.linalg.eigvals(covar).min() > min_eigenvalue:
                covariances.append(covar)
                break

    for covar in covariances:
        np.testing.assert_almost_equal(covar, covar.T)

    # Returns a symmetric topology matrix
    topology = topology + np.triu(topology, k=1).T
    topology += np.eye(*topology.shape)

    assert(np.all(topology == topology.T))

    signals = []
    mean = np.zeros(topology.shape[0])
    n_samples = rand_gen.randint(min_samples, high=max_samples,
                                 size=len(covariances))

    for n, covar in zip(n_samples, covariances):
        signals.append(rand_gen.multivariate_normal(mean, covar, (n,)))

    return signals, covariances, topology


if __name__ == "__main__":
#    test_plot_quadratic_trust_region()
    test_quadratic_trust_region()
    sys.exit(0)

    display = False

    signals, covariances, topology = generate_multi_task_gg_model(density=0.3)
    covar_est = [np.dot(s.T, s) / len(s) for s in signals]

    if display:
        import pylab as pl

        for n, value in enumerate(zip(signals, covariances)):
            s, covar = value
            pl.figure()
            pl.imshow(covar_est[n], interpolation="nearest",
                      vmin=-1.2, vmax=1.2, cmap="gray")
            pl.colorbar()
            pl.title("{n:d}".format(n=n))

    rho = 0.01
    n_samples = [len(signal) for signal in signals]

    omega = honorio_samaras(rho, covar_est, n_samples, n_iter=3)

    print (omega.shape)

    if display:
        pl.show()
