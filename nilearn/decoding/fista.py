"""
Generic FISTA for solving TV-l1, S-Lasso, etc., problems.
For problems on which the prox of the nonsmooth term cannot be computed
closed-form (e.g TV-l1), approximate it using a inner FISTA loop.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from math import sqrt
import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state


def check_lipschitz_continuous(f, ndim, lipschitz_constant, n_trials=10,
                               err_msg=None):
    """Empirically check Lipschitz continuity of a function.

    If this test is passed, then we are empirically confident in the
    Lipschitz continuity of the function with respect to the given
    constant `L`. This confidence increases with the `n_trials` parameter.

    Parameters
    ----------
    f : callable,
      The function to be checked for Lipschitz continuity.
      `f` takes a vector of float as unique argument.
      The size of the input vector is determined by `ndim`.

    ndim : int,
      Dimension of the input of the function to be checked for Lipschitz
      continuity (i.e. it corresponds to the size of the vector that `f`
      takes as an argument).

    lischitz_constant : float,
      Constant associated to the Lipschitz continuity.

    n_trials : int,
      Number of tests performed when assessing the Lipschitz continuity of
      function `f`. The more tests, the more confident we are in the
      Lipschitz continuity of `f` if the test passes.

    err_msg : {str, or None},
      String used to tune the output message when the test fails.
      If `None`, we'll generate our own.

    Raises
    ------
    AssertionError

    """

    rng = check_random_state(42)

    for x in rng.randn(n_trials, ndim):
        for y in rng.randn(n_trials, ndim):
            err_msg = "LC counter example: (%s, %s)" % (
                x, y) if err_msg is None else err_msg
            a = linalg.norm(f(x).ravel() - f(y).ravel(), 2)
            b = lipschitz_constant * linalg.norm(x - y, 2)
            assert a <= b, err_msg + ("(a = %g >= %g)" % (a, b))


def mfista(f1_grad, f2_prox, total_energy, lipschitz_constant, w_size,
           dgap_tol=None, init=None, max_iter=1000, tol=1e-4,
           check_lipschitz=False, dgap_factor=None, callback=None,
           verbose=2):
    """

    Parameters
    ----------
    f1_grad : callable(w) -> np.array
        Gradient of smooth part of energy

    f2_prox : callable(w, stepsize, dgap_tol, init?) -> float, dict
        Proximal operator of non-smooth part of energy (f2).
        The returned dict should have a key "converged", whose value
        indicates whether the prox computation converged.

    total_energy : callable(w) -> float
        total energy (i.e smooth (f1) + nonsmooth (f2) parts)

    lipschitz_constant : float
        Lipschitz constant of gradient of f1_grad.

    w_size : int
        Size of the solution. f1, f2, f1_grad, f2_prox (fixed l, tol) must
        accept a w such that w.size = w_size.

    tol : float
        Tolerance on the primal energy functional.

    dgap_tol : float
        If None, the nonsmooth_prox argument returns a float, with the value,
        if not 0, the nonsmooth_prox accepts a third parameter tol, which is
        the tolerance on the computation of the proximal operator and returns a
        float, and a dict with the key "converged", that says if the method to
        compute f2_prox converged or not.

    callback : callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    max_iter : int
        Maximum number of iterations for the solver.

    pure_ista : bool, optional (default False)
        if set, then solver defaults to a pure ISTA algorithm

    Returns
    -------
    w : np.array of size w_size
       The Solution.

    solver_info : float
        Solver information, for warm start.

    objective : array of floats
        Objective function (fval) computed on every iteration.

    """

    # initializations
    if init is None:
        init = dict()
    w = init.get('w', np.zeros(w_size))
    z = init.get("z", w.copy())
    t = init.get("t", 1.)
    stepsize = init.get("stepsize", 1. / lipschitz_constant)
    if dgap_tol is None:
        dgap_tol = init.get('dgap_tol', np.inf)
    if dgap_factor is None:
        dgap_factor = init.get("dgap_factor", 1.)

    # check lipschitz continuity of gradient of smooth part
    if check_lipschitz:
        check_lipschitz_continuous(f1_grad, w_size, lipschitz_constant)

    # aux variables
    old_energy = total_energy(w)
    energy_delta = np.inf
    best_w = w.copy()
    best_energy = old_energy
    best_dgap_tol = dgap_tol
    ista_step = False
    best_z = z.copy()
    best_t = t
    best_dgap_tol = dgap_tol
    prox_info = dict(converged=True)
    stepsize = 1. / lipschitz_constant
    objective = []

    # FISTA loop
    for i in xrange(int(max_iter)):
        objective.append(old_energy)
        w_old = w.copy()

        # Invoke callback.
        if verbose:
            print 'mFISTA: Iteration % 2i/%2i: E = %7.4e, dE % 4.4e' % (
                i + 1, max_iter, old_energy, energy_delta)
        if callback and callback(locals()):
            break
        if np.abs(energy_delta) < tol:
            if verbose:
                print "\tConverged (|dE| < %g)" % tol
            break

        # Forward (gradient) step
        gradient_buffer = f1_grad(z)

        # Backward (prox) step
        for _ in range(10):
            w, prox_info = f2_prox(z - stepsize * gradient_buffer, stepsize,
                                   dgap_factor * dgap_tol, init=w)
            w = w.ravel()
            energy = total_energy(w)

            if ista_step and prox_info['converged'] and old_energy <= energy:
                # Even when doing ISTA steps we are not decreasing.
                # Thus we need a tighter dual_gap on the prox_tv
                # This corresponds to a line search on the dual_gap
                # tolerance.
                dgap_factor *= .2
                if verbose:
                    print "decreased dgap_tol"
            else:
                break

        # energy house-keeping
        energy_delta = old_energy - energy
        old_energy = energy

        # z update
        if energy_delta < 0.:
            # M-FISTA strategy: rewind and switch temporarily to an ISTA step
            z = w_old.copy()
            w = w_old.copy()
            ista_step = True
            if verbose:
                print 'Monotonous FISTA: Switching to ISTA'
        else:
            if ista_step:
                z = w
            else:
                t0 = t
                t = 0.5 * (1. + sqrt(1. + 4. * t * t))
                z = w + ((t0 - 1.) / t) * (w - w_old)
            ista_step = False

        # miscellaneous
        if energy_delta != 0.:
            # We need to decrease the tolerance on the dual_gap as 1/i**4
            # (see Mark Schmidt, Nicolas le Roux and Francis Bach, NIPS
            # 2011), thus we need to count how many times we are called,
            # hence the callable class. In practice, empirically I (Gael)
            # have found that such a sharp decrease was counter
            # productive in terms of computation time, as it leads to too
            # much time spent in the prox_tvl1 calls.
            #
            # For this reason, we rely more on the linesearch-like
            # strategy to set the dgap_tol
            dgap_tol = abs(energy_delta) / (i + 1.)

        # dgap_tol house-keeping
        if energy < best_energy:
            best_energy = energy
            best_w = w.copy()
            best_z = z.copy()
            best_t = t
            best_dgap_tol = dgap_tol
            best_dgap_tol = dgap_tol

    return (best_w,
            objective,
            dict(w=best_w.copy(), z=best_z, t=best_t, dgap_tol=best_dgap_tol,
                 stepsize=stepsize))
