"""
FISTA for solving TV-l1, S-Lasso, etc., problems.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from math import sqrt
from functools import partial
import numpy as np
from common import check_lipschitz_continuous


def mfista(f1, f1_grad, f2_prox, total_energy, lipschitz_constant,
           w_size, dgap_tol=None, init=None, backtracking=False,
           max_iter=1000, tol=1e-4, pure_ista=False, callback=None,
           verbose=2, check_lipschitz=False, check_monotonous=False,
           dgap_factor=None):
    """

    Parameters
    ----------
    f1: callable(w) -> float
         Gmooth part of energy (= the loss term).

    f1_grad: callable(w) -> np.array
        Gradient of smooth part of energy

    f2_prox: callable(w, stepsize, dgap_tol, init?) -> float, dict
        Proximal operator of non-smooth part of energy (f2).
        The returned dict should have a key "converged", whose value
        indicates whether the prox computation converged.

    total_energy: callable(w) -> float
        total energy (i.e smooth (f1) + nonsmooth (f2) parts)

    lipschitz_constant: float
        Lipschitz constant of gradient of f1_grad.

    w_size: int
        Size of the solution. f1, f2, f1_grad, f2_prox (fixed l, tol) must
        accept a w such that w.size = w_size

    backtracking: bool
        If True, the solver does backtracking in the step size for the proximal
        operator

    tol: float
        Tolerance on the variation of the objective function before breaking.
        The meaning of tol can be manipulated with the callback function

    dgap_tol: float
        If None, the nonsmooth_prox argument returns a float, with the value,
        if not 0, the nonsmooth_prox accepts a third parameter tol, which is
        the tolerance on the computation of the proximal operator and returns a
        float, and a dict with the key "converged", that says if the method to
        compute f2_prox converged or not.

    callback: callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    max_iter: int
        Maximum number of iterations for the solver.

    pure_ista: bool, optional (default False)
        if set, then solver defaults to a pure ISTA algorithm

    Returns
    -------
    w: np.array of size w_size
       The Solution.

    solver_info: float
        Solver information, for warm start.

    objective: array of floats
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
    gradient_buffer = np.zeros(w.shape, dtype=w.dtype)
    old_energy = total_energy(w)
    energy_delta = np.inf
    best_w = w.copy()
    best_energy = old_energy
    best_dgap_tol = dgap_tol
    ista_step = pure_ista
    best_z = z.copy()
    best_t = t
    best_dgap_tol = dgap_tol
    prox_info = dict(converged=True)
    objective = []

    # FISTA loop
    rho = .5
    name = "%s%s" % ("mISTA" if pure_ista else "mFISTA",
                     "-BT" if backtracking else "")
    for i in xrange(int(max_iter)):
        objective.append(old_energy)
        w_old = w.copy()

        if verbose:
            print '%s: Iteration % 2i/%2i: E = %7.4e, dE % 4.4e' % (
                name, i + 1, max_iter, old_energy, energy_delta)

        if callback and callback(locals()):
            break

        if np.abs(energy_delta) < tol:
            if verbose:
                print "\tConverged (|dE| < %g)" % tol
            break

        # The gradient of the smooth function is computed only in the
        # mask
        gradient_buffer = f1_grad(z)

        # Backtracking: Find largest stepsize (s) verifying the inequality:
        #    g(p_s) <= f(z) + <p_s - z, grad_f(z)> + 1 / (2s) ||p_s - z||^2
        #    , where p_s := prox_s(z - s * grad_f(z))
        if backtracking:
            if verbose:
                print "%s: Starting backtracking loop..." % name
        else:
            stepsize = 1. / lipschitz_constant
        while True:
            if backtracking and verbose:
                print "\tTrying stepsize %g..." % stepsize

            # This loop is a line-search like strategy on the dual gap of
            # the prox
            for _ in range(10):
                w, prox_info = f2_prox(z - stepsize * gradient_buffer,
                                       stepsize, dgap_factor * dgap_tol,
                                       init=w)
                w = w.ravel()
                energy = total_energy(w)

                if ista_step and prox_info[
                        'converged'] and old_energy <= energy:
                    # Even when doing ISTA steps we are not decreasing.
                    # Thus we need a tighter dual_gap on the prox_tv
                    # This corresponds to a line search on the dual_gap
                    # tolerance.
                    dgap_factor *= .2
                    if verbose:
                        print "decreased dgap_tol"
                else:
                    break

            # if backtracking disabled then we're done here
            if not backtracking:
                break

            # check if current stepsize does the job
            aux = w - z
            F = f1(w)
            Q = f1(z) + np.dot(aux, gradient_buffer) + (
                .5 / stepsize) * np.dot(aux, aux)
            if F <= Q:
                if verbose:
                    print (
                        "\tHurray! Last stepsize of %g worked like a "
                        "charm!" % stepsize)
                break

            # OK, decrease stepsize
            stepsize *= rho

        # ISTA is provably monotonous
        if i > 0 and check_monotonous:
            assert energy <= 1.1 * old_energy, (
                "Oops! old_energy = %g < energy = %g. This is "
                "unacceptable since ISTA and mFISTA are provably monotonous."
                " The must be a bug in the code. For example, maybe "
                "you are assuming a wrong lipschitz constant or you "
                "have bugs in the way you compute the gradient of"
                " the smooth part of the energy.") % (old_energy, energy)

        # energy house-keeping
        energy_delta = old_energy - energy
        old_energy = energy

        # z update
        # if (not ista_step) and energy_delta <= 0 and (not pure_ista):
        if energy_delta < 0.:
            # M-FISTA strategy: rewind and switch temporarily to an ISTA step
            z = w_old.copy()
            w = w_old.copy()
            ista_step = True
            if verbose:
                print 'Monotonous FISTA: Switching to ISTA'
        else:
            if ista_step or pure_ista:
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
            # thus the callable class. In practice, empirically I (Gael)
            # have found that such a sharp decrease was counter
            # productive in terms of computation time, as it leads to too
            # much time spent in the prox_tv_l1 calls.
            #
            # For this reason, we rely more on the line-search like
            # strategy to set the dgap_tol
            dgap_tol = abs(energy_delta) / (i + 1.)

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

# pure ISTA
ista = partial(mfista, pure_ista=True)
