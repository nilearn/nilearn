"""Generic FISTA for solving TV-L1, Graph-Net, etc., problems.

For problems on which the prox of the nonsmooth term \
cannot be computed closed-form (e.g TV-L1), \
we approximate the prox using an inner FISTA loop.
"""

# Author: DOHMATOB Elvis Dopgima,
#         PIZARRO Gaspar,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,
#         THIRION Bertrand

from math import sqrt

import numpy as np
from scipy import linalg

from nilearn._utils import logger


def _check_lipschitz_continuous(
    f, ndim, lipschitz_constant, n_trials=10, random_state=42
):
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

    lispchitz_constant : float,
      Constant associated to the Lipschitz continuity.

    n_trials : int,
      Number of tests performed when assessing the Lipschitz continuity of
      function `f`. The more tests, the more confident we are in the
      Lipschitz continuity of `f` if the test passes.

    random_state : int, optional (default 42)
        Random state for initializing local rng.

    Raises
    ------
    RuntimeError
    """
    rng = np.random.default_rng(random_state)
    for x in rng.standard_normal((n_trials, ndim)):
        for y in rng.standard_normal((n_trials, ndim)):
            a = linalg.norm(f(x).ravel() - f(y).ravel(), 2)
            b = lipschitz_constant * linalg.norm(x - y, 2)
            if a > b:
                raise RuntimeError(f"Counter example: ({x}, {y})")


def mfista(
    f1_grad,
    f2_prox,
    total_energy,
    lipschitz_constant,
    w_size,
    dgap_tol=None,
    init=None,
    max_iter=1000,
    tol=1e-4,
    check_lipschitz=False,
    dgap_factor=None,
    callback=None,
    verbose=2,
):
    """Solve FISTA in a generic way.

    Minimizes the a sum `f + g` of two convex functions f (smooth)
    and g (proximable nonsmooth).

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

    check_lipschitz : boolean, default=False
        If True, check Lipschitz continuity of gradient of smooth part.

    w_size : int
        Size of the solution. f1, f2, f1_grad, f2_prox (fixed l, tol) must
        accept a w such that w.shape = (w_size,).

    tol : float, default=1e-4
        Tolerance on the (primal) cost function.

    dgap_tol : float, optional
        If None, the nonsmooth_prox argument returns a float, with the value,
        if not 0, the nonsmooth_prox accepts a third parameter tol, which is
        the tolerance on the computation of the proximal operator and returns a
        float, and a dict with the key "converged", that says if the method to
        compute f2_prox converged or not.

    dgap_factor : float, optional
        Dual gap factor. Used for debugging purpose (control the convergence).

    init : dict-like, optional
        Dictionary of initialization parameters. Possible keys are 'w',
        'stepsize', 'z', 't', 'dgap_factor', etc.

    callback : callable(dict) -> bool
        Function called on every iteration. If it returns True, then the loop
        breaks.

    max_iter : integer, default=1000
        Maximum number of iterations for the solver.

    verbose : integer, default=2
        Indicate the level of verbosity.

    Returns
    -------
    w : ndarray, shape (w_size,)
       A minimizer for `f + g`.

    solver_info : float
        Solver information, for warm starting.

    cost : array of floats
        Cost function (fval) computed on every iteration.

    Notes
    -----
    A motivation for the choice of FISTA as a solver for the TV-L1
    penalized problems emerged in the paper: Elvis Dohmatob,
    Alexandre Gramfort, Bertrand Thirion, Gael Varoquaux,
    "Benchmarking solvers for TV-L1 least-squares and logistic regression
    in brain imaging". Pattern Recognition in Neuroimaging (PRNI),
    Jun 2014, Tubingen, Germany. IEEE

    """
    # initialization
    if init is None:
        init = {}
    w = init.get("w", np.zeros(w_size))
    z = init.get("z", w.copy())
    t = init.get("t", 1.0)
    stepsize = init.get("stepsize", 1.0 / lipschitz_constant)
    if dgap_tol is None:
        dgap_tol = init.get("dgap_tol", np.inf)
    if dgap_factor is None:
        dgap_factor = init.get("dgap_factor", 1.0)

    # check Lipschitz continuity of gradient of smooth part
    if check_lipschitz:
        _check_lipschitz_continuous(f1_grad, w_size, lipschitz_constant)

    # aux variables
    old_energy = total_energy(w)
    energy_delta = np.inf
    best_w = w.copy()
    best_energy = old_energy
    best_dgap_tol = dgap_tol
    ista_step = False
    best_z = z.copy()
    best_t = t
    prox_info = {"converged": True}
    stepsize = 1.0 / lipschitz_constant
    history = []
    w_old = w.copy()

    # FISTA loop
    for i in range(max_iter):
        history.append(old_energy)
        w_old[:] = w

        # invoke callback
        logger.log(
            f"mFISTA: Iteration {i + 1: 2}/{max_iter:2}: "
            f"E = {old_energy:7.4e}, dE {energy_delta: 4.4e}",
            verbose,
        )
        if callback and callback(locals()):
            break
        if np.abs(energy_delta) < tol:
            logger.log(f"\tConverged (|dE| < {tol:g})", verbose)
            break

        # forward (gradient) step
        gradient_buffer = f1_grad(z)

        # backward (prox) step
        for _ in range(10):
            w, prox_info = f2_prox(
                z - stepsize * gradient_buffer,
                stepsize,
                dgap_factor * dgap_tol,
                init=w,
            )
            energy = total_energy(w)
            if (
                not ista_step
                or not prox_info["converged"]
                or old_energy > energy
            ):
                break

            # Even when doing ISTA steps we are not decreasing.
            # Thus we need a tighter dual_gap on the prox_tv
            # This corresponds to a line search on the dual_gap
            # tolerance.
            dgap_factor *= 0.2

            logger.log("decreased dgap_tol", verbose)
        # energy house-keeping
        energy_delta = old_energy - energy
        old_energy = energy

        # z update
        if energy_delta < 0.0:
            # M-FISTA strategy: rewind and switch temporarily to an ISTA step
            z[:] = w_old
            w[:] = w_old
            ista_step = True
            logger.log("Monotonous FISTA: Switching to ISTA", verbose)
        else:
            if ista_step:
                z = w
            else:
                t0 = t
                t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
                z = w + ((t0 - 1.0) / t) * (w - w_old)
            ista_step = False

        # misc
        if energy_delta != 0.0:
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
            dgap_tol = abs(energy_delta) / (i + 1.0)

        # dgap_tol house-keeping
        if energy < best_energy:
            best_energy = energy
            best_w[:] = w
            best_z[:] = z
            best_t = t
            best_dgap_tol = dgap_tol

    init = {
        "w": best_w.copy(),
        "z": best_z,
        "t": best_t,
        "dgap_tol": best_dgap_tol,
        "stepsize": stepsize,
    }
    return best_w, history, init
