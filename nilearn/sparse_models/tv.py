"""
Synopsis: TV-l1 regression. Handles squared loss and logistic too.
Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np
from .common import (compute_mse_lipschitz_constant,
                     compute_logistic_lipschitz_constant,
                     tv_l1_reg_objective, mse_loss, mse_loss_grad,
                     logistic_grad as logistic_loss_grad,
                     logistic as logistic_loss)
from .prox_tv_l1 import prox_tv_l1, intercepted_prox_tv_l1
from .fista import mfista


def tvl1_solver(X, y, alpha, l1_ratio, mask=None, loss=None,
                rescale_alpha=True, lipschitz_constant=None,
                prox_max_iter=5000, verbose=0, tol=1e-4, **kwargs):
    """Minimizes empirical risk for TV-l1 penalized least-squares (
    mean square error --a.k.a mse) or logisitc regression. The same solver
    works for both of this losses.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Parameters
    ----------
    alpha : float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    prox_max_iter: int, optional (default 5000)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    loss: string
        Loss model for regression. Cab be "mse" (for squared loss) or
        "logistic" (for logistic loss).

    lipschitz_constant: float, optional (default None)
        Lipschitz constant (i.e an upper bound of) of gradient of smooth part
        of the energy being minimized. If no value is specified (None),
        then it will be calculated.

    Returns
    -------
    w: np.array of size w_size
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    solver_info : float
        Solver information, for warm start.

    objective : array of floats
        Objective function (fval) computed on every iteration.


    """

    # sanitize loss
    if loss is None or loss not in ["mse", "logistic"]:
        raise ValueError("'%s' loss not implemented. Should be 'mse' or "
                         "'logistic" % loss)

    # shape of image box
    if mask is not None:
        volume_shape = mask.shape
    else:
        # when no mask is provided, the volume is assumed to be a flat image
        volume_shape = (X.shape[1],)

    # XXX We'll work on the full brain, and do the masking / unmasking
    # magic when needed
    w_size = np.prod(volume_shape) + int(loss == "logistic")

    # rescale alpha parameter (= amount of regularization) to handle
    # 1 / n_samples factor in model
    if rescale_alpha:
        alpha *= X.shape[0]

    # fuction to compute f1 = smooth part of energy = the loss term
    def f1(w):
        if loss == "logistic":
            return logistic_loss(X, y, w, mask=mask)
        else:
            return mse_loss(X, y, w, mask=mask)

    # function to compute derivative of f1
    def f1_grad(w):
        if loss == "logistic":
            return logistic_loss_grad(X, y, w, mask=mask)
        else:
            return mse_loss_grad(X, y, w, mask=mask)

    # function to compute total energy (i.e smooth (f1) + nonsmooth (f2) parts)
    total_energy = lambda w: tv_l1_reg_objective(
        X, y, w, alpha=alpha, l1_ratio=l1_ratio, mask=mask, loss=loss)

    # lispschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * compute_mse_lipschitz_constant(X)
        else:
            lipschitz_constant = 1.1 * compute_logistic_lipschitz_constant(X)

    # proximal operator of nonsmooth proximable part of energy (f2)
    if loss == "mse":
        f2_prox = lambda w, stepsize, dgap_tol, init=None: prox_tv_l1(
            w.reshape(volume_shape), weight=alpha * stepsize,
            l1_ratio=l1_ratio, dgap_tol=dgap_tol, return_info=True,
            init=init.reshape(volume_shape) if init is not None else init,
            max_iter=prox_max_iter, verbose=False)
    else:
        f2_prox = lambda w, stepsize, dgap_tol, init=None: (
            intercepted_prox_tv_l1(
                w, volume_shape, l1_ratio, alpha * stepsize, dgap_tol,
                prox_max_iter, init=init[:-1] if init is not None else None))

    # invoke m-FISTA solver
    w, obj, init = mfista(
        f1, f1_grad, f2_prox, total_energy, lipschitz_constant, w_size,
        dgap_factor=(.1 + l1_ratio) ** 2, tol=tol, verbose=verbose, **kwargs)
    # assert mask is not None
    if mask is not None:
        if loss == "mse":
            w = w[mask.ravel()]
        else:
            w = np.append(w[:-1][mask.ravel()], w[-1])

    return w, obj, init
