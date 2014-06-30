"""
TV-l1 regression. Handles squared loss and logistic too.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Virgile Fritsch,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

import numpy as np
from .common import (compute_mse_lipschitz_constant, gradient_id,
                     compute_logistic_lipschitz_constant,
                     mse_loss, mse_loss_grad, _unmask,
                     logistic_grad as logistic_loss_grad,
                     logistic as logistic_loss)
from .operators import prox_tv_l1, intercepted_prox_tv_l1
from .fista import mfista


def _tvl1_objective_from_gradient(gradient):
    """Our total-variation like norm: TV + l1

    Parameters
    ----------
    gradient: array, of shape [3] + list(img_shape)
       precomputed "gradient + id" array

    """

    tv_term = np.sum(np.sqrt(np.sum(gradient[:-1] * gradient[:-1],
                                    axis=0)))
    l1_term = np.abs(gradient[-1]).sum()
    return l1_term + tv_term


def tvl1_objective(X, y, w, alpha, l1_ratio, mask=None, shape=None,
                   loss="mse"):
    """The TV + l1 squared loss regression objective functions,

        w can be a 2D or 3D array

    """

    if shape is None:
        if mask is not None:
            shape = mask.shape
        else:
            if loss == "mse":
                shape = w.shape
            else:
                shape = (len(np.ravel(w)) - 1,)

    # if not mask is None: mask = mask.ravel()
    loss = loss.lower()
    assert loss in ['mse', 'logistic']

    w = w.ravel()
    if loss == "mse":
        out = mse_loss(X, y, w, mask=mask)
    else:
        out = logistic_loss(X, y, w, mask=mask)
        w = w[:-1]

    grad_id = gradient_id(w.reshape(shape), l1_ratio=l1_ratio)
    out += alpha * _tvl1_objective_from_gradient(grad_id)

    return out


def tvl1_solver(X, y, alpha, l1_ratio, mask=None, loss=None,
                rescale_alpha=True, lipschitz_constant=None,
                prox_max_iter=5000, verbose=0, tol=1e-4, **kwargs):
    """Minimizes empirical risk for TV-l1 penalized models.

    Can handle least-squares (mean square error --a.k.a mse) or logistic
    regression. The same solver works for both of these losses.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Parameters
    ----------
    X: 2D array of shape (n_samples, n_features)
        Design matrix.

    y: 1D array of length n_samples
        Target / response vector.

    alpha: float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio: float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0: just smooth. l1_ratio == 1: just lasso.
        Defaults to 0.5.

    mask: multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter: int
        Defines the iterations for the solver. Defaults to 1000

    prox_max_iter: int, optional (default 10)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    tol: float
        Defines the tolerance for convergence. Defaults to 1e-4.

    loss: string
        Loss model for regression. Can be "mse" (for squared loss) or
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

    solver_info: float
        Solver information, for warm start.

    objective: array of floats
        Objective function (fval) computed on every iteration.


    """

    # sanitize loss
    if loss not in ["mse", "logistic"]:
        raise ValueError("'%s' loss not implemented. Should be 'mse' or "
                         "'logistic" % loss)

    # shape of image box
    if mask is not None:
        flat_mask = mask.ravel()
        volume_shape = mask.shape
    else:
        # when no mask is provided, the volume is assumed to be a flat image
        volume_shape = (X.shape[1],)
        flat_mask = None

    # XXX We'll work on the full brain, and do the masking / unmasking
    # magic when needed
    w_size = X.shape[1] + int(loss == "logistic")

    # rescale alpha parameter (= amount of regularization) to handle
    # 1 / n_samples factor in model
    if rescale_alpha:
        alpha *= X.shape[0]

    def unmaskvec(w):
        if None in [w, mask]:
            return w
        elif loss == "mse":
            return _unmask(w, mask)
        else:
            return np.append(_unmask(w[:-1], mask), w[-1])

    def maskvec(w):
        if None in [w, mask]:
            return w
        elif loss == "mse":
            return w[flat_mask]
        else:
            return np.append(w[:-1][flat_mask], w[-1])

    # fuction to compute f1 = smooth part of energy = the loss term
    def f1(w):
        if loss == "logistic":
            return logistic_loss(X, y, w)
        else:
            return mse_loss(X, y, w)

    # function to compute derivative of f1
    def f1_grad(w):
        if loss == "logistic":
            return logistic_loss_grad(X, y, w)
        else:
            return mse_loss_grad(X, y, w)

    # function to compute total energy (i.e smooth (f1) + nonsmooth (f2) parts)
    total_energy = lambda w: tvl1_objective(
        X, y, unmaskvec(w), alpha=alpha, l1_ratio=l1_ratio, mask=mask,
        loss=loss)

    # lispschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * compute_mse_lipschitz_constant(X)
        else:
            lipschitz_constant = 1.1 * compute_logistic_lipschitz_constant(X)

    # proximal operator of nonsmooth proximable part of energy (f2)
    if loss == "mse":
        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = prox_tv_l1(
                unmaskvec(w), weight=alpha * stepsize, l1_ratio=l1_ratio,
                dgap_tol=dgap_tol, return_info=True, init=unmaskvec(init),
                max_iter=prox_max_iter, verbose=verbose)
            return maskvec(out.ravel()), info
    else:
        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = intercepted_prox_tv_l1(
                unmaskvec(w), volume_shape, l1_ratio, alpha * stepsize,
                dgap_tol, prox_max_iter, init=_unmask(
                    init[:-1], mask) if init is not None else None,
                verbose=verbose)
            return maskvec(out.ravel()), info

    # invoke m-FISTA solver
    w, obj, init = mfista(
        f1, f1_grad, f2_prox, total_energy, lipschitz_constant, w_size,
        dgap_factor=(.1 + l1_ratio) ** 2, tol=tol, verbose=verbose, **kwargs)
 
    return w, obj, init
