"""Regression with spatial priors like TV-L1 and Graph-Net."""

# Author: DOHMATOB Elvis Dopgima,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Gaspar Pizarro,
#         Virgile Fritsch,
#         Bertrand Thirion,
#         and others.

from math import sqrt

import numpy as np

from nilearn.masking import unmask_from_to_3d_array

from ._objective_functions import (
    divergence,
    gradient,
    gradient_id,
    logistic_loss,
    logistic_loss_grad,
    logistic_loss_lipschitz_constant,
    spectral_norm_squared,
    squared_loss,
    squared_loss_grad,
)
from ._proximal_operators import (
    prox_l1,
    prox_l1_with_intercept,
    prox_tvl1,
    prox_tvl1_with_intercept,
)
from .fista import mfista


def _squared_loss_and_spatial_grad(X, y, w, mask, grad_weight):
    """Compute the squared loss (data fidelity term) + squared l2 norm \
    of gradient (penalty term).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    grad_weight : float
        l1_ratio * alpha.

    Returns
    -------
    float
        Value of Graph-Net objective.
    """
    data_section = np.dot(X, w) - y
    grad_buffer = np.zeros(mask.shape)
    grad_buffer[mask] = w
    grad_mask = np.tile(mask, [mask.ndim] + [1] * mask.ndim)
    grad_section = gradient(grad_buffer)[grad_mask]
    return 0.5 * (
        np.dot(data_section, data_section)
        + grad_weight * np.dot(grad_section, grad_section)
    )


def _squared_loss_and_spatial_grad_derivative(X, y, w, mask, grad_weight):
    """Compute the derivative of _squared_loss_and_spatial_grad.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    grad_weight : float
        l1_ratio * alpha

    Returns
    -------
    ndarray, shape (n_features,)
        Derivative of _squared_loss_and_spatial_grad function.
    """
    data_section = np.dot(X, w) - y
    image_buffer = np.zeros(mask.shape)
    image_buffer[mask] = w
    return (
        np.dot(X.T, data_section)
        - grad_weight * divergence(gradient(image_buffer))[mask]
    )


def _graph_net_data_function(X, w, mask, grad_weight):
    """Compute dot([X; grad_weight * grad], w).

    This function is made for the Lasso-like interpretation of the
    Graph-Net.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    grad_weight : float
        l1_ratio * alpha.

    Returns
    -------
    ndarray, shape (n_features + mask.ndim * n_samples,)
        Data-fit term augmented with design matrix augmented with
        nabla operator (for spatial gradient).
    """
    data_buffer = np.zeros(mask.shape)
    data_buffer[mask] = w
    w_g = grad_weight * gradient(data_buffer)
    out = np.ndarray(X.shape[0] + mask.ndim * X.shape[1])
    out[: X.shape[0]] = X.dot(w)
    out[X.shape[0] :] = np.concatenate(
        tuple(w_g[i][mask] for i in range(mask.ndim))
    )
    return out


def _graph_net_adjoint_data_function(X, w, adjoint_mask, grad_weight):
    """Compute the adjoint of the _graph_net_data_function.

    That is:

    np.dot([X.T; grad_weight * div], w).

    This function is made for the Lasso-like interpretation of the Graph-Net.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    grad_weight : float
        l1_ratio * alpha.

    Returns
    -------
    ndarray, shape (n_samples,)
        Value of adjoint.
    """
    n_samples, _ = X.shape
    out = X.T.dot(w[:n_samples])
    div_buffer = np.zeros(adjoint_mask.shape)
    div_buffer[adjoint_mask] = w[n_samples:]
    out -= grad_weight * divergence(div_buffer)[adjoint_mask[0]]
    return out


def _squared_loss_derivative_lipschitz_constant(
    X, mask, grad_weight, n_iterations=100
):
    """Compute the lipschitz constant of the gradient of the smooth part \
    of the Graph-Net regression problem (squared_loss + grad_weight*grad) \
    via power method.
    """
    rng = np.random.RandomState(42)
    a = rng.randn(X.shape[1])
    a /= sqrt(np.dot(a, a))
    adjoint_mask = np.tile(mask, [mask.ndim] + [1] * mask.ndim)

    # Since we are putting the coefficient into the matrix, which
    # is squared in the data loss function, it must be the
    # square root of the desired weight
    actual_grad_weight = sqrt(grad_weight)
    for _ in range(n_iterations):
        a = _graph_net_adjoint_data_function(
            X,
            _graph_net_data_function(X, a, mask, actual_grad_weight),
            adjoint_mask,
            actual_grad_weight,
        )
        a /= sqrt(np.dot(a, a))

    lipschitz_constant = np.dot(
        _graph_net_adjoint_data_function(
            X,
            _graph_net_data_function(X, a, mask, actual_grad_weight),
            adjoint_mask,
            actual_grad_weight,
        ),
        a,
    ) / np.dot(a, a)

    return lipschitz_constant


def _logistic_derivative_lipschitz_constant(
    X, mask, grad_weight, n_iterations=100
):
    """Compute the lipschitz constant of the gradient of the smooth part \
    of the Graph-Net classification problem (logistic_loss + \
    grad_weight*grad) via analytical formula on the logistic loss + \
    power method on the smooth part.
    """
    # L. constant for the data term (logistic)
    # data_constant = sp.linalg.norm(X, 2) ** 2
    data_constant = logistic_loss_lipschitz_constant(X)

    rng = np.random.RandomState(42)
    a = rng.randn(X.shape[1])
    a /= sqrt(np.dot(a, a))
    grad_buffer = np.zeros(mask.shape)
    for _ in range(n_iterations):
        grad_buffer[mask] = a
        a = -divergence(gradient(grad_buffer))[mask] / sqrt(np.dot(a, a))

    grad_buffer[mask] = a
    grad_constant = -np.dot(
        divergence(gradient(grad_buffer))[mask], a
    ) / np.dot(a, a)

    return data_constant + grad_weight * grad_constant


def _logistic_data_loss_and_spatial_grad(X, y, w, mask, grad_weight):
    """Compute the smooth part of the Graph-Net objective, \
    with logistic loss.
    """
    grad_buffer = np.zeros(mask.shape)
    grad_buffer[mask] = w[:-1]
    grad_mask = np.array([mask for _ in range(mask.ndim)])
    grad_section = gradient(grad_buffer)[grad_mask]
    return logistic_loss(X, y, w) + 0.5 * grad_weight * np.dot(
        grad_section, grad_section
    )


def _logistic_data_loss_and_spatial_grad_derivative(
    X, y, w, mask, grad_weight
):
    """Compute the derivative of _logistic_loss_and_spatial_grad."""
    image_buffer = np.zeros(mask.shape)
    image_buffer[mask] = w[:-1]
    data_section = logistic_loss_grad(X, y, w)
    data_section[:-1] = (
        data_section[:-1]
        - grad_weight * divergence(gradient(image_buffer))[mask]
    )
    return data_section


def graph_net_squared_loss(
    X,
    y,
    alpha,
    l1_ratio,
    mask,
    init=None,
    max_iter=1000,
    tol=1e-4,
    callback=None,
    lipschitz_constant=None,
    verbose=0,
):
    """Compute a solution for the Graph-Net regression problem.

    This function invokes the mfista backend (from fista.py) to solve the
    underlying optimization problem.

    Returns
    -------
    w : ndarray, shape (n_features,)
        Solution vector.

    solver_info : float
        Solver information, for warm start.

    objective : array of floats
        Objective function (fval) computed on every iteration.

    """
    _, n_features = X.shape

    # misc
    model_size = n_features
    l1_weight = alpha * l1_ratio
    grad_weight = alpha * (1.0 - l1_ratio)

    if lipschitz_constant is None:
        lipschitz_constant = _squared_loss_derivative_lipschitz_constant(
            X, mask, grad_weight
        )

        # it's always a good idea to use something a bit bigger
        lipschitz_constant *= 1.05

    # smooth part of energy, and gradient thereof
    def f1(w):
        return _squared_loss_and_spatial_grad(X, y, w, mask, grad_weight)

    def f1_grad(w):
        return _squared_loss_and_spatial_grad_derivative(
            X, y, w, mask, grad_weight
        )

    # prox of nonsmooth path of energy (account for the intercept)
    def f2(w):
        return np.sum(np.abs(w)) * l1_weight

    def f2_prox(w, step_size, *args, **kwargs):  # noqa: ARG001
        return prox_l1(w, step_size * l1_weight), {"converged": True}

    # total energy (smooth + nonsmooth)
    def total_energy(w):
        return f1(w) + f2(w)

    return mfista(
        f1_grad,
        f2_prox,
        total_energy,
        lipschitz_constant,
        model_size,
        dgap_factor=(0.1 + l1_ratio) ** 2,
        callback=callback,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        init=init,
    )


def graph_net_logistic(
    X,
    y,
    alpha,
    l1_ratio,
    mask,
    init=None,
    max_iter=1000,
    tol=1e-4,
    callback=None,
    verbose=0,
    lipschitz_constant=None,
):
    """Compute a solution for the Graph-Net classification problem, \
    with response vector in {-1, 1}^n_samples.

    This function invokes the mfista backend (from fista.py) to solve the
    underlying optimization problem.

    Returns
    -------
    w : ndarray of shape (n_features,)
       The solution vector (Where `n_features` is the size of the support
       of the mask.)

    solver_info : dict
        Solver information for warm starting. See fista.py.mfista(...)
        function for detailed documentation.

    objective : array of floats
        Cost function (fval) computed on every iteration.

    """
    _, n_features = X.shape

    # misc
    model_size = n_features + 1
    l1_weight = alpha * l1_ratio
    grad_weight = alpha * (1 - l1_ratio)

    if lipschitz_constant is None:
        lipschitz_constant = _logistic_derivative_lipschitz_constant(
            X, mask, grad_weight
        )

        # it's always a good idea to use somethx a bit bigger
        lipschitz_constant *= 1.1

    # smooth part of energy, and gradient of
    def f1(w):
        return _logistic_data_loss_and_spatial_grad(X, y, w, mask, grad_weight)

    def f1_grad(w):
        return _logistic_data_loss_and_spatial_grad_derivative(
            X, y, w, mask, grad_weight
        )

    # prox of nonsmooth path of energy (account for the intercept)
    def f2(w):
        return np.sum(np.abs(w[:-1])) * l1_weight

    def f2_prox(w, step_size, *args, **kwargs):  # noqa: ARG001
        return prox_l1_with_intercept(w, step_size * l1_weight), {
            "converged": True
        }

    # total energy (smooth + nonsmooth)
    def total_energy(w):
        return f1(w) + f2(w)

    # finally, run the solver proper
    return mfista(
        f1_grad,
        f2_prox,
        total_energy,
        lipschitz_constant,
        model_size,
        dgap_factor=(0.1 + l1_ratio) ** 2,
        callback=callback,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        init=init,
    )


def _tvl1_objective_from_gradient(gradient):
    """Compute TV-l1 objective function from gradient.

    Parameters
    ----------
    gradient : ndarray, shape (4, nx, ny, nz)
       precomputed "gradient + id" array

    Returns
    -------
    float
        Value of TV-L1 penalty.
    """
    tv_term = np.sum(np.sqrt(np.sum(gradient[:-1] * gradient[:-1], axis=0)))
    l1_term = np.abs(gradient[-1]).sum()
    return l1_term + tv_term


def _tvl1_objective(X, y, w, alpha, l1_ratio, mask, loss="mse"):
    """Compute the TV-L1 squared loss regression objective functions.

    Returns
    -------
    float
        Value of TV-L1 penalty.
    """
    loss = loss.lower()
    if loss not in ["mse", "logistic"]:
        raise ValueError(
            f"loss must be one of 'mse' or 'logistic'; got '{loss}'"
        )

    if loss == "mse":
        out = squared_loss(X, y, w)
    else:
        out = logistic_loss(X, y, w)
        w = w[:-1]

    grad_id = gradient_id(unmask_from_to_3d_array(w, mask), l1_ratio=l1_ratio)
    out += alpha * _tvl1_objective_from_gradient(grad_id)

    return out


def tvl1_solver(
    X,
    y,
    alpha,
    l1_ratio,
    mask,
    loss=None,
    max_iter=100,
    lipschitz_constant=None,
    init=None,
    prox_max_iter=5000,
    tol=1e-4,
    callback=None,
    verbose=1,
):
    """Minimizes empirical risk for TV-L1 penalized models.

    Can handle least squares (mean squared error --a.k.a mse) or logistic
    regression. The same solver works for both of these losses.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    alpha : float, default=1.0
        Constant that scales the overall regularization term.

    l1_ratio : float in the interval [0, 1]; default=0.5
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.

    mask : ndarray, shape (nx, ny, nz)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int, default=100
        Defines the iterations for the solver.

    prox_max_iter : int, default=5000
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    tol : float, default=1e-4
        Defines the tolerance for convergence.

    loss : string
        Loss model for regression. Can be "mse" (for squared loss) or
        "logistic" (for logistic loss).

    lipschitz_constant : float, optional (default None)
        Lipschitz constant (i.e an upper bound of) of gradient of smooth part
        of the energy being minimized. If no value is specified (None),
        then it will be calculated.

    callback : callable(dict) -> bool, optional (default None)
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Returns
    -------
    w : ndarray, shape (n_features,)
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    objective : array of floats
        Objective function (fval) computed on every iteration.

    solver_info : float
        Solver information, for warm start.

    """
    # sanitize loss
    if loss not in ["mse", "logistic"]:
        raise ValueError(
            f"'{loss}' loss not implemented. Should be 'mse' or 'logistic"
        )

    # shape of image box
    flat_mask = mask.ravel()
    volume_shape = mask.shape

    # in logistic regression, we fit the intercept explicitly
    w_size = X.shape[1] + int(loss == "logistic")

    def unmaskvec(w):
        if loss == "mse":
            return unmask_from_to_3d_array(w, mask)
        else:
            return np.append(unmask_from_to_3d_array(w[:-1], mask), w[-1])

    def maskvec(w):
        return (
            w[flat_mask]
            if loss == "mse"
            else np.append(w[:-1][flat_mask], w[-1])
        )

    # function to compute derivative of f1
    def f1_grad(w):
        if loss == "logistic":
            return logistic_loss_grad(X, y, w)
        else:
            return squared_loss_grad(X, y, w)

    # function to compute total energy (i.e smooth (f1) + nonsmooth (f2) parts)
    def total_energy(w):
        return _tvl1_objective(X, y, w, alpha, l1_ratio, mask, loss=loss)

    # Lipschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * spectral_norm_squared(X)
        else:
            lipschitz_constant = 1.1 * logistic_loss_lipschitz_constant(X)

    # proximal operator of nonsmooth proximable part of energy (f2)
    if loss == "mse":

        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = prox_tvl1(
                unmaskvec(w),
                weight=alpha * stepsize,
                l1_ratio=l1_ratio,
                dgap_tol=dgap_tol,
                init=unmaskvec(init),
                max_iter=prox_max_iter,
                verbose=verbose,
            )
            return maskvec(out.ravel()), info

    else:

        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = prox_tvl1_with_intercept(
                unmaskvec(w),
                volume_shape,
                l1_ratio,
                alpha * stepsize,
                dgap_tol,
                prox_max_iter,
                init=(
                    unmask_from_to_3d_array(init[:-1], mask)
                    if init is not None
                    else None
                ),
                verbose=verbose,
            )
            return maskvec(out.ravel()), info

    # invoke m-FISTA solver
    w, obj, init = mfista(
        f1_grad,
        f2_prox,
        total_energy,
        lipschitz_constant,
        w_size,
        dgap_factor=(0.1 + l1_ratio) ** 2,
        tol=tol,
        init=init,
        verbose=verbose,
        max_iter=max_iter,
        callback=callback,
    )

    return w, obj, init
