"""
Smooth-Lasso

Module that implements a sklearn-like estimator for Smooth-Lasso regression

For the regression task, the data-dependent function to minimize is the mean
squared error: 1./(2 * n_samples)||Xw - y||^2
For the classification task, given labels in {-1, 1}^n_samples, the
data-dependent function to minimize is the logistic loss:
    - sum(log(sigmoid(y_i*w.T*X_i)))

Implements two estimators:
SmoothLassoRegressor
SmoothLassoClassifier

And their cross-validated variants:
SmoothLassoRegressorCV
SmoothLassoClassifierCV

General call graph:
SmoothLasso{Regressor|Classifier}CV.fit
 -> path_scores
     -> smooth_lasso_{squared|logistic}_loss
         -> mfista

SmoothLasso{Regressor|Classifier}.fit
 -> smooth_lasso_{squared|logistic}_loss
     -> mfista
"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

import warnings
import numpy as np
from ..._utils.fixes import is_classifier, LabelBinarizer
from .common import (logistic, logistic_grad,
                     compute_logistic_lipschitz_constant)
from .image import grad as gradient, div as divergence
from .operators import prox_l1
from .fista import mfista


def squared_loss_and_spatial_grad(X, y, w, mask, grad_weight):
    """
    Computes the differentiable loss function (squared_loss + g2_weight
    * gradient) for Smooth-Lasso regularization

    Parameters
    ----------
    X: design matrix
    y: sampled data
    w: model candidate
    mask: mask for w
    g2_weight: weight of the gradient operator
    """
    data_section = np.dot(X, w) - y
    grad_buffer = np.zeros(mask.shape)
    grad_buffer[mask] = w
    grad_mask = np.tile(mask, [mask.ndim] + [1] * mask.ndim)
    grad_section = gradient(grad_buffer)[grad_mask]
    return 0.5 * (np.dot(data_section, data_section)
                  + grad_weight * np.dot(grad_section, grad_section))


def squared_loss_and_spatial_grad_derivative(X, y, w, mask, grad_weight):
    """
    Computes the gradient of squared_loss_and_spatial_grad

    Parameters
    ----------
    X: design matrix
    y: sampled data
    w: model candidate
    mask: mask for w
    grad_weight: weight of the gradient operator
    """
    data_section = np.dot(X, w) - y
    image_buffer = np.zeros(mask.shape)
    image_buffer[mask] = w
    return (np.dot(X.T, data_section)
            - grad_weight * divergence(gradient(image_buffer))[mask])


def data_function(X, w, mask, grad_weight):
    """
    Computes [X;grad_weight*grad]w. This function is made for the Lasso-like
    interpretation of the Smooth Lasso

    Parameters
    ----------
    X: design matrix
    w: data vector
    mask: mask matrix. It has the "shape" of w
    grad_weight: weight of the gradient operator
    """
    data_buffer = np.zeros(mask.shape)
    data_buffer[mask] = w
    w_g = grad_weight * gradient(data_buffer)
    out = np.ndarray(X.shape[0] + mask.ndim * X.shape[1])
    out[:X.shape[0]] = X.dot(w)
    out[X.shape[0]:] = np.concatenate(
        tuple([w_g[i][mask] for i in xrange(mask.ndim)]))
    return out


def adjoint_data_function(X, w, adjoint_mask, grad_weight):
    """
    Computes the adjoint of the data_function, that is [X.T;grad_weight*div]w.
    This function is made for the Lasso-like interpretation of the Smooth Lasso

    Parameters
    ----------
    X: design matrix
    w: vector to evaluate to. It Has a n+{2|3}p shape
    adjoint_mask: mask of the data, with gradient shape, for example
        adjoint_mask.shape = (2, 512, 512)
    grad_weight: weight of the gradient operator
    """
    n_samples, _ = X.shape
    out = X.T.dot(w[:n_samples])
    div_buffer = np.zeros(adjoint_mask.shape)
    div_buffer[adjoint_mask] = w[n_samples:]
    out -= grad_weight * divergence(div_buffer)[adjoint_mask[0]]
    return out


def smooth_lasso_squared_loss_objective(X, y, w, mask, l1_weight, grad_weight):
    """
    Computes the full-blown risk function for SmoothLassoRegressor:
    squared_loss + gradient + l1_norm
    """
    return squared_loss_and_spatial_grad(X, y, w, mask, grad_weight)\
        + l1_weight * np.sum(np.abs(w))


def smooth_lasso_logistic_objective(X, y, w, mask, l1_weight, grad_weight):
    """
    Computes the full-blown risk function for SmoothLassoClassifier:
    squared_loss(X, y, w) + l1_weight * w grad_weight * spatial_gradient
    """
    penalty = np.sum(np.abs(w[:-1]))
    return (logistic_data_loss_and_spatial_grad(X, y, w, mask, grad_weight)
            + l1_weight * penalty)


def smooth_lasso_squared_loss_dual_objective(X, y, w, mask, l1_weight,
                                             grad_weight):
    """
    Compute dual gap for SmoothLassoRegressor model to check
    KKT optimality conditions

    Returns
    -------
    dual_objective: float
        the value of the objective function of the dual problem, that is:
        - 0.5 * np.dot(z, z) - np.dot(y, z), with

        z := alpha * l1_ratio * (Xw - y) / ||Xt(Xw-y)||_inf

    """
    n_samples, _ = X.shape
    # Since we are putting the coefficient into the X+G matrix, which
    # is squared in the data loss function, it must be the
    # square root of the desired weight
    actual_grad_weight = np.sqrt(grad_weight)

    dual_var = data_function(X, w, mask, actual_grad_weight)
    dual_var[:n_samples] -= y

    adjoint_mask = np.tile(mask, [mask.ndim] + [1] * mask.ndim)
    dual_norm = np.max(np.abs(adjoint_data_function(
        X, dual_var, adjoint_mask, actual_grad_weight)))
    const = np.min((l1_weight / dual_norm, 1))
    dual_var *= const

    dual_objective = (- 0.5 * np.dot(dual_var, dual_var)
                      - np.dot(y, dual_var[:n_samples]))
    return dual_objective


def squared_loss_derivative_lipschitz_constant(X, mask, grad_weight,
                                               n_iterations=100):
    """
    Computes the lipschitz constant of the gradient of the smooth part
    of the smooth lasso regression problem (squared_loss + grad_weight*grad)
    via power method

    XXX Do we really want to be using power method here?

    """
    a = np.random.randn(X.shape[1])
    a /= np.sqrt(np.dot(a, a))
    adjoint_mask = np.tile(mask, [mask.ndim] + [1] * mask.ndim)
    # Since we are putting the coefficient into the matrix, which
    # is squared in the data loss function, it must be the
    # square root of the desired weight
    actual_grad_weight = np.sqrt(grad_weight)
    for _ in range(n_iterations):
        a = adjoint_data_function(
            X, data_function(X, a, mask, actual_grad_weight),
            adjoint_mask, actual_grad_weight)
        a /= np.sqrt(np.dot(a, a))

    lipschitz_constant = np.dot(adjoint_data_function(
        X, data_function(X, a, mask, actual_grad_weight),
        adjoint_mask, actual_grad_weight), a) / np.dot(a, a)

    return lipschitz_constant


def logistic_derivative_lipschitz_constant(X, mask, grad_weight,
                                           n_iterations=100):
    """
    Computes the lipschitz constant of the gradient of the smooth part
    of the smooth lasso classification problem (logistic_loss +
    grad_weight*grad) via analytical formula on the logistic loss +
    power method on the smooth part
    """
    # L. constant for the data term (logistic)
    # data_constant = sp.linalg.norm(X, 2) ** 2
    data_constant = compute_logistic_lipschitz_constant(X)

    a = np.random.randn(X.shape[1])
    a /= np.sqrt(np.dot(a, a))
    grad_buffer = np.zeros(mask.shape)
    for _ in xrange(n_iterations):
        grad_buffer[mask] = a
        a = - divergence(gradient(grad_buffer))[mask] / np.sqrt(np.dot(a, a))

    grad_buffer[mask] = a
    grad_constant = (- np.dot(divergence(gradient(grad_buffer))[mask], a)
                     / np.dot(a, a))

    return data_constant + grad_weight * grad_constant


def intercepted_prox_l1(x, tau):
    """The same as prox_l1, but just for the n-1 components"""
    x[:-1] = prox_l1(x[:-1], tau)
    return x


def logistic_data_loss_and_spatial_grad(X, y, w, mask, grad_weight):
    """Compute the smooth part of the smooth lasso objective, with
    logistic loss"""
    grad_buffer = np.zeros(mask.shape)
    grad_buffer[mask] = w[:-1]
    grad_mask = np.array([mask for _ in xrange(mask.ndim)])
    grad_section = gradient(grad_buffer)[grad_mask]
    return (logistic(X, y, w)
            + 0.5 * grad_weight * np.dot(grad_section, grad_section))


def logistic_data_loss_and_spatial_grad_derivative(X, y, w, mask, grad_weight):
    """Compute the derivative of logistic_loss_and_spatial_grad"""
    image_buffer = np.zeros(mask.shape)
    image_buffer[mask] = w[:-1]
    data_section = logistic_grad(X, y, w)
    data_section[:-1] = data_section[:-1]\
        - grad_weight * divergence(gradient(image_buffer))[mask]
    return data_section


def max_alpha_logistic(X, y, l1_ratio):
    """
    Computes the theoretical upper bound for the overall
    regularization, as derived in "An Interior-Point Method for Large-Scale
    l1-Regularized Logistic Regression", by Koh, Kim, Boyd, in Journal of
    Machine Learning Research, 8:1519-1555, July 2007.
    url: http://www.stanford.edu/~boyd/papers/pdf/l1_logistic_reg.pdf

    XXX Dead code!

    """

    m = float(y.size)
    m_plus = float(y[y == 1].size)
    m_minus = float(y[y == -1].size)
    b = np.zeros(y.size)
    b[y == 1] = m_minus / m
    b[y == -1] = - m_plus / m
    return np.max(np.abs(X.T.dot(b))) / l1_ratio


def logistic_alpha_grid(X, y, l1_ratio, eps=1e-3, n_alphas=100):
    """Computes a grid of alphas, bounded for the value obtained
    with max_alpha_logistic
    XXX Dead code! """

    alpha_max = max_alpha_logistic(X, y, l1_ratio)
    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                         num=n_alphas)[::-1]
    return alphas


def _debias(w, X, y, copy=True):
    """Scales size of the model (size, not shape) to the data (X, y). If the
    model is just zeros, this method does nothing
    XXX Dead code! """

    if copy:
        w = w.copy()
    Xw = np.dot(X, w)
    if np.all(Xw != 0.0):
        w *= np.dot(y, Xw) / np.dot(Xw, Xw)
    return w


def _pre_fit_labels(model, y):
    """Sets the attributes about labels in a classifier. If model is a
    regressor, this method does nothing

    XXX Dead code! """
    if is_classifier(model):
        model._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        model.classes_ = model._enc.fit(y).classes_


def smooth_lasso_squared_loss(X, y, alpha, l1_ratio, mask=None, init=None,
                              max_iter=1000, tol=1e-4, callback=None,
                              lipschitz_constant=None, rescale_alpha=True,
                              verbose=0, backtracking=True):
    """Computes a solution for the Smooth Lasso regression problem, as in the
    SmoothLassoRegressor estimator, with no data preprocessing.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

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

    n_samples, n_features = X.shape

    # XXX smooth_lasso code breaks-down if mask is None!
    if mask is None:
        warnings.warn(
            "mask is None. Defaulting to full 1D mask of length = X.shape[1&]")
        mask = np.ones(n_features).astype(np.bool)

    # misc
    model_size = n_features

    # Instead of dividing the data term for the number of samples, we
    # rescale the overall regularization term (L1 + G2)
    # for the number of samples
    if rescale_alpha:
        alpha *= n_samples

    l1_weight = alpha * l1_ratio
    grad_weight = alpha * (1. - l1_ratio)

    if lipschitz_constant is None:
        lipschitz_constant = squared_loss_derivative_lipschitz_constant(
            X, mask, grad_weight)

        # it's always a good idea to use somethx a bit bigger
        # assert 0, X.sum()
        lipschitz_constant *= 1.05

    # smooth part of energy, and gradient of
    f1 = lambda w: squared_loss_and_spatial_grad(X, y, w, mask, grad_weight)
    f1_grad = lambda w: squared_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)

    # prox of nonsmooth path of energy (account for the intercept)
    f2 = lambda w: np.sum(np.abs(w)) * l1_weight
    f2_prox = lambda w, l, *args, **kwargs: (prox_l1(w, l * l1_weight),
                                             dict(converged=True))

    # total energy (smooth + nonsmooth)
    total_energy = lambda w: f1(w) + f2(w)

    return mfista(
        f1, f1_grad, f2_prox, total_energy, lipschitz_constant,
        model_size, dgap_factor=(.1 + l1_ratio) ** 2, callback=callback,
        tol=tol, max_iter=max_iter, verbose=verbose, init=init,
        backtracking=backtracking)


def smooth_lasso_logistic(X, y, alpha, l1_ratio, mask=None, init=None,
                          max_iter=1000, tol=1e-4, backtracking=True,
                          callback=None, verbose=0, lipschitz_constant=None,
                          rescale_alpha=True):
    """Computes a solution for the Smooth Lasso classification problem, with
    response vector in {-1, 1}^n_samples.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

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

    n_samples, n_features = X.shape

    # XXX smooth_lasso code breaks-down if mask is None!
    if mask is None:
        warnings.warn(
            "mask is None. Defaulting to full 1D mask of length = X.shape[1]")
        mask = np.ones(n_features).astype(np.bool)

    # misc
    model_size = n_features + 1

    # rescale alpha
    if rescale_alpha:
        alpha *= n_samples

    # Instead of dividing the data term for the number of samples, we
    # rescale the overall regularization term (L1 + G2)
    # for the number of samples
    l1_weight = alpha * l1_ratio
    grad_weight = alpha * (1 - l1_ratio)

    if lipschitz_constant is None:
        lipschitz_constant = logistic_derivative_lipschitz_constant(
            X, mask, grad_weight)

        # it's always a good idea to use somethx a bit bigger
        lipschitz_constant *= 1.1

    # smooth part of energy, and gradient of
    f1 = lambda w: logistic_data_loss_and_spatial_grad(
        X, y, w, mask, grad_weight)
    f1_grad = lambda w: logistic_data_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)

    # prox of nonsmooth path of energy (account for the intercept)
    f2 = lambda w: np.sum(np.abs(w[:-1])) * l1_weight
    f2_prox = lambda w, l, *args, **kwargs: (intercepted_prox_l1(
        w, l * l1_weight), dict(converged=True))

    # total energy (smooth + nonsmooth)
    total_energy = lambda w: f1(w) + f2(w)

    # finally, run the solver proper
    return mfista(
        f1, f1_grad, f2_prox, total_energy, lipschitz_constant,
        model_size, dgap_factor=(.1 + l1_ratio) ** 2, callback=callback,
        tol=tol, max_iter=max_iter, verbose=verbose, init=init,
        backtracking=backtracking)
