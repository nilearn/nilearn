"""
Regression with spatial priors like TV-l1 and Smooth LASSO.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Gaspar Pizarro,
#         Virgile Fritsch,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

import numpy as np
from sklearn.base import is_classifier
from .._utils.fixes.sklearn_basic_backports import LabelBinarizer
from .objective_functions import (squared_loss_lipschitz_constant,
                                  gradient_id,
                                  logistic_loss_lipschitz_constant,
                                  squared_loss, squared_loss_grad, _unmask,
                                  logistic_grad as logistic_loss_grad,
                                  logistic as logistic_loss)
from .objective_functions import gradient, div as divergence
from .proximal_operators import prox_l1, prox_tv_l1, intercepted_prox_tv_l1
from .fista import mfista


def squared_loss_and_spatial_grad(X, y, w, mask, grad_weight):
    """
    Computes the differentiable loss function (squared_loss + g2_weight
    * gradient) for Smooth-Lasso regularization

    Parameters
    ----------
    X : design matrix
    y : sampled data
    w : model candidate
    mask : mask for w
    g2_weight : weight of the gradient operator
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
    X : design matrix
    y : sampled data
    w : model candidate
    mask : mask for w
    grad_weight : weight of the gradient operator
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
    X : design matrix
    w : data vector
    mask : mask matrix. It has the "shape" of w
    grad_weight : weight of the gradient operator
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
    X : design matrix
    w : vector to evaluate to. It Has a n+{2|3}p shape
    adjoint_mask : mask of the data, with gradient shape, for example
        adjoint_mask.shape = (2, 512, 512)
    grad_weight : weight of the gradient operator
    """
    n_samples, _ = X.shape
    out = X.T.dot(w[:n_samples])
    div_buffer = np.zeros(adjoint_mask.shape)
    div_buffer[adjoint_mask] = w[n_samples:]
    out -= grad_weight * divergence(div_buffer)[adjoint_mask[0]]
    return out


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
    data_constant = logistic_loss_lipschitz_constant(X)

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
    return (logistic_loss(X, y, w)
            + 0.5 * grad_weight * np.dot(grad_section, grad_section))


def logistic_data_loss_and_spatial_grad_derivative(X, y, w, mask, grad_weight):
    """Compute the derivative of logistic_loss_and_spatial_grad"""
    image_buffer = np.zeros(mask.shape)
    image_buffer[mask] = w[:-1]
    data_section = logistic_loss_grad(X, y, w)
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
                              verbose=0):
    """Computes a solution for the Smooth Lasso regression problem, as in the
    SmoothLassoRegressor estimator, with no data preprocessing.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Returns
    -------
    w : np.array of size w_size
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    solver_info : float
        Solver information, for warm start.

    objective : array of floats
        Objective function (fval) computed on every iteration.

    """

    n_samples, n_features = X.shape

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
        tol=tol, max_iter=max_iter, verbose=verbose, init=init)


def smooth_lasso_logistic(X, y, alpha, l1_ratio, mask, init=None,
                          max_iter=1000, tol=1e-4, callback=None, verbose=0,
                          lipschitz_constant=None, rescale_alpha=True):
    """Computes a solution for the Smooth Lasso classification problem, with
    response vector in {-1, 1}^n_samples.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Returns
    -------
    w : np.array of size w_size
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    solver_info : float
        Solver information, for warm start.

    objective : array of floats
        Objective function (fval) computed on every iteration.

    """

    n_samples, n_features = X.shape

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
        tol=tol, max_iter=max_iter, verbose=verbose, init=init)


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


def tvl1_objective(X, y, w, alpha, l1_ratio, mask, loss="mse"):
    """The TV + l1 squared loss regression objective functions.

    """

    loss = loss.lower()
    if loss not in ['mse', 'logistic']:
        raise ValueError(
            "loss must be one of 'mse' or 'logistic'; got '%s'" % loss)

    if loss == "mse":
        out = squared_loss(X, y, w)
    else:
        out = logistic_loss(X, y, w)
        w = w[:-1]

    grad_id = gradient_id(_unmask(w, mask), l1_ratio=l1_ratio)
    out += alpha * _tvl1_objective_from_gradient(grad_id)

    return out


def tvl1_solver(X, y, alpha, l1_ratio, mask, loss=None, max_iter=100,
                rescale_alpha=True, lipschitz_constant=None, init=None,
                prox_max_iter=5000, tol=1e-4, callback=None, verbose=1):
    """Minimizes empirical risk for TV-l1 penalized models.

    Can handle least-squares (mean square error --a.k.a mse) or logistic
    regression. The same solver works for both of these losses.

    This function invokes the mfista backend (from fista.py) to solver the
    underlying optimization problem.

    Parameters
    ----------
    X : 2D array of shape (n_samples, n_features)
        Design matrix.

    y : 1D array of length n_samples
        Target / response vector.

    alpha : float
        Constant that scales the overall regularization term. Defaults to 1.0.

    l1_ratio : float in the interval [0, 1]; optinal (default .5)
        Constant that mixes L1 and TV penalization.
        l1_ratio == 0 : just smooth. l1_ratio == 1 : just lasso.
        Defaults to 0.5.

    mask : multidimensional array of booleans, optional (default None)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int
        Defines the iterations for the solver. Defaults to 1000

    prox_max_iter : int, optional (default 10)
        Maximum number of iterations for inner FISTA loop in which
        the prox of TV is approximated.

    tol : float
        Defines the tolerance for convergence. Defaults to 1e-4.

    loss : string
        Loss model for regression. Can be "mse" (for squared loss) or
        "logistic" (for logistic loss).

    lipschitz_constant : float, optional (default None)
        Lipschitz constant (i.e an upper bound of) of gradient of smooth part
        of the energy being minimized. If no value is specified (None),
        then it will be calculated.

    callback : callable(dict) -> bool
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Returns
    -------
    w : np.array of size w_size
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    objective : array of floats
        Objective function (fval) computed on every iteration.

    solver_info: float
        Solver information, for warm start.

    """

    # sanitize loss
    if loss not in ["mse", "logistic"]:
        raise ValueError("'%s' loss not implemented. Should be 'mse' or "
                         "'logistic" % loss)

    # shape of image box
    flat_mask = mask.ravel()
    volume_shape = mask.shape

    # XXX We'll work on the full brain, and do the masking / unmasking
    # magic when needed
    w_size = X.shape[1] + int(loss == "logistic")

    # rescale alpha parameter (= amount of regularization) to handle
    # 1 / n_samples factor in model
    if rescale_alpha:
        alpha *= X.shape[0]

    def unmaskvec(w):
        if loss == "mse":
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
            return squared_loss(X, y, w)

    # function to compute derivative of f1
    def f1_grad(w):
        if loss == "logistic":
            return logistic_loss_grad(X, y, w)
        else:
            return squared_loss_grad(X, y, w)

    # function to compute total energy (i.e smooth (f1) + nonsmooth (f2) parts)
    total_energy = lambda w: tvl1_objective(
        X, y, w, alpha, l1_ratio, mask, loss=loss)

    # lispschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * squared_loss_lipschitz_constant(X)
        else:
            lipschitz_constant = 1.1 * logistic_loss_lipschitz_constant(X)

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
        dgap_factor=(.1 + l1_ratio) ** 2, tol=tol, init=init, verbose=verbose,
        max_iter=max_iter, callback=callback)
 
    return w, obj, init
