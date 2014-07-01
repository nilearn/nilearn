import os
import sys
import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal
from sklearn.utils import extmath
from sklearn.utils import check_random_state
from ..image import grad as gradient
from ..image import div as divergence
from ..smooth_lasso import (
    data_function,
    adjoint_data_function,
    squared_loss_and_spatial_grad,
    logistic_data_loss_and_spatial_grad,
    squared_loss_and_spatial_grad_derivative,
    logistic_data_loss_and_spatial_grad_derivative,
    smooth_lasso_squared_loss_objective,
    smooth_lasso_squared_loss_dual_objective,
    squared_loss_derivative_lipschitz_constant,
    logistic_derivative_lipschitz_constant,
    max_alpha_logistic,
    _pre_fit_labels,
    _debias,
    mfista)
from ..estimators import (SmoothLassoRegressor,
                          SmoothLassoClassifier)

# Data used in almost all tests
fn = lambda f, x, n: f(fn(f, x, n - 1)) if n > 1 else f(x)
ROOT = fn(os.path.dirname, os.path.dirname(__file__), 4)
CACHE = os.path.join(ROOT, "cache")
sys.path.append(os.path.join(ROOT, "examples/proximal"))
from .simulate_smooth_lasso_data import (
    create_smooth_simulation_data, create_simulation_data)
X, y, w, mask = create_smooth_simulation_data(
    snr=1., n_samples=10, size=4, n_points=5, random_state=42)
rng = check_random_state(42)


def get_gradient_matrix(w_size, mask):
    """
    Given a number of features and a mask (which has the property
    mask[mask==True].size == w_size) computes a matrix G such that for
    a w vector we have np.dot(G, w) == gradient(w_masked)[mask]
    """
    grad_matrix = np.zeros((mask.ndim * w_size, w_size))
    grad_mask = np.array([mask for _ in xrange(mask.ndim)])
    image_buffer = np.zeros(mask.shape)

    for i in xrange(w_size):
        base_vector = np.zeros(w_size)
        base_vector[i] = 1
        image_buffer[mask] = base_vector
        gradient_column = gradient(image_buffer)[grad_mask]
        grad_matrix[:, i] = gradient_column

    return grad_matrix


def test_grad_matrix():
    """Test for matricial form of gradient"""
    G = get_gradient_matrix(w.size, mask)
    image_buffer = np.zeros(mask.shape)
    grad_mask = np.array([mask for _ in xrange(mask.ndim)])
    for _ in xrange(10):
        v = rng.rand(w.size) * rng.randint(1000)
        image_buffer[mask] = v
        assert_almost_equal(gradient(image_buffer)[grad_mask], np.dot(G, v))


def test_adjointness(size=4):
    """Tests for adjointness between gradient and divergence operators"""
    for _ in xrange(3):
        image_1 = rng.rand(size, size, size)
        image_2 = rng.rand(3, size, size, size)
        Axdoty = np.dot((gradient(image_1).ravel()), image_2.ravel())
        xdotAty = np.dot((divergence(image_2).ravel()), image_1.ravel())
        assert_almost_equal(Axdoty, - xdotAty)


def test_identity_adjointness(size=4):
    """Tests adjointess between data_function and
    adjoint_data_function, with identity design matrix"""
    # A mask full of ones
    mask = np.ones((size, size, size), dtype=np.bool)
    # But with some zeros
    mask[0:3, 0:3, 0:3] = 0
    adjoint_mask = np.array([mask for _ in xrange(mask.ndim)])
    n_samples = np.sum(mask)
    X = np.eye(n_samples)
    l1_ratio = 0.5
    for _ in xrange(10):
        x = rng.rand(np.sum(mask))
        y = rng.rand(n_samples + np.sum(mask) * mask.ndim)
        Axdoty = np.dot(data_function(X, x, mask, l1_ratio), y)
        xdotAty = np.dot(adjoint_data_function(X, y, adjoint_mask, l1_ratio),
                         x)
        assert_almost_equal(Axdoty, xdotAty)


def test_operators_adjointness(size=4):
    """The same as test_identity_adjointness, but with generic design matrix"""
    # A mask full of ones
    mask = np.ones((size, size, size), dtype=np.bool)
    # But with some zeros
    mask[0:3, 0:3, 0:3] = 0
    adjoint_mask = np.array([mask for _ in xrange(mask.ndim)])
    n_samples = 200
    X = rng.rand(n_samples, np.sum(mask))
    l1_ratio = 0.5
    for _ in xrange(10):
        x = rng.rand(np.sum(mask))
        y = rng.rand(n_samples + np.sum(mask) * mask.ndim)
        Axdoty = np.dot(data_function(X, x, mask, l1_ratio), y)
        xdotAty = np.dot(adjoint_data_function(X, y, adjoint_mask, l1_ratio),
                         x)
        np.testing.assert_almost_equal(Axdoty, xdotAty)


def test_squared_loss_gradient_at_simple_points():
    """Tests gradient of data loss function in points near to zero. This is
    a not so hard test, just for detecting big errors"""
    X, y, w, mask = create_simulation_data(n_samples=10, size=4, roi_size=2)
    grad_weight = 1
    func = lambda w: squared_loss_and_spatial_grad(X, y, w, mask,
                                                   grad_weight)
    func_grad = lambda w: squared_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)
    for i in range(0, w.size, 2):
        point = np.zeros(*w.shape)
        point[i] = 1
        assert_almost_equal(sp.optimize.check_grad(func, func_grad, point),
                            0, decimal=3)


def test_logistic_gradient_at_simple_points():
    # Tests gradient of logistic data loss function in points near to zero.
    # This is a not so hard test, just for detecting big errors
    X, y, w, mask = create_simulation_data(n_samples=10, size=4, roi_size=2)
    grad_weight = 1
    # Add the intercept
    w = np.append(w, 0)
    func = lambda w: logistic_data_loss_and_spatial_grad(
        X, y, w, mask, grad_weight)
    func_grad = lambda w: logistic_data_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)
    for i in xrange(0, w.size, 7):
        point = np.zeros(*w.shape)
        point[i] = 1
        assert_almost_equal(sp.optimize.check_grad(func, func_grad, point),
                            0, decimal=3)


def test_squared_loss_derivative_lipschitz_constant():
    # Tests Lipschitz-continuity of the derivative of squared_loss loss
    # function
    grad_weight = 2.08e-1
    lipschitz_constant = squared_loss_derivative_lipschitz_constant(
        X, mask, grad_weight)
    for _ in xrange(20):
        x_1 = rng.rand(*w.shape) * rng.randint(1000)
        x_2 = rng.rand(*w.shape) * rng.randint(1000)
        gradient_difference = extmath.norm(
            squared_loss_and_spatial_grad_derivative(X, y, x_1, mask,
                                                     grad_weight)
            - squared_loss_and_spatial_grad_derivative(X, y, x_2, mask,
                                                       grad_weight))
        point_difference = extmath.norm(x_1 - x_2)
        assert gradient_difference <= lipschitz_constant * point_difference


def test_logistic_derivative_lipschitz_constant():
    # Tests Lipschitz-continuity of of the derivative of logistic loss
    grad_weight = 2.08e-1
    lipschitz_constant = logistic_derivative_lipschitz_constant(
        X, mask, grad_weight)
    for _ in xrange(20):
        x_1 = rng.rand((w.shape[0] + 1)) * rng.randint(1000)
        x_2 = rng.rand((w.shape[0] + 1)) * rng.randint(1000)
        gradient_difference = extmath.norm(
            logistic_data_loss_and_spatial_grad_derivative(
                X, y, x_1, mask, grad_weight)
            - logistic_data_loss_and_spatial_grad_derivative(
                X, y, x_2, mask, grad_weight))
        point_difference = extmath.norm(x_1 - x_2)
        assert gradient_difference <= lipschitz_constant * point_difference


def test_duality_gap():
    """Tests that the objective function never goes below
    the dual objective function, because the "primal" is
    a minimization problem"""
    alpha = 1
    l1_ratio = 0.5
    for _ in xrange(10):
        point = rng.rand(*w.shape) * rng.randint(100)
        primal_obj = smooth_lasso_squared_loss_objective(
            X, y, point, mask, alpha, l1_ratio)
        dual_obj = smooth_lasso_squared_loss_dual_objective(
            X, y, w, mask, alpha, l1_ratio)
        assert primal_obj > dual_obj


def test_fista_convergence():
    """Tests fista 1/k**2 convergence, theorem 4.4, "A Fast Iterative
    Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
    url:http://mechroom.technion.ac.il/~becka/papers/71654.pdf"""
    alpha = 1
    l1_ratio = 0.5
    reg = SmoothLassoRegressor(mask=mask, alpha=alpha, l1_ratio=l1_ratio)
    reg.fit(X, y)
    objs = reg.objective_
    # Since we don't have the optimum, we just aproximate the optimum
    # model and objetive with the last model the estimator computes
    optimum = objs[-1]
    model = reg.coef_
    l_c = squared_loss_derivative_lipschitz_constant(
        X, mask, alpha * (1 - l1_ratio) * y.size)
    # If you look at the paper, you take the norm of (starting point -
    # optimum), but, in this implementation (the test depend on the
    # implementation, this is awful, but simple) the initial point
    # is a zero vector
    for i in xrange(len(objs)):
        assert objs[i] - optimum <= 2 * l_c * np.dot(model, model)\
            / (i + 1) ** 2


def test_max_alpha_logistic():
    """Tests that models with l1 regularization over the theoretical bound
    are full of zeros, for logistic regression"""
    X, y, w, mask = create_smooth_simulation_data(task="classification")
    l1_ratios = np.linspace(0.1, 1, 3)
    clf = SmoothLassoClassifier(mask=mask, max_iter=10)
    for l1_ratio in l1_ratios:
        clf.l1_ratio = l1_ratio
        # We set alpha bigger than the theoretic bound
        clf.alpha = max_alpha_logistic(X, y, l1_ratio) * 1.1
        clf.fit(X, y)
        assert_almost_equal(clf.coef_, 0)


def test_max_alpha_squared_loss():
    """Tests that models with l1 regularization over the theoretical bound
    are full of zeros, for logistic regression"""
    l1_ratios = np.linspace(0.1, 1, 3)
    reg = SmoothLassoRegressor(mask=mask, max_iter=10)
    for l1_ratio in l1_ratios:
        reg.l1_ratio = l1_ratio
        reg.alpha = np.max(np.dot(X.T, y)) / (l1_ratio * y.size) * 1.1
        reg.fit(X, y)
        assert_almost_equal(reg.coef_, 0)


def test_tikhonov_regularization_vs_smooth_lasso():
    # Test for one of the extreme cases of Smooth Lasso: That is, with
    # l1_ratio = 0 (pure Smooth), we compare Smooth Lasso's performance
    # with the analytical solution for Tikhonov Regularization

    # XXX A small dataset here (this test is very lengthy)
    X, y, w, mask = create_smooth_simulation_data(
        snr=1., n_samples=50, size=8, n_points=4, random_state=42)
    G = get_gradient_matrix(w.size, mask)
    smooth_lasso = SmoothLassoRegressor(mask=mask, alpha=1, l1_ratio=0,
                                        max_iter=400, fit_intercept=False,
                                        normalize=False)
    smooth_lasso.fit(X, y)
    optimal_model = np.dot(sp.linalg.pinv(
        np.dot(X.T, X) + y.size * np.dot(G.T, G)), np.dot(X.T, y))
    smooth_lasso_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, smooth_lasso.coef_) - y) ** 2\
        + 0.5 * extmath.norm(np.dot(G, smooth_lasso.coef_)) ** 2
    optimal_model_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, optimal_model) - y) ** 2\
        + 0.5 * extmath.norm(np.dot(G, optimal_model)) ** 2
    assert_almost_equal(smooth_lasso_perf, optimal_model_perf, decimal=4)


def test_debiasing_model():
    """Tests that a debiased model has better performance that a no debiased
    one
    """
    X_train = X[:100]
    y_train = y[:100]
    X_test = X[100:]
    y_test = y[100:]
    smooth_lasso = SmoothLassoRegressor(mask=mask, alpha=0.01)
    smooth_lasso.fit(X_train, y_train)
    biased_score = smooth_lasso.score(X_test, y_test)
    smooth_lasso.coef_ = _debias(smooth_lasso.coef_, X_test, y_test)
    unbiased_score = smooth_lasso.score(X_test, y_test)
    assert unbiased_score >= biased_score


def test_pre_fit():
    # Mostly a smoke test
    sm_reg = SmoothLassoRegressor(mask=mask)
    sm_clf = SmoothLassoClassifier(mask=mask)
    assert not hasattr(sm_reg, "_enc")
    assert not hasattr(sm_clf, "_enc")
    _pre_fit_labels(sm_reg, y)
    _pre_fit_labels(sm_clf, y.astype(np.int))
    assert not hasattr(sm_reg, "_enc")
    assert hasattr(sm_clf, "_enc")


def test_mfista_solver_just_smooth():
    w = np.zeros(2)
    X = np.array([[1, 0], [0, 4]])
    y = np.array([-10, 20])
    f1 = lambda w: 0.5 * np.dot(np.dot(X, w) - y, np.dot(X, w) - y)
    f1_grad = lambda w: np.dot(X.T, np.dot(X, w) - y)
    f2_prox = lambda w, l, *args, **kwargs: (w, dict(converged=True))
    lipschitz_constant = squared_loss_derivative_lipschitz_constant(
        X, (np.eye(2) == 1).astype(np.bool), 1)
    estimate_solution, _, _ = mfista(
        f1, f1_grad, f2_prox, f1, lipschitz_constant, w.size, tol=1e-8)

    solution = np.array([-10, 5])
    assert_almost_equal(estimate_solution, solution, decimal=4)
