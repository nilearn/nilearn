from nose.tools import assert_true
import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal
from sklearn.utils import extmath
from sklearn.utils import check_random_state
from nilearn.decoding.objective_functions import _gradient, _div
from nilearn.decoding.space_net_solvers import (
    _graph_net_data_function,
    _graph_net_adjoint_data_function,
    _squared_loss_and_spatial_grad,
    _logistic_data_loss_and_spatial_grad,
    _squared_loss_and_spatial_grad_derivative,
    _logistic_data_loss_and_spatial_grad_derivative,
    _squared_loss_derivative_lipschitz_constant,
    _logistic_derivative_lipschitz_constant,
    mfista)
from nilearn.decoding.space_net import BaseSpaceNet
from nilearn._utils.compat import get_affine

# Data used in almost all tests
import nibabel
from .test_same_api import to_niimgs
from .simulate_graph_net_data import create_graph_net_simulation_data


def _make_data(task="regression", size=4):
    X, y, w, mask = create_graph_net_simulation_data(
        snr=1., n_samples=10, size=size, n_points=5, random_state=42,
        task=task)
    X_, _ = to_niimgs(X, [size] * 3)
    mask_ = nibabel.Nifti1Image(mask.astype(np.float),
                                get_affine(X_))
    return X, y, w, mask, mask_, X_

X, y, w, mask, mask_, X_ = _make_data()


def get_gradient_matrix(w_size, mask):
    """
    Given a number of features and a mask (which has the property
    mask[mask==True].size == w_size) computes a matrix G such that for
    a w vector we have np.dot(G, w) == gradient(w_masked)[mask]
    """
    grad_matrix = np.zeros((mask.ndim * w_size, w_size))
    grad_mask = np.array([mask for _ in range(mask.ndim)])
    image_buffer = np.zeros(mask.shape)

    for i in range(w_size):
        base_vector = np.zeros(w_size)
        base_vector[i] = 1
        image_buffer[mask] = base_vector
        gradient_column = _gradient(image_buffer)[grad_mask]
        grad_matrix[:, i] = gradient_column

    return grad_matrix


def test_grad_matrix():
    """Test for matricial form of gradient"""
    rng = check_random_state(42)
    G = get_gradient_matrix(w.size, mask)
    image_buffer = np.zeros(mask.shape)
    grad_mask = np.array([mask for _ in range(mask.ndim)])
    for _ in range(10):
        v = rng.rand(w.size) * rng.randint(1000)
        image_buffer[mask] = v
        assert_almost_equal(_gradient(image_buffer)[grad_mask], np.dot(G, v))


def test_adjointness(size=4):
    """Tests for adjointness between gradient and div operators"""
    rng = check_random_state(42)
    for _ in range(3):
        image_1 = rng.rand(size, size, size)
        image_2 = rng.rand(3, size, size, size)
        Axdoty = np.dot((_gradient(image_1).ravel()), image_2.ravel())
        xdotAty = np.dot((_div(image_2).ravel()), image_1.ravel())
        assert_almost_equal(Axdoty, - xdotAty)


def test_identity_adjointness(size=4):
    """Tests adjointess between _graph_net_data_function and
    _graph_net_adjoint_data_function, with identity design matrix"""
    rng = check_random_state(42)

    # A mask full of ones
    mask = np.ones((size, size, size), dtype=np.bool)

    # But with some zeros
    mask[0:3, 0:3, 0:3] = 0
    adjoint_mask = np.array([mask for _ in range(mask.ndim)])
    n_samples = np.sum(mask)
    X = np.eye(n_samples)
    l1_ratio = 0.5
    for _ in range(10):
        x = rng.rand(np.sum(mask))
        y = rng.rand(n_samples + np.sum(mask) * mask.ndim)
        Axdoty = np.dot(_graph_net_data_function(X, x, mask, l1_ratio), y)
        xdotAty = np.dot(_graph_net_adjoint_data_function(
            X, y, adjoint_mask, l1_ratio), x)
        assert_almost_equal(Axdoty, xdotAty)


def test_operators_adjointness(size=4):
    """The same as test_identity_adjointness, but with generic design matrix"""
    rng = check_random_state(42)

    # A mask full of ones
    mask = np.ones((size, size, size), dtype=np.bool)

    # But with some zeros
    mask[0:3, 0:3, 0:3] = 0
    adjoint_mask = np.array([mask for _ in range(mask.ndim)])
    n_samples = 200
    X = rng.rand(n_samples, np.sum(mask))
    l1_ratio = 0.5
    for _ in range(10):
        x = rng.rand(np.sum(mask))
        y = rng.rand(n_samples + np.sum(mask) * mask.ndim)
        Axdoty = np.dot(_graph_net_data_function(X, x, mask, l1_ratio), y)
        xdotAty = np.dot(_graph_net_adjoint_data_function(
            X, y, adjoint_mask, l1_ratio), x)
        np.testing.assert_almost_equal(Axdoty, xdotAty)


def test__squared_loss_gradient_at_simple_points():
    """Tests gradient of data loss function in points near to zero. This is
    a not so hard test, just for detecting big errors"""
    X, y, w, mask = create_graph_net_simulation_data(n_samples=10, size=4)
    grad_weight = 1
    func = lambda w: _squared_loss_and_spatial_grad(X, y, w, mask,
                                                    grad_weight)
    func_grad = lambda w: _squared_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)
    for i in range(0, w.size, 2):
        point = np.zeros(*w.shape)
        point[i] = 1
        assert_almost_equal(sp.optimize.check_grad(func, func_grad, point),
                            0, decimal=3)


def test_logistic_gradient_at_simple_points():
    # Tests gradient of logistic data loss function in points near to zero.
    # This is a not so hard test, just for detecting big errors
    X, y, w, mask = create_graph_net_simulation_data(n_samples=10, size=4)
    grad_weight = 1
    # Add the intercept
    w = np.append(w, 0)
    func = lambda w: _logistic_data_loss_and_spatial_grad(
        X, y, w, mask, grad_weight)
    func_grad = lambda w: _logistic_data_loss_and_spatial_grad_derivative(
        X, y, w, mask, grad_weight)
    for i in range(0, w.size, 7):
        point = np.zeros(*w.shape)
        point[i] = 1
        assert_almost_equal(sp.optimize.check_grad(func, func_grad, point),
                            0, decimal=3)


def test__squared_loss_derivative_lipschitz_constant():
    # Tests Lipschitz-continuity of the derivative of _squared_loss loss
    # function
    rng = check_random_state(42)
    grad_weight = 2.08e-1
    lipschitz_constant = _squared_loss_derivative_lipschitz_constant(
        X, mask, grad_weight)
    for _ in range(20):
        x_1 = rng.rand(*w.shape) * rng.randint(1000)
        x_2 = rng.rand(*w.shape) * rng.randint(1000)
        gradient_difference = extmath.norm(
            _squared_loss_and_spatial_grad_derivative(X, y, x_1, mask,
                                                      grad_weight)
            - _squared_loss_and_spatial_grad_derivative(X, y, x_2, mask,
                                                        grad_weight))
        point_difference = extmath.norm(x_1 - x_2)
        assert_true(
            gradient_difference <= lipschitz_constant * point_difference)


def test_logistic_derivative_lipschitz_constant():
    # Tests Lipschitz-continuity of of the derivative of logistic loss
    rng = check_random_state(42)
    grad_weight = 2.08e-1
    lipschitz_constant = _logistic_derivative_lipschitz_constant(
        X, mask, grad_weight)
    for _ in range(20):
        x_1 = rng.rand((w.shape[0] + 1)) * rng.randint(1000)
        x_2 = rng.rand((w.shape[0] + 1)) * rng.randint(1000)
        gradient_difference = extmath.norm(
            _logistic_data_loss_and_spatial_grad_derivative(
                X, y, x_1, mask, grad_weight)
            - _logistic_data_loss_and_spatial_grad_derivative(
                X, y, x_2, mask, grad_weight))
        point_difference = extmath.norm(x_1 - x_2)
        assert_true(
            gradient_difference <= lipschitz_constant * point_difference)


def test_max_alpha__squared_loss():
    """Tests that models with L1 regularization over the theoretical bound
    are full of zeros, for logistic regression"""
    l1_ratios = np.linspace(0.1, 1, 3)
    reg = BaseSpaceNet(mask=mask_, max_iter=10, penalty="graph-net",
                       is_classif=False)
    for l1_ratio in l1_ratios:
        reg.l1_ratios = l1_ratio
        reg.alphas = np.max(np.dot(X.T, y)) / l1_ratio
        reg.fit(X_, y)
        assert_almost_equal(reg.coef_, 0.)


def test_tikhonov_regularization_vs_graph_net():
    # Test for one of the extreme cases of Graph-Net: That is, with
    # l1_ratio = 0 (pure Smooth), we compare Graph-Net's performance
    # with the analytical solution for Tikhonov Regularization

    # XXX A small dataset here (this test is very lengthy)
    G = get_gradient_matrix(w.size, mask)
    optimal_model = np.dot(sp.linalg.pinv(
        np.dot(X.T, X) + y.size * np.dot(G.T, G)), np.dot(X.T, y))
    graph_net = BaseSpaceNet(
        mask=mask_, alphas=1. * X.shape[0], l1_ratios=0., max_iter=400,
        fit_intercept=False,
        screening_percentile=100., standardize=False)
    graph_net.fit(X_, y.copy())
    coef_ = graph_net.coef_[0]
    graph_net_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, coef_) - y) ** 2\
        + 0.5 * extmath.norm(np.dot(G, coef_)) ** 2
    optimal_model_perf = 0.5 / y.size * extmath.norm(
        np.dot(X, optimal_model) - y) ** 2\
        + 0.5 * extmath.norm(np.dot(G, optimal_model)) ** 2
    assert_almost_equal(graph_net_perf, optimal_model_perf, decimal=1)


def test_mfista_solver_graph_net_no_l1_term():
    w = np.zeros(2)
    X = np.array([[1, 0], [0, 4]])
    y = np.array([-10, 20])
    f1 = lambda w: 0.5 * np.dot(np.dot(X, w) - y, np.dot(X, w) - y)
    f1_grad = lambda w: np.dot(X.T, np.dot(X, w) - y)
    f2_prox = lambda w, l, *args, **kwargs: (w, dict(converged=True))
    lipschitz_constant = _squared_loss_derivative_lipschitz_constant(
        X, (np.eye(2) == 1).astype(np.bool), 1)
    estimate_solution, _, _ = mfista(
        f1_grad, f2_prox, f1, lipschitz_constant, w.size, tol=1e-8)

    solution = np.array([-10, 5])
    assert_almost_equal(estimate_solution, solution, decimal=4)
