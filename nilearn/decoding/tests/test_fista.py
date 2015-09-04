from nose.tools import assert_equal, assert_true
import numpy as np
from nilearn.decoding.fista import mfista
from nilearn.decoding.proximal_operators import _prox_l1
from nilearn.decoding.objective_functions import (
    _squared_loss,
    _logistic,
    _squared_loss_grad,
    _logistic_loss_lipschitz_constant,
    spectral_norm_squared)
from nilearn.decoding.fista import _check_lipschitz_continuous


def test_logistic_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = _logistic_loss_lipschitz_constant(X)
        _check_lipschitz_continuous(lambda w: _logistic(
            X, y, w), n_features + 1, L)


def test_squared_loss_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = spectral_norm_squared(X)
        _check_lipschitz_continuous(lambda w: _squared_loss_grad(
            X, y, w), n_features, L)


def test_input_args_and_kwargs():
    rng = np.random.RandomState(42)
    p = 125
    noise_std = 1e-1
    sig = np.zeros(p)
    sig[[0, 2, 13, 4, 25, 32, 80, 89, 91, 93, -1]] = 1
    sig[:6] = 2
    sig[-7:] = 2
    sig[60:75] = 1
    y = sig + noise_std * rng.randn(*sig.shape)
    X = np.eye(p)
    mask = np.ones((p,)).astype(np.bool)
    alpha = .01
    alpha_ = alpha * X.shape[0]
    l1_ratio = .2
    l1_weight = alpha_ * l1_ratio
    f1 = lambda w: _squared_loss(X, y, w, compute_grad=False)
    f1_grad = lambda w: _squared_loss(X, y, w, compute_grad=True,
                                      compute_energy=False)
    f2_prox = lambda w, l, *args, **kwargs: (_prox_l1(w, l * l1_weight),
                                             dict(converged=True))
    total_energy = lambda w: f1(w) + l1_weight * np.sum(np.abs(w))
    for cb_retval in [0, 1]:
        for verbose in [0, 1]:
            for dgap_factor in [1., None]:
                best_w, objective, init = mfista(
                    f1_grad, f2_prox, total_energy, 1., p,
                    dgap_factor=dgap_factor,
                    callback=lambda _: cb_retval, verbose=verbose,
                    max_iter=100)
                assert_equal(best_w.shape, mask.shape)
                assert_true(isinstance(objective, list))
                assert_true(isinstance(init, dict))
                for key in ["w", "t", "dgap_tol", "stepsize"]:
                    assert_true(key in init)
