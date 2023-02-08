import numpy as np
from nilearn.decoding.fista import mfista
from nilearn.decoding.proximal_operators import _prox_l1
from nilearn.decoding.objective_functions import (
    _squared_loss,
    _logistic,
    _squared_loss_grad,
    _logistic_loss_lipschitz_constant,
    spectral_norm_squared,
)
from nilearn.decoding.fista import _check_lipschitz_continuous


def test_logistic_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = _logistic_loss_lipschitz_constant(X)
        _check_lipschitz_continuous(
            lambda w: _logistic(X, y, w), n_features + 1, L
        )


def test_squared_loss_lipschitz(n_samples=4, n_features=2, random_state=42):
    rng = np.random.RandomState(random_state)

    for scaling in np.logspace(-3, 3, num=7):
        X = rng.randn(n_samples, n_features) * scaling
        y = rng.randn(n_samples)
        n_features = X.shape[1]

        L = spectral_norm_squared(X)
        _check_lipschitz_continuous(
            lambda w: _squared_loss_grad(X, y, w), n_features, L
        )


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
    mask = np.ones((p,)).astype(bool)
    alpha = 0.01
    alpha_ = alpha * X.shape[0]
    l1_ratio = 0.2
    l1_weight = alpha_ * l1_ratio

    def f1(w):
        return _squared_loss(X, y, w, compute_grad=False)

    def f1_grad(w):
        return _squared_loss(X, y, w, compute_grad=True, compute_energy=False)

    def f2_prox(w, l, *args, **kwargs):  # noqa: E741
        return _prox_l1(w, l * l1_weight), dict(converged=True)

    def total_energy(w):
        return f1(w) + l1_weight * np.sum(np.abs(w))

    for cb_retval in [0, 1]:
        for verbose in [0, 1]:
            for dgap_factor in [1.0, None]:
                best_w, objective, init = mfista(
                    f1_grad,
                    f2_prox,
                    total_energy,
                    1.0,
                    p,
                    dgap_factor=dgap_factor,
                    callback=lambda _: cb_retval,
                    verbose=verbose,
                    max_iter=100,
                )
                assert best_w.shape == mask.shape
                assert isinstance(objective, list)
                assert isinstance(init, dict)
                for key in ["w", "t", "dgap_tol", "stepsize"]:
                    assert key in init
