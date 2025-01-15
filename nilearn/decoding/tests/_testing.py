"""Test utilities for decoding module."""

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter


def create_graph_net_simulation_data(
    snr=1.0,
    n_samples=200,
    size=8,
    n_points=10,
    random_state=42,
    task="regression",
    smooth_X=1,
):
    """Generate graph net simulation data."""
    generator = np.random.default_rng(random_state)
    # Coefs
    w = np.zeros((size, size, size))
    for _ in range(n_points):
        point = (
            generator.integers(0, size),
            generator.integers(0, size),
            generator.integers(0, size),
        )
        w[point] = 1.0
    mask = np.ones((size, size, size), dtype=bool)
    w = gaussian_filter(w, sigma=1)
    w = w[mask]

    # Generate smooth background noise
    XX = generator.standard_normal((n_samples, size, size, size))
    noise = []
    for i in range(n_samples):
        Xi = gaussian_filter(XX[i, :, :, :], smooth_X)
        Xi = Xi[mask]
        noise.append(Xi)
    noise = np.array(noise)

    # Generate the signal y
    if task == "regression":
        y = generator.standard_normal(n_samples)
    elif task == "classification":
        y = np.ones(n_samples)
        y[::2] = -1
    X = np.dot(y[:, np.newaxis], w[np.newaxis])
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.0)
    noise_coef = norm_noise / linalg.norm(noise, 2)
    noise *= noise_coef
    snr = 20 * np.log(linalg.norm(X, 2) / linalg.norm(noise, 2))

    # Mixing of signal + noise and splitting into train/test
    X += noise
    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]

    return X, y, w, mask
