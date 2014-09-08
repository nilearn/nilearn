"""
Simple code to simulate data
"""

# Licence: BSD

import numpy as np
from scipy import linalg, ndimage
from sklearn.utils import check_random_state


def create_simulation_data(snr=1., n_samples=200, size=12, roi_size=2,
                           random_state=1, task="regression"):
    """
    Function to generate data

    Parameters
    ---------
    task: "classification" | "regression"
        If "classification", examples are drawn randomly from {-1, 1}, with no
        guarantee that the classes are equally weighted
        If "regression", examples are drawn from a normal distribution, with
        mean 0, variance 1.
    """

    generator = check_random_state(random_state)
    smooth_X = 1
    ### Coefs
    w = np.zeros((size, size, size))
    w[0:roi_size, 0:roi_size, 0:roi_size] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:] = -0.6
    w[-roi_size:, 0:roi_size:, -roi_size:] = 0.5
    w[(size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2] = 0.5
    mask = np.ones((size, size, size), dtype=np.bool)
    w = w[mask]

    ### Generate smooth background noise
    XX = generator.randn(n_samples, size, size, size)
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(XX[i, :, :, :], smooth_X)
        Xi = Xi[mask]
        noise.append(Xi)
    noise = np.array(noise)

    ### Generate the signal y
    if task == "regression":
        y = generator.randn(n_samples)
    elif task == "classification":
        y = np.zeros(n_samples)
        y[::2] = 1
    X = np.dot(y[:, np.newaxis], w[np.newaxis])
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.)
    noise_coef = norm_noise / linalg.norm(noise, 2)
    noise *= noise_coef
    snr = 20 * np.log(linalg.norm(X, 2) / linalg.norm(noise, 2))

    ### Mixing of signal + noise and splitting into train/test
    X += noise
    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]

    return X, y, w, mask


def create_smooth_simulation_data(snr=1., n_samples=200,
        size=20, n_points=10, random_state=42, task="regression"):
    """
    Function to generate data

    """
    generator = check_random_state(random_state)
    smooth_X = 1
    ### Coefs
    w = np.zeros((size, size, size))
    for _ in xrange(n_points):
        point = (generator.randint(0, size), generator.randint(0, size),
                 generator.randint(0, size))
        w[point] = 1.0
    mask = np.ones((size, size, size), dtype=np.bool)
    w = ndimage.gaussian_filter(w, sigma=1)
    w = w[mask]

    ### Generate smooth background noise
    XX = generator.randn(n_samples, size, size, size)
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(XX[i, :, :, :], smooth_X)
        Xi = Xi[mask]
        noise.append(Xi)
    noise = np.array(noise)

    ### Generate the signal y
    if task == "regression":
        y = generator.randn(n_samples)
    elif task == "classification":
        y = np.ones(n_samples)
        y[0::2] = -1
    X = np.dot(y[:, np.newaxis], w[np.newaxis])
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.)
    noise_coef = norm_noise / linalg.norm(noise, 2)
    noise *= noise_coef
    snr = 20 * np.log(linalg.norm(X, 2) / linalg.norm(noise, 2))

    ### Mixing of signal + noise and splitting into train/test
    X += noise
    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]

    return X, y, w, mask
