
"""
Common functions and base classes.

"""
# Author: DOHMATOB Elvis Dopgima,
#         PIZARRO Gaspar,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,
#         PEDREGOSA Fabian
# License: simplified BSD

from functools import partial
import numpy as np
from scipy import linalg


def spectral_norm_squared(X):
    """Computes square of the operator 2-norm (spectral norm) of X

    This corresponds to the Lipschitz constant of the gradient of the
    squared-loss function:

        w -> .5 * ||y - Xw||^2

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      Design matrix.

    Returns
    -------
    lipschitz_constant : float
      The square of the spectral norm of X.

    """
    # On big matrices like those that we have in neuroimaging, svdvals
    # is faster than a power iteration (even when using arpack's)
    return linalg.svdvals(X)[0] ** 2


def _logistic_loss_lipschitz_constant(X):
    """Compute the Lipschitz constant (upper bound) for the gradient of the
    logistic sum:

         w -> \sum_i log(1+exp(-y_i*(x_i*w + v)))

    """
    # N.B: we handle intercept!
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return spectral_norm_squared(X)


def _squared_loss(X, y, w, compute_energy=True, compute_grad=False):
    """Compute the MSE error, and optionally, its gradient too.

    The cost / energy function is

        MSE = .5 * ||y - Xw||^2

    A (1 / n_samples) factor is applied to the MSE.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    compute_energy : bool, optional (default True)
        If set then energy is computed, otherwise only gradient is computed.

    compute_grad : bool, optional (default False)
        If set then gradient is computed, otherwise only energy is computed.

    Returns
    -------
    energy : float
        Energy (returned if `compute_energy` is set).

    gradient : ndarray, shape (n_features,)
        Gradient of energy (returned if `compute_grad` is set).

    """
    if not (compute_energy or compute_grad):
        raise RuntimeError(
            "At least one of compute_energy or compute_grad must be True.")

    residual = np.dot(X, w) - y

    # compute energy
    if compute_energy:
        energy = .5 * np.dot(residual, residual)
        if not compute_grad:
            return energy

    grad = np.dot(X.T, residual)

    if not compute_energy:
        return grad

    return energy, grad


def _tv_l1_from_gradient(spatial_grad):
    """Energy contribution due to penalized gradient, in TV-L1 model.

    Parameters
    ----------
    spatial_grad : ndarray, shape (4, nx, ny, nx)
       precomputed "gradient + id" array

    Returns
    -------
    out : float
        Energy contribution due to penalized gradient.
    """

    tv_term = np.sum(np.sqrt(np.sum(spatial_grad[:-1] * spatial_grad[:-1],
                                    axis=0)))
    l1_term = np.abs(spatial_grad[-1]).sum()
    return l1_term + tv_term


def _div_id(grad, l1_ratio=.5):
    """Compute divergence + id of image gradient + id

    Parameters
    ----------
    grad : ndarray, shape (4, nx, ny, nz, ...)
        where `img_shape` is the shape of the brain bounding box, and
        n_axes = len(img_shape).

    l1_ratio : float in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in the penalization.

    Returns
    -------
    res : ndarray, shape (nx, ny, nz, ...)
        The computed divergence + id operator.

    Raises
    ------
    RuntimeError

    """

    if not (0. <= l1_ratio <= 1.):
        raise RuntimeError(
            "l1_ratio must be in the interval [0, 1]; got %s" % l1_ratio)

    res = np.zeros(grad.shape[1:])

    # the divergence part
    for d in range((grad.shape[0] - 1)):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        if len(this_grad) > 1:
            this_res[-1] -= this_grad[-2]

    res *= (1. - l1_ratio)

    # the identity part
    res -= l1_ratio * grad[-1]

    return res


def _gradient_id(img, l1_ratio=.5):
    """Compute gradient + id of an image

    Parameters
    ----------
    img : ndarray, shape (nx, ny, nz, ...)
        N-dimensional image

    l1_ratio : float in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in the penalization.

    Returns
    -------
    gradient : ndarray, shape (4, nx, ny, nz, ...).
        Spatial gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original array img.

    Raises
    ------
    RuntimeError

    """

    if not (0. <= l1_ratio <= 1.):
        raise RuntimeError(
            "l1_ratio must be in the interval [0, 1]; got %s" % l1_ratio)

    shape = [img.ndim + 1] + list(img.shape)
    gradient = np.zeros(shape, dtype=np.float)

    # the gradient part: 'Clever' code to have a view of the gradient
    # with dimension i stop at -1
    slice_all = [0, slice(None, -1)]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))

    gradient[:-1] *= (1. - l1_ratio)

    # the identity part
    gradient[-1] = l1_ratio * img

    return gradient


def _unmask(w, mask):
    """Unmask an image into whole brain, with off-mask voxels set to 0.

    Parameters
    ----------
    w : ndarray, shape (n_features,)
      The image to be unmasked.

    mask : ndarray, shape (nx, ny, nz)
      The mask used in the unmasking operation. It is required that
      mask.sum() == n_features.

    Returns
    -------
    out : 3d of same shape as `mask`.
        The unmasked version of `w`
    """

    if mask.sum() != len(w):
        raise ValueError("Expecting mask.sum() == len(w).")
    out = np.zeros(mask.shape, dtype=w.dtype)
    out[mask] = w
    return out


def _sigmoid(t, copy=True):
    """Helper function: return 1. / (1 + np.exp(-t))"""
    if copy:
        t = np.copy(t)
    t *= -1.
    t = np.exp(t, t)
    t += 1.
    t = np.reciprocal(t, t)
    return t


def _logistic(X, y, w):
    """Compute the logistic function of the data: sum(sigmoid(yXw))

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector. Each entry must be +1 or -1.

    w : ndarray, shape (n_features,)
        Unmasked, ravelized input map.

    Returns
    -------
    energy : float
        Energy contribution due to logistic data-fit term.
    """

    z = np.dot(X, w[:-1]) + w[-1]
    yz = y * z
    idx = yz > 0
    out = np.empty_like(yz)
    out[idx] = np.log1p(np.exp(-yz[idx]))
    out[~idx] = -yz[~idx] + np.log1p(np.exp(yz[~idx]))
    out = out.sum()
    return out


def _logistic_loss_grad(X, y, w):
    """Computes the derivative of logistic"""
    z = np.dot(X, w[:-1]) + w[-1]
    yz = y * z
    z = _sigmoid(yz, copy=False)
    z0 = (z - 1.) * y
    grad = np.empty(w.shape)
    grad[:-1] = np.dot(X.T, z0)
    grad[-1] = np.sum(z0)
    return grad


# gradient of squared loss function
_squared_loss_grad = partial(_squared_loss, compute_energy=False,
                             compute_grad=True)


def _gradient(w):
    """Pure spatial gradient"""
    return _gradient_id(w, l1_ratio=0.)[:-1]  # pure nabla


def _div(v):
    """Pure spatial divergence"""
    return _div_id(np.vstack((v, [np.zeros_like(v[0])])), l1_ratio=0.)
