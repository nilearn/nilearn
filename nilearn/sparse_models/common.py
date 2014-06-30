"""
Common functions and base classes. Used by more specialized modules like
tv.py, smooth_lasso.py, etc.

"""
# Author: DOHMATOB Elvis Dopgima,
#         Gaspar Pizarro,
#         Fabian Pedragosa,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

from functools import partial
import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state


def check_lipschitz_continuous(f, ndim, L, n_trials=10, err_msg=None):
    """Empirically check Lipschitz continuity of a function.

    If this test is passed, then we are empirically confident in the
    Lipschitz continuity of the function with respect to the given
    constant `L`. This confidence increases with the `n_trials` parameter.

    Parameters
    ----------
    f: callable,
      The function to be checked for Lipschitz continuity.
      `f` takes a vector of float as unique argument.
      The size of the input vector is determined by `ndim`.

    ndim: int,
      Dimension of the input of the function to be checked for Lipschitz
      continuity (i.e. it corresponds to the size of the vector that `f`
      takes as an argument).

    L: float,
      Constant associated to the Lipschitz continuity.

    n_trials: int,
      Number of tests performed when assessing the Lipschitz continuity of
      function `f`. The more tests, the more confident we are in the
      Lipschitz continuity of `f` if the test passes.

    err_msg: {str, or None},
      String used to tune the output message when the test fails.
      If `None`, we'll generate our own.

    Notes
    -----
    If you are implementing a proximal gradient type algorithm (FISTA, etc.),
    then you should strongly consider testing Lipschitz continuity of your
    smooth terms. Failure of this test typically implies you have a bug in
    the way you are computing the gradient of your smooth terms, or the
    way you are bounding their Lipschitz constant!

    Raises
    ------
    AssertionError

    """

    # check random state
    rng = check_random_state(42)

    for x in rng.randn(n_trials, ndim):
        for y in rng.randn(n_trials, ndim):
            err_msg = "LC counter example: (%s, %s)" % (
                x, y) if err_msg is None else err_msg
            a = linalg.norm(f(x).ravel() - f(y).ravel(), 2)
            b = L * linalg.norm(x - y, 2)
            assert a <= b, err_msg + ("(a = %g >= %g)" % (a, b))


# xxx: isn't it a more generic function?
def compute_mse_lipschitz_constant(X):
    """Compute the Lipschitz constant (upper bound) for the gradient of a map:

        w -> .5 * ||y - Xw||^2

    Parameters
    ----------
    X: np.ndarray,
      Input map.

    Returns
    -------
    lipschitz_constant: float,
      Lipschitz constant of the gradient of the input map.

    """
    # On big matrices like those that we have in neuroimaging, svdvals
    # is faster than a power iteration (even when using arpack's)
    return linalg.svdvals(X)[0] ** 2


def compute_logistic_lipschitz_constant(X):
    """Compute the Lipschitz constant (upper bound) for the gradient of the
    logistic sum:

         w -> \sum_i log(1+exp(-y_i*(x_i*w + v)))

    """
    # N.B: we handle intercept!
    X = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))
    return compute_mse_lipschitz_constant(X)  # XXX doubtful


# XXX: functions that return variable number of outputs are bad
def compute_mse(X, y, w, mask=None, compute_energy=True, compute_grad=True,
                unmask_grad=True, compute_hess=False, unmask_hess=True):

    """Compute the MSE error, and optionally, its gradient too.

    The energy is

        MSE = .5 * ||y - Xw||^2 / n_samples

    A (1 / n_samples) factor is applied to the MSE.

    Parameters
    ----------
    X: 2D array of shape (n_samples, n_features)
        Design matrix.

    y: 1D array of length n_samples
        Target / response vector.

    w: array_like, shape (n_voxels,)
        Unmasked, ravelized input map.

    mask: array_like of same shape as w, optional (default None)
        mask for ROI

    compute_energy: bool, optional (default True)
        If set then energy is computed, otherwise only gradient is computed.

    compute_grad: bool, optional (default True)
        If set then energy is computed, otherwise only energy is computed.

    unmask_grad: bool, optional (default True)
        If set, then computed gradient is unmasked before returned.

    Returns
    -------
    energy: float
        Energy (returned if `compute_energy` is set.

    gradient: 1D array
        Gradient of energy (returned if `compute_grad` is set.

    """
    assert compute_energy or compute_grad

    # mask the input vector w
    w = w.ravel()
    if mask is not None:
        w = w[mask.ravel()]
    residual = np.dot(X, w) - y

    # compute energy
    if compute_energy:
        energy = .5 * np.dot(residual, residual)
        if not compute_grad:
            return energy

    grad = np.dot(X.T, residual)  # XXX use sk's fast_dot
    if unmask_grad and mask is not None:
        grad = _unmask(grad, mask).ravel()

    if not compute_energy and not compute_hess:
        return grad

    if compute_energy and not compute_hess:
        return energy, grad

    def hess_matvec(vec):
        if mask is not None:
            vec = vec[mask]

        out = np.dot(X.T, np.dot(X, vec))  # XXX use sk's fast_dot
        if unmask_hess and mask is not None:
            out = _unmask(out, mask)

        return out

    if not compute_energy:
        return grad, hess_matvec

    return energy, grad, hess_matvec


def get_gradient_id_shape(img_shape):
    """Return the shape of the gradient + id operator output.

    """
    return [len(img_shape) + 1] + list(img_shape)


def tv_l1_from_gradient(gradient):
    """Our total-variation like norm: TV + l1

    Parameters
    ----------
    gradient: array
       precomputed "gradient + id" array

    """

    tv_term = np.sum(np.sqrt(np.sum(gradient[:-1] * gradient[:-1],
                                    axis=0)))

    l1_term = np.abs(gradient[-1]).sum()

    return l1_term + tv_term


def div_id(grad, l1_ratio=.5):
    """Compute divergence + id of image gradient + id

    Parameters
    ----------
    l1_ratio: float, optional (default .5)
        relative weight of l1; float between 0 and 1 inclusive.
        TV+L1 penalty will be (alpha not shown here):
        (1 - l1_ratio) * ||w||_TV + l1_ratio * ||w||_1

    Returns
    -------
    res: ndarray of shape grad.shape[1:]
    otherwise
        the computed divergent or divergent + id, operator

    """

    assert 0. <= l1_ratio <= 1., (
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


def gradient_id(img, l1_ratio=.5):
    """Compute gradient + id of an image

    Parameters
    ===========
    img: ndarray
        N-dimensional image

    Parameters
    ----------
    l1_ratio: float, optional (default .5)
        relative weight of l1; float between 0 and 1 inclusive.
        TV+L1 penalty will be (alpha not shown here):

        (1 - l1_ratio) * ||w||_TV + l1_ratio * ||w||_1

    Returns
    -------
    gradient: ndarray of shape (img.ndim, *img.shape) if `with_id` is True,
    or of shape (img.ndim + 1, *img.shape) otherwise
        gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original
        array img.

    """

    assert 0. <= l1_ratio <= 1., (
        "l1_ratio must be in the interval [0, 1]; got %s" % l1_ratio)

    # shape = [img.ndim * int(l1_ratio < 1.) + int(l1_ratio > 0.)
    #          ] + list(img.shape)
    shape = [img.ndim + 1] + list(img.shape)
    gradient = np.zeros(shape, dtype=np.float)  # xxx: img.dtype?

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
    """Unmask an image into whole brain, with off-mask voxels cast to 0.

    Parameters
    ----------
    w: np.ndarray,
      The image to be unmasked.

    mask: np.ndarray or None,
      The mask used in the unmasking operation.

    """
    if mask is None:
        return w

    out = np.zeros(mask.shape, dtype=w.dtype)
    out[mask] = np.ravel(w)

    return out


def test_grad_div_adjoint_arbitrary_ndim(size=5, max_ndim=5, random_state=42):
    # We need to check that <D x, y> = <x, DT y> for x and y random vectors
    random_state = np.random.RandomState(random_state)

    for ndim in xrange(1, max_ndim):
        shape = tuple([size] * ndim)
        x = np.random.normal(size=shape)
        y = np.random.normal(size=[ndim + 1] + list(shape))
        for l1_ratio in [0., .1, .3, .5, .7, .9, 1.]:
            np.testing.assert_almost_equal(
                np.sum(gradient_id(x, l1_ratio=l1_ratio) * y),
                -np.sum(x * div_id(y, l1_ratio=l1_ratio)))


def _sigmoid(t, copy=True):
    """Helper function: return 1. / (1 + np.exp(-t))"""
    if copy:
        t = np.copy(t)
    t *= -1.
    t = np.exp(t, t)
    t += 1.
    t = np.reciprocal(t, t)
    return t


def logistic(X, y, w, mask=None):
    """Compute the logistic function of the data: sum(sigmoid(yXw))"""
    if mask is not None:
        mask = mask.ravel()
        # last coef of w is the intercept_
        w = np.append(w[:-1][mask], w[-1])
    z = np.dot(X, w[:-1]) + w[-1]
    yz = y * z
    idx = yz > 0
    out = np.empty_like(yz)
    out[idx] = np.log1p(np.exp(-yz[idx]))
    out[~idx] = -yz[~idx] + np.log1p(np.exp(yz[~idx]))
    out = out.sum()
    return out


def logistic_grad(X, y, w, mask=None):
    """Computes the derivative of logistic"""
    if mask is not None:
        mask = mask.ravel()
        w = np.append(w[:-1][mask], w[-1])
    z = np.dot(X, w[:-1]) + w[-1]
    yz = y * z
    z = _sigmoid(yz, copy=False)
    z0 = (z - 1) * y
    grad = np.empty(w.shape)
    grad[:-1] = np.dot(X.T, z0)
    grad[-1] = np.sum(z0)
    if mask is not None:
        grad = np.append(_unmask(grad[:-1], mask), grad[-1])
    return grad

# aliases (for "backward compatibility")
mse_loss = squared_loss = partial(compute_mse, compute_grad=False)
mse_loss_grad = squared_loss_grad = partial(compute_mse, compute_energy=False)
squared_loss_lipschitz_constant = compute_mse_lipschitz_constant
logistic_lipschitz_constant = compute_logistic_lipschitz_constant
