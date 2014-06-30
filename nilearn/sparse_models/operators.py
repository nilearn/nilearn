"""Implementations of multiple proximal operators for TV-1, S-LASSO, etc.

For TV-l1, the core idea here is to modify the analysis operator in the Beck &
Teboulle approach (actually Chambolle) to keep the identity and thus to
end up with an l1.

"""
# Author: DOHMATOB Elvis Dopgima, ...
# License: simplified BSD

from math import sqrt
import numpy as np
from .common import (tv_l1_from_gradient, div_id, gradient_id,
                     get_gradient_id_shape)


def prox_l1(y, alpha, copy=True):
    """proximity operator for l1 norm"""
    shrink = np.zeros(y.shape)
    if copy:
        y = y.copy()
    y_nz = y.nonzero()
    shrink[y_nz] = np.maximum(1 - alpha / np.abs(y[y_nz]), 0)
    y *= shrink
    return y


def _projector_on_dual(grad, l1_ratio):
    """
    modifies IN PLACE the gradient + id to project it
    on the l21 unit ball in the gradient direction and the l1 ball in the
    identity direction
    """

    # The l21 ball for the gradient direction
    if l1_ratio < 1.:
        end = len(grad) - int(l1_ratio > 0.)
        norm = np.sqrt(np.sum(grad[:end] * grad[:end], 0))
        norm.clip(1., out=norm)  # set everythx < 1 to 1
        for grad_comp in grad[:end]:
            grad_comp /= norm

    # The l1 ball for the identity direction
    if l1_ratio > 0.:
        norm = np.abs(grad[-1])
        norm.clip(1., out=norm)
        grad[-1] /= norm

    return grad


def dual_gap(input_img_norm, new, gap, weight, l1_ratio=1.):
    """
    dual gap of total variation denoising
    see "Total variation regularization for fMRI-based prediction of behavior",
    by Michel et al. (2011) for a derivation of the dual gap
    """
    tv_new = tv_l1_from_gradient(gradient_id(new, l1_ratio=l1_ratio))
    d_gap = (gap ** 2).sum() + 2 * weight * tv_new - input_img_norm + (
        new * new).sum()
    return 0.5 * d_gap


def _objective_function(input_img, output_img, gradient, l1_ratio, weight):
    diff = (input_img - output_img).ravel()
    return (.5 * (diff * diff).sum()
            + weight * tv_l1_from_gradient(gradient))


def prox_tv_l1(im, l1_ratio=.05, weight=50, dgap_tol=5.e-5, x_tol=None,
               max_iter=200, check_gap_frequency=4, val_min=None,
               val_max=None, verbose=False, fista=True, callback=None,
               init=None, return_info=False):
    """
    Compute the TV + l1 proximal (ie total-variation +l1 denoising) on
    2-d and 3-d images

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TV(res),

    where TV is the isotropic l1 norm of the gradient.

    Parameters
    ----------
    im: ndarray of floats (2-d or 3-d)
        input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight: float, optional
        denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    dgap_tol: float, optional
        precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the
        squared l2 norm of the image (for contrast invariance).

    x_tol: float or None, optional
        The maximal relative difference between input and output. If
        specified, this specifies a stopping criterion on x, rather than
        the dual gap

    max_iter: int, optional
        maximal number of iterations used for the optimization.

    val_min: None or float, optional
        an optional lower bound constraint on the reconstructed image

    val_max: None or float, optional
        an optional upper bound constraint on the reconstructed image

    verbose: bool, optional
        if True, print the dual gap of the optimization

    fista: bool, optional
        if True, uses a FISTA loop to perform the optimization.
        if False, uses an ISTA loop

    callback: callable
        Callable that takes the local variables at each
        steps. Useful for tracking

    init: array of shape shape as im
        Starting point for the optimization

    return_info: boolean
        If True, a dictionnary of inner variables is returned in addition

    Returns
    -------
    out: ndarray
        denoised array

    Notes
    -----
    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This function implements the FISTA (Fast Iterative Shrinkage
    Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
    total variation denoising in "Fast gradient-based algorithms for
    constrained total variation image denoising and deblurring problems"
    (2009).

    For details on implementing the bound constraints, read the Beck and
    Teboulle paper.
    """
    weight = float(weight)
    input_img = im
    input_img_norm = (im.ravel() * im.ravel()).sum()
    if not input_img.dtype.kind == 'f':
        input_img = input_img.astype(np.float)
    shape = get_gradient_id_shape(im.shape)
    grad_im = np.zeros(shape)
    grad_aux = np.zeros(shape)
    t = 1.
    i = 0

    # XXX should use same formula as in primal_dual.tv_l1.py
    lipschitz_constant = 1.1 * (4 * input_img.ndim * (1 - l1_ratio)
                                ** 2 + l1_ratio ** 2)

    # negated_output is the negated primal variable in the optimization
    # loop
    if init is None:
        negated_output = -input_img
    else:
        negated_output = -init

    # Clipping values for the inner loop
    negated_val_min = np.inf
    negated_val_max = -np.inf
    if val_min is not None:
        negated_val_min = -val_min
    if val_max is not None:
        negated_val_max = -val_max
    if True or (val_min is not None or val_max is not None):
        # With bound constraints, the stopping criterion is on the
        # evolution of the output
        negated_output_old = negated_output.copy()
    grad_tmp = None
    old_dgap = np.inf
    dgap = np.inf

    # A boolean to control if we are going to do a fista step
    fista_step = fista

    while i < max_iter:
        grad_tmp = gradient_id(negated_output, l1_ratio=l1_ratio)
        grad_tmp *= 1. / (lipschitz_constant * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_dual(
            grad_aux, l1_ratio
            )  # XXX this makes grad_aux and grad_tmp point to thesame buffer!

        # Carefull, in the next few lines, grad_tmp and grad_aux are a
        # view on the same array, as _projector_on_dual returns a view
        # on the input array
        t_new = 0.5 * (1. + sqrt(1. + 4. * t * t))
        t_factor = (t - 1.) / t_new
        if fista_step:
            grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
        else:
            grad_aux = grad_tmp
        grad_im = grad_tmp
        t = t_new
        gap = weight * div_id(grad_aux, l1_ratio=l1_ratio)

        # Compute the primal variable
        negated_output = gap - input_img
        if (val_min is not None or val_max is not None):
            negated_output = negated_output.clip(negated_val_max,
                                                 negated_val_min,
                                                 out=negated_output)
        if (i % check_gap_frequency) == 0:
            if x_tol is None:
                # Stopping criterion based on the dual_gap
                if val_min is not None or val_max is not None:
                    # We need to recompute the dual variable
                    gap = negated_output + input_img
                old_dgap = dgap
                dgap = dual_gap(input_img_norm, -negated_output,
                                gap, weight, l1_ratio=l1_ratio)
                if verbose:
                    print '\tProxTVl1: Iteration % 2i, dual gap: % 6.3e' % (
                        i, dgap)
                if dgap < dgap_tol:
                    break
                if old_dgap < dgap:
                    # M-FISTA strategy: switch to an ISTA to have
                    # monotone convergence
                    fista_step = False
                elif fista:
                    fista_step = True
            else:
                # Stopping criterion based on x_tol
                diff = np.max(np.abs(negated_output_old - negated_output))
                diff /= np.max(np.abs(negated_output))
                if verbose:
                    print ('\tProxTVl1 iteration % 2i, relative difference:'
                           ' % 6.3e, energy: % 6.3e') % (
                        i, diff, _objective_function(
                                   input_img, -negated_output, gradient_id(
                                       negated_output, l1_ratio=l1_ratio),
                                   l1_ratio, weight))
                if diff < x_tol:
                    break
                negated_output_old = negated_output
        if callback is not None:
            callback(locals())
        i += 1

    # Compute the primal variable, however, here we must use the ista
    # value, not the fista one
    output = input_img - weight * div_id(grad_im, l1_ratio=l1_ratio)
    if (val_min is not None or val_max is not None):
        output = output.clip(val_min, val_max, out=output)
    if return_info:
        return output, dict(converged=(i < max_iter))
    return output


def intercepted_prox_tv_l1(w, shape, l1_ratio, weight, dgap_tol, max_iter=5000,
                           init=None, verbose=False):
    """
    Computation of TV-l1 prox, taking into account the intercept.

    Parameters
    ----------
    weight: float
       Weight in prox. This would be something like `alpha_ * stepsize`,
       where `alpha_` is the effective (i.e re-scaled) alpha.

    w: np.array of `w_size` floats
        The point at which the prox is being computed

    init: np.array of `w_size` - 1 floats, optional (default None)
        Initialization vector for the prox.

    dgap_tol: float
        Dual-gap tolerance for TV-l1 prox operator approximation loop.

    """

    init = init.reshape(shape) if not init is None else init
    out, prox_info = prox_tv_l1(
        w[:-1].reshape(shape), weight=weight,
        l1_ratio=l1_ratio, dgap_tol=dgap_tol, return_info=True,
        init=init, max_iter=max_iter, verbose=verbose)

    return np.append(out, w[-1]), prox_info


# def prox_l21(Y, alpha, axis=1, copy=True):
#     """proximity operator for l21 norm"""
#     shrink = np.zeros(Y.shape[(axis + 1) % 2], 1)
#     if copy:
#         Y = Y.copy()
#     l2_norms = np.sqrt(np.sum(Y ** 2, axis=axis))
#     nz = l2_norms.nonzero()
#     shrink[nz] = np.maximum(1 - alpha / l2_norms[nz], 0)
#     Y *= shrink
#     return Y


def prox_l21(x, l1_ratio, tau, isotropic=True):
    """
    Prox tau * L21 inplace,

    where l21 is the l2 norm of the first ndim (=1, 2, 3, etc.) lines of x,
    and the l1 of all the rest (including the group defined by these, and
    the remaing  last line of x).

    Parameters
    ----------
    l1_ratio: positive float in the interval [0, 1]
        the usual trade-off parameter between l1 and TV terms of the
        underlying penalty.

    tau: positive float
        the radius of the l21 ball of the projection (note that
        s(a) = a - P_tau(a) as usual)

    """

    shrink = np.zeros_like(x)

    if isotropic:
        shrink[:-1] = np.sqrt((x[:-1] * x[:-1]).sum(axis=0))
    else:
        shrink[:-1] = np.abs(x[:-1]).sum(axis=0)

    shrink[-1] = np.abs(x[-1])

    shrink[shrink == 0.] = 1.
    shrink = np.maximum(shrink - tau, 0) / shrink
    x *= shrink

    return x


# def estimate_lipschitz_constant_graph(w0, L):
#     """Compute approximate lipschitz constant
#     of callable linear operator: x -> Lx
#     using a power method"""
#     a = np.random.randn(*w0.shape)
#     a /= linalg.norm(a)
#     for i in range(100):
#         b = L(a)
#         a = b / linalg.norm(b)

#     lipschitz_constant = (b * a).sum()
#     return 1.1 * lipschitz_constant
