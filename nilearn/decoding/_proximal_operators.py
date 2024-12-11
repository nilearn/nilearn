"""Implementation of multiple proximal operators for TV-L1, Graph-Net, etc."""

# Author: DOHMATOB Elvis Dopgima,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,

from math import sqrt

import numpy as np

from nilearn._utils import logger

from ._objective_functions import (
    divergence_id,
    gradient_id,
    tv_l1_from_gradient,
)


def prox_l1(y, alpha, copy=True):
    """Compute proximity operator for L1 norm."""
    shrink = np.zeros(y.shape)
    if copy:
        y = y.copy()
    y_nz = y.nonzero()
    shrink[y_nz] = np.maximum(1 - alpha / np.abs(y[y_nz]), 0)
    y *= shrink
    return y


def prox_l1_with_intercept(x, tau):
    """Return the same as prox_l1, but just for the n-1 components."""
    x[:-1] = prox_l1(x[:-1], tau)
    return x


def _projector_on_tvl1_dual(grad, l1_ratio):
    """Compute TV-l1 duality gap.

    Modifies IN PLACE the gradient + id to project it
    on the l21 unit ball in the gradient direction and the L1 ball in the
    identity direction.
    """
    # The l21 ball for the gradient direction
    if l1_ratio < 1.0:
        # infer number of axes and include an additional axis if l1_ratio > 0
        end = len(grad) - int(l1_ratio > 0.0)
        norm = np.sqrt(np.sum(grad[:end] * grad[:end], 0))
        norm.clip(1.0, out=norm)  # set everythx < 1 to 1
        for grad_comp in grad[:end]:
            grad_comp /= norm

    # The L1 ball for the identity direction
    if l1_ratio > 0.0:
        norm = np.abs(grad[-1])
        norm.clip(1.0, out=norm)
        grad[-1] /= norm

    return grad


def _dual_gap_prox_tvl1(input_img_norm, new, gap, weight, l1_ratio=1.0):
    """Compute dual gap of total variation denoising.

    See "Total variation regularization for fMRI-based prediction of behavior",
    by Michel et al. (2011) for a derivation of the dual gap
    """
    tv_new = tv_l1_from_gradient(gradient_id(new, l1_ratio=l1_ratio))
    gap = gap.ravel()
    d_gap = (
        np.dot(gap, gap)
        + 2 * weight * tv_new
        - input_img_norm
        + (new * new).sum()
    )
    return 0.5 * d_gap


def _objective_function_prox_tvl1(input_img, output_img, gradient, weight):
    diff = (input_img - output_img).ravel()
    return 0.5 * (diff * diff).sum() + weight * tv_l1_from_gradient(gradient)


def prox_tvl1(
    input_img,
    l1_ratio=0.05,
    weight=50,
    dgap_tol=5.0e-5,
    x_tol=None,
    max_iter=200,
    check_gap_frequency=4,
    val_min=None,
    val_max=None,
    verbose=0,
    fista=True,
    init=None,
):
    """
    Compute the TV-L1 proximal (ie total-variation +l1 denoising) on 3d images.

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TVl1(res),

    Parameters
    ----------
    input_img : ndarray of floats (2-d or 3-d)
        Input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight : float, optional
        Denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    dgap_tol : float, optional
        Precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the
        squared l2 norm of the image (for contrast invariance).

    x_tol : float or None, optional
        The maximal relative difference between input and output. If
        specified, this specifies a stopping criterion on x, rather than
        the dual gap.

    max_iter : int, optional
        Maximal number of iterations used for the optimization.

    val_min : None or float, optional
        An optional lower bound constraint on the reconstructed image.

    val_max : None or float, optional
        An optional upper bound constraint on the reconstructed image.

    verbose : int or bool, optional
        If True or 1, print the dual gap of the optimization

    fista : bool, optional
        If True, uses a FISTA loop to perform the optimization.
        if False, uses an ISTA loop.

    callback : callable
        Callable that takes the local variables at each
        steps. Useful for tracking.

    init : array of shape as im
        Starting point for the optimization.

    check_gap_frequency : int, optional (default 4)
        Frequency at which duality gap is checked for convergence.

    Returns
    -------
    out : ndarray
        TV-l1-denoised image.

    Notes
    -----
    The principle of total variation denoising is explained in
    https://en.wikipedia.org/wiki/Total_variation_denoising

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

    For details on implementing the bound constraints, read the aforementioned
    Beck and Teboulle paper.
    """
    if verbose is False:
        verbose = 0
    if verbose is True:
        verbose = 1

    weight = float(weight)
    input_img_flat = input_img.view()
    input_img_flat.shape = input_img.size
    input_img_norm = np.dot(input_img_flat, input_img_flat)
    if input_img.dtype.kind != "f":
        input_img = input_img.astype(np.float64)
    shape = [len(input_img.shape) + 1, *input_img.shape]
    grad_im = np.zeros(shape)
    grad_aux = np.zeros(shape)
    t = 1.0
    i = 0
    lipschitz_constant = 1.1 * (
        4 * input_img.ndim * (1 - l1_ratio) ** 2 + l1_ratio**2
    )

    # negated_output is the negated primal variable in the optimization
    # loop
    negated_output = -input_img if init is None else -init

    # Clipping values for the inner loop
    negated_val_min = np.inf
    negated_val_max = -np.inf
    if val_min is not None:
        negated_val_min = -val_min
    if val_max is not None:
        negated_val_max = -val_max
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
        grad_tmp *= 1.0 / (lipschitz_constant * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_tvl1_dual(grad_aux, l1_ratio)

        # Careful, in the next few lines, grad_tmp and grad_aux are a
        # view on the same array, as _projector_on_tvl1_dual returns a view
        # on the input array
        t_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
        t_factor = (t - 1.0) / t_new

        grad_aux = grad_tmp
        if fista_step:
            grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im

        grad_im = grad_tmp
        t = t_new
        gap = weight * divergence_id(grad_aux, l1_ratio=l1_ratio)

        # Compute the primal variable
        negated_output = gap - input_img
        if val_min is not None or val_max is not None:
            negated_output = negated_output.clip(
                negated_val_max, negated_val_min, out=negated_output
            )
        if (i % check_gap_frequency) == 0:
            if x_tol is None:
                # Stopping criterion based on the dual gap
                if val_min is not None or val_max is not None:
                    # We need to recompute the dual variable
                    gap = negated_output + input_img
                old_dgap = dgap
                dgap = _dual_gap_prox_tvl1(
                    input_img_norm,
                    -negated_output,
                    gap,
                    weight,
                    l1_ratio=l1_ratio,
                )

                logger.log(
                    f"\tProxTVl1: Iteration {i: 2}, "
                    f"dual gap: {dgap: 6.3e}",
                    verbose,
                )

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

                gid = gradient_id(negated_output, l1_ratio=l1_ratio)
                energy = _objective_function_prox_tvl1(
                    input_img, -negated_output, gid, weight
                )
                logger.log(
                    f"\tProxTVl1 iteration {i: 2}, "
                    f"relative difference: {diff: 6.3e}, "
                    f"energy: {energy: 6.3e}",
                    verbose,
                )

                if diff < x_tol:
                    break
                negated_output_old = negated_output
        i += 1

    # Compute the primal variable, however, here we must use the ista
    # value, not the fista one
    output = input_img - weight * divergence_id(grad_im, l1_ratio=l1_ratio)
    if val_min is not None or val_max is not None:
        output = output.clip(val_min, val_max, out=output)
    return output, {"converged": (i < max_iter)}


def prox_tvl1_with_intercept(
    w,
    shape,
    l1_ratio,
    weight,
    dgap_tol,
    max_iter=5000,
    init=None,
    verbose=0,
):
    """Compute TV-L1 prox taking into account the intercept.

    Parameters
    ----------
    weight : float
       Weight in prox. This would be something like `alpha_ * stepsize`,
       where `alpha_` is the effective (i.e. re-scaled) alpha.

    w : ndarray, shape (w_size,)
        The point at which the prox is being computed

    init : ndarray, shape (w_size - 1,), optional (default None)
        Initialization vector for the prox.

    max_iter : int
        Maximum number of iterations for the solver.

    verbose : int or bool, optional
        If True or 1, print the dual gap of the optimization

    dgap_tol : float
        Dual-gap tolerance for TV-L1 prox operator approximation loop.

    """
    if verbose is False:
        verbose = 0
    if verbose is True:
        verbose = 1

    init = init.reshape(shape) if init is not None else init
    out, prox_info = prox_tvl1(
        w[:-1].reshape(shape),
        weight=weight,
        l1_ratio=l1_ratio,
        dgap_tol=dgap_tol,
        init=init,
        max_iter=max_iter,
        verbose=verbose,
    )

    return np.append(out, w[-1]), prox_info
