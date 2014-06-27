"""Implementations of multiple proximal operators
"""

import numpy as np
from scipy import linalg
import image
from optim import fmin_prox

###############################################################################
# Mixed norms

def prox_l1(y, alpha, copy=True):
    """proximity operator for l1 norm"""
    shrink = np.zeros(y.shape)
    if copy:
        y = y.copy()
    y_nz = y.nonzero()
    shrink[y_nz] = np.maximum(1 - alpha / np.abs(y[y_nz]), 0)
    y *= shrink
    return y


# def prox_l1_l2(y, alpha, beta, copy=True):
#     """proximity operator for l1 norm + l2 norm (a.k.a. Elastic-Net)
#
#     Penalty = alpha |y|_1 + 0.5 * beta |y|^2_2
#
#     """
#     shrink = np.zeros(y.shape)
#     if copy:
#         y = y.copy()
#     y_nz = y.nonzero()
#     shrink[y_nz] = np.maximum(1 - (alpha / (1 + beta)) / np.abs(y[y_nz]), 0)
#     y *= shrink
#     return y

def prox_l21(Y, alpha, axis=1, copy=True):
    """proximity operator for l21 norm"""
    shrink = np.zeros(Y.shape[(axis + 1) % 2], 1)
    if copy:
        Y = Y.copy()
    l2_norms = np.sqrt(np.sum(Y**2, axis=axis))
    nz = l2_norms.nonzero()
    shrink[nz] = np.maximum(1 - alpha / l2_norms[nz], 0)
    Y *= shrink
    return Y

###############################################################################
# Smooth Lasso

def estimate_lipschitz_constant_graph(w0, L):
    """Compute approximate lipschitz constant
    of callable linear operator : x -> Lx
    using a power method"""
    a = np.random.randn(*w0.shape)
    a /= linalg.norm(a)
    for i in range(100):
        b = L(a)
        a = b / linalg.norm(b)

    lipschitz_constant = (b * a).sum()
    return 1.1 * lipschitz_constant


def prox_smooth_lasso_graph(y, L, alpha, beta, maxit=10, verbose=True, tol=0,
                        mode='f', lipschitz_constant=None):
    """Smooth Lasso regression model with L1 + L2 regularization.

    w,v = argmin    1/2 || y - w ||^2 + alpha ||w||_1
            w,v                            + beta/2 w'Lw

    where L is the graph laplacian (function that runs L(x) when called)

    w is the loadings vector
    """
    y = y.ravel()
    n_features = y.size

    # and the proximity operator for the non-smooth part
    prior = lambda w: alpha * linalg.norm(w, ord=1)
    prior_prox = lambda w, l: prox_l1(w, l * alpha)
    data_fit = lambda w: 0.5 * linalg.norm(y - w)**2 \
                            + 0.5 * beta * np.sum(w * L(w))
    data_fit_grad = lambda w: w - y + beta * L(w)
    dual_gap = None # XXX

    w0 = np.zeros(n_features)

    if lipschitz_constant is None:
        lc_L = estimate_lipschitz_constant_graph(w0, L)
        lc_L *= 1.1 # upper bound on lipshitz constant
        lipschitz_constant = 1.05 + beta * lc_L

    w, objective = fmin_prox(x0=w0, f1=data_fit,
                                f1_grad=data_fit_grad, f2=prior,
                                f2_prox=prior_prox,
                                lipschitz_constant=lipschitz_constant,
                                verbose=verbose, maxit=maxit, tol=tol,
                                mode=mode, dual_gap=dual_gap)

    return w, objective

if __name__ == '__main__':
    import pylab as pl
    import time

    ### Run optim
    maxit = 2000
    N = 32
    W_init = np.zeros((N, N))
    W_init[5:12, 5:12] = 1

    # Mask
    x, y = np.indices((N, N))
    distances = np.sqrt((x - 0.5*N)**2 + (y - 0.5*N)**2)
    mask = np.ones(W_init.shape, dtype=np.bool)
    mask[distances > int(N*0.6)] = False
    mask[distances < int(N*0.3)] = False

    N = 16
    W_init = W_init[:N, :N]
    mask = mask[:N, :N]

    nvoxels = mask.sum()

    np.random.seed(0)
    XX = np.random.randn(500, mask.shape[0], mask.shape[1])
    XX_m = XX[:, mask]
    Y = np.dot(XX_m, W_init[mask].ravel())
    Y += 0.01 * np.random.randn(Y.shape[0])

    pl.close('all')

    pl.figure()
    pl.imshow(mask)
    pl.title('Mask')

    pl.figure()
    pl.imshow(W_init)
    pl.title("Ground truth")

    ### Masked data
    alpha = 0.1
    time_ = time.time()
    rho = 0.5

    grad_ind = image.indices_for_grad(mask)
    div_ind = image.indices_for_div(mask)
    grad = lambda im: image.grad_from_indices(im, grad_ind)
    div = lambda im: image.div_from_indices(im, div_ind)
    L = lambda w: - div(grad(w)).ravel()

    Y_prox = W_init[mask].ravel()
    np.random.seed(0)
    Y_prox += 0.3 * np.random.randn(*Y_prox.shape)

    alpha = 0.1
    beta = 3

    W_mask, E = prox_smooth_lasso_graph(Y_prox, L, alpha, beta, maxit=1000)

    mask_time = time.time() - time_
    W = np.zeros_like(W_init)
    W[mask] = W_mask

    pl.figure()
    pl.imshow(W)
    pl.title("Mask - Restored Output")

    pl.figure()
    pl.plot(E)
    pl.title("Mask - Energy")
    pl.xlabel("Iteration number")
    pl.ylabel("Energy")

    pl.show()
