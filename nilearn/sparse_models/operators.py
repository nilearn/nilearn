"""Implementations of multiple proximal operators
"""

import numpy as np
from scipy import linalg


def prox_l1(y, alpha, copy=True):
    """proximity operator for l1 norm"""
    shrink = np.zeros(y.shape)
    if copy:
        y = y.copy()
    y_nz = y.nonzero()
    shrink[y_nz] = np.maximum(1 - alpha / np.abs(y[y_nz]), 0)
    y *= shrink
    return y


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


# def estimate_lipschitz_constant_graph(w0, L):
#     """Compute approximate lipschitz constant
#     of callable linear operator : x -> Lx
#     using a power method"""
#     a = np.random.randn(*w0.shape)
#     a /= linalg.norm(a)
#     for i in range(100):
#         b = L(a)
#         a = b / linalg.norm(b)

#     lipschitz_constant = (b * a).sum()
#     return 1.1 * lipschitz_constant
