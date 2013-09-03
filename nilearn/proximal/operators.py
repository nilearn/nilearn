"""Implementations of multiple proximal operators
"""

import numpy as np
from scipy import linalg

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
