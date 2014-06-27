# XXX This code is duplicated from common.py!

import numpy as np


def div(grad):
    """ Compute divergence of image gradient """
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def grad(img):
    """ Compute gradient of an image

        Parameters
        ===========
        img: ndarray
            N-dimensional image

        Returns
        =======
        gradient: ndarray
            Image of the gradient: the i-th component along the first
            axis is the gradient along the i-th axis of the original
            array img
    """
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    # 'Clever' code to have a view of the gradient with dimension i stop
    # at -1
    slice_all = [0, slice(None, -1), ]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient
