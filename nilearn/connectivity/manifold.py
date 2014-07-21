import warnings

import numpy as np
from scipy import linalg

from .._utils.testing import is_spd


def stack_newaxis(arrays):
    """Stack arrays in sequence along inserted new axis.

    Parameters
    ==========
    arrays : sequence of numpy.ndarrays
        Arrays to be stacked. The arrays must have the same shape.

    Returns
    =======
    stacked: numpy.ndarray
        The array formed by stacking the given arrays.
    """
    stacked = np.concatenate([a[np.newaxis] for a in arrays])
    return stacked


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be square rooted. Raise an error if the array is not
        square.

    Returns
    =======
    mat_sqrtm: (M, M) numpy.ndarray
        The symmetric matrix square root of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but result will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_sqrtm = np.dot(vecs * np.sqrt(vals), vecs.T)
    return mat_sqrtm


def inv(mat):
    """ Inverse of matrix, for symmetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be inverted. Raise an error if the array is not square.

    Returns
    =======
    mat_inv: (M, M) numpy.ndarray
        The inverse matrix of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but result will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_inv = np.dot(vecs / vals, vecs.T)
    return mat_inv


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be square rooted and inverted. Raise an error if the array
        is not square.

    Returns
    =======
    mat_inv_sqrtm: (M, M) numpy.ndarray
        The inverse matrix of the symmetric square root of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but result will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_inv_sqrtm = np.dot(vecs / np.sqrt(vals), vecs.T)
    return mat_inv_sqrtm


def logm(mat):
    """ Logarithm of matrix, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array whose logarithm to be computed. Raise an error if the array is
        not square.

    Returns
    =======
    mat_logm: (M, M) numpy.ndarray
        Matrix logatrithm of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but result will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_logm = np.dot(vecs * np.log(vals), vecs.T)
    return mat_logm


def expm(mat):
    """ Exponential of matrix, for real symmetric matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array whose exponential to be computed. Raise an error if the array
        is not square.

    Returns
    =======
    mat_exp: (M, M) numpy.ndarray
        Matrix exponential of mat.

    Note
    ====
    If input matrix is not real symmetric, no error is reported but result
    will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_exp = np.dot(vecs * np.exp(vals), vecs.T)
    return mat_exp


def geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """ Compute the geometric mean of a list of symmetric positive definite
    matrices.

    Minimization of the objective function by an intrinsic gradient descent in
    the manifold: moving from the current point geo to the next one is
    done along a short geodesic arc in the opposite direction of the covariant
    derivative of the objective function evaluated at point geo.

    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ==========
    mats: list of numpy.array
        List of symmetric positive definite matrices, same shape. Raise an
        error if the arrays are not all symmetric positive definite of the same
        shape.
    init: numpy.array or None, optional (default to None)
        Initialization matrix. Symmetric positive definite, same shape as
        elements of mats. Raise an error if the array is not symmetric positive
        definite of the same shape as the elements of mats. Set to arithmetic
        mean of mats if None.
    max_iter: int, optional (default to 10)
        Maximal number of iterations.
    tol: float, optional (default to 1e-7)
        Tolerance.

    Returns
    =======
    geo: numpy.array
        Geometric mean of the matrices.
    """
    # Shape and symmetry positive definiteness checks
    s = mats[0].shape[0]
    for mat in mats:
        if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
            raise ValueError('at least one array is not square')
        if mat.shape[0] != s:
            raise ValueError("matrices are not of the same shape")
        if not is_spd(mat):
            raise ValueError("at least one matrix is not spd")

    # Initialization
    mats = stack_newaxis(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        if init.ndim != 2 or init.shape[0] != init.shape[-1]:
            raise ValueError('initialization is not square')
        if init.shape[0] != s:
            raise ValueError("initialization has not the correct shape")
        if not is_spd(init):
            raise ValueError("initialization is not spd")
        geo = init

    tolerance_reached = False
    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in xrange(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = (vecs_geo / np.sqrt(vals_geo)).dot(vecs_geo.T)
        eighs = [linalg.eigh(geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)) for
                 mat in mats]
        logs = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        try:
            assert np.all(np.isfinite(logs_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = (vecs_geo * np.sqrt(vals_geo)).dot(vecs_geo.T)
        geo = geo_sqrt.dot(vecs_log * np.exp(vals_log * step)).dot(
            vecs_log.T).dot(geo_sqrt)  # Move along the geodesic with step size
                                       # step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / geo.size < tol:
            tolerance_reached = True
            break

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without" +\
                      " getting to the requested tolerance level.")

    return geo


def grad_geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """ Return at each iteration step of the geometric_mean algorithm the norm
    of the covariant derivative.

    Norm is intrinsic norm on the tangent space at the geometric mean for the
    current step.

    Parameters
    ==========
    mats: list of array
        List of symmetric positive definite matrices, same shape.
    init: numpy.array or None, optional (default to None)
        Initialization matrix. Symmetric positive definite, same shape as
        elements of mats. Set to arithmetic mean of mats if None.
    max_iter: int, optional (default to 30)
        Maximal number of iterations.
    tol: float, optional (default to 1e-7)
        Tolerance.

    Returns
    =======
    grad_norm: list of float
        Norm of the covariant derivative in the tangent space at each step.
    """
    mats = stack_newaxis(mats)

    # Initialization
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        if not is_spd(init):
            raise ValueError("initialization is not spd")
        geo = init
    norm_old = np.inf
    step = 1.
    grad_norm = []
    for n in xrange(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = (vecs_geo / np.sqrt(vals_geo)).dot(vecs_geo.T)
        eighs = [linalg.eigh(geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)) for
                 mat in mats]
        logs = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        try:
            assert np.all(np.isfinite(logs_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = (vecs_geo * np.sqrt(vals_geo)).dot(vecs_geo.T)
        geo = geo_sqrt.dot(vecs_log * np.exp(vals_log * step)).dot(
            vecs_log.T).dot(geo_sqrt)  # Move along the geodesic with step size
                                       # step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old

        grad_norm.append(norm / geo.size)
        if tol is not None and norm / geo.size < tol:
            break

    return grad_norm


def random_diagonal(s, eig_min=0., eig_max=1.):
    """Generate random diagonal matrix, with diagonal elements in given range.

    Parameters
    ==========
    s: int
        The first dimension of the array.
    eig_min: float, optional (default to 0.)
        Lower bound for the diagonal elements.
    eig_max: float, optional (default to 1.)
        Upper bound for the diagonal elements.

    Returns
    =======
    diag: numpy.ndarray
        2D output diaogonal array, shape (s, s).
    """
    diag = np.random.rand(s) * (eig_max - eig_min) + eig_min
    return np.diag(diag)


def random_diagonal_spd(s, eig_min=1., eig_max=2.):
    """Generate random positive definite diagonal matrix"""
    assert(eig_min > 0)
    assert(eig_max > 0)
    return random_diagonal(s, eig_min, eig_max)


def random_spd(s, eig_min=1.0, eig_max=2.0):
    """Generate random symmetric positive definite matrix with eigenvalues in
    a given range.

    Parameters
    ==========
    s: int
        The first dimension of the array.
    eig_min: float, optional (default to 1.)
        Lower bound for the eigenvalues.
    eig_max: float, optional (default to 2.)
        Upper bound for the eigenvalues.

    Returns
    =======
    numpy.ndarray
        2D output array, shape (s, s).
    """
    ran = np.random.rand(s, s)
    q, _ = linalg.qr(ran)
    d = random_diagonal_spd(s, eig_min, eig_max)
    return q.dot(d).dot(q.T)


def random_non_singular(s):
    """Generate random non singular matrix.

    Parameters
    ==========
    s: int
        The first dimension of the array.

    Returns
    =======
    numpy.ndarray
        2D output array, shape (s, s).
    """
    d = random_diagonal_spd(s)
    ran1 = np.random.rand(s, s)
    ran2 = np.random.rand(s, s)
    u, _ = linalg.qr(ran1)
    v, _ = linalg.qr(ran2)
    return u.dot(d).dot(v.T)