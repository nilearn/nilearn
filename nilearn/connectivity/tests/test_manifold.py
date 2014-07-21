import warnings
import copy

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, \
    assert_array_less
from nose.tools import assert_tuple_equal, assert_equal, assert_not_equal, \
    assert_is_instance, assert_raises
from scipy import linalg

from ..._utils.testing import is_spd
from ..manifold import random_diagonal_spd, random_spd, random_non_singular, \
    inv, sqrtm, inv_sqrtm, logm, expm, stack_newaxis, geometric_mean, \
    grad_geometric_mean


def test_random_diagonal_spd():
    """Testing random_diagonal_spd function"""
    d = random_diagonal_spd(15, eig_min=1e-2, eig_max=1e3)
    diag = np.diag(d)
    assert_array_almost_equal(d, np.diag(diag))
    assert_array_less(0.0, diag)


def test_random_spd():
    """Testing random_spd function"""
    spd = random_spd(17, eig_min=1e-3, eig_max=1e2)
    assert(is_spd(spd))


def test_random_non_singular():
    """Testing random_non_singular function"""
    non_sing = random_non_singular(23)
    assert_not_equal(linalg.det(non_sing), 0.)


def test_inv():
    """Testing inv function"""
    m = random_spd(41)
    m_inv = inv(m)
    assert_array_almost_equal(m.dot(m_inv), np.eye(41))


def test_sqrtm():
    """Testing sqrtm function"""
    m = random_spd(12)
    m_sqrt = sqrtm(m)
    assert(is_spd(m_sqrt))
    assert_array_almost_equal(m_sqrt.dot(m_sqrt), m)


def test_expm():
    """Testing expm function"""
    m = np.random.rand(15, 15)
    m = m + m.T
    m_exp = expm(m)
    assert(is_spd(m_exp))
    assert_array_almost_equal(linalg.logm(m_exp), m)


def test_logm():
    """Testing logm function"""
    m = random_spd(11)
    m_log = logm(m)
    assert_array_almost_equal(linalg.expm(m_log), m)


def test_inv_sqrtm():
    """Testing inv_sqrtm function"""
    m = random_spd(21)
    m_inv_sqrt = inv_sqrtm(m)
    assert(is_spd(m_inv_sqrt))
    assert_array_almost_equal(m_inv_sqrt.dot(m_inv_sqrt).dot(m), np.eye(21))


def test_stack_newaxis():
    """Testing stack_newaxis function"""
    mats = []
    for n in xrange(7):
        mats.append(np.random.rand(31, 31))
    stacked = stack_newaxis(mats)
    assert_is_instance(stacked, np.ndarray)
    assert_tuple_equal(stacked.shape, (7, 31, 31))
    for n in xrange(7):
        assert_array_equal(stacked[n], mats[n])


def test_geometric_mean_couple():
    """Testing geometric_mean function for two matrices"""
    n_features = 7
    spd1 = np.ones((n_features, n_features))
    spd1 = spd1.dot(spd1) + n_features * np.eye(n_features)
    spd2 = np.tril(np.ones((n_features, n_features)))
    spd2 = spd2.dot(spd2.T)
    spd2_sqrt = sqrtm(spd2)
    spd2_inv_sqrt = inv_sqrtm(spd2)
    exact_geo = spd2_sqrt.dot(sqrtm(
        spd2_inv_sqrt.dot(spd1).dot(spd2_inv_sqrt))).dot(spd2_sqrt)
    geo = geometric_mean([spd1, spd2])
    assert_array_almost_equal(geo, exact_geo)


def test_geometric_mean_diagonal():
    """Testing geometric_mean function for diagonal matrices"""
    n_matrices = 20
    n_features = 5
    diags = []
    for k in xrange(n_matrices):
        diag = np.eye(n_features)
        diag[k % n_features, k % n_features] = 1e4 + k
        diag[(n_features - 1) // (k + 1), (n_features - 1) // (k + 1)] = \
            (k + 1) * 1e-4
        diags.append(diag)
    exact_geo = np.prod(stack_newaxis(diags), axis=0) ** \
        (1 / float(len(diags)))
    geo = geometric_mean(diags)
    assert_array_almost_equal(geo, exact_geo)


def test_geometric_mean_geodesic():
    """Testing geometric_mean function for single geodesic matrices"""
    n_matrices = 10
    n_features = 6
    sym = np.arange(n_features) / np.linalg.norm(np.arange(n_features))
    sym = sym * sym[:, np.newaxis]
    times = np.arange(n_matrices) / 2. - 10.
    non_singular = np.eye(n_features)
    non_singular[1:3, 1:3] = np.array([[-1, -.5], [-.5, -1]])
    spds = []
    for time in times:
        spds.append(non_singular.dot(expm(time * sym)).dot(
            non_singular.T))
    exact_geo = non_singular.dot(expm(times.mean() * sym)).dot(
        non_singular.T)
    geo = geometric_mean(spds)
    assert_array_almost_equal(geo, exact_geo)


def test_geometric_mean_properties():
    """Testing geometric_mean function for random spd matrices
    """
    n_matrices = 35
    n_features = 15
    warnings.simplefilter('always', UserWarning)
    for p in [0, .5, 1]:  # proportion of badly conditionned matrices
        spds = []
        for k in xrange(int(p * n_matrices)):
            spds.append(random_spd(n_features, eig_min=1e-4,
                                            eig_max=1e4))
        for k in xrange(int(p * n_matrices), n_matrices):
            spds.append(random_spd(n_features, eig_min=1.,
                                            eig_max=10.))

        input_spds = copy.copy(spds)
        geo = geometric_mean(spds)

        # Generic
        assert(isinstance(spds, list))
        for spd, input_spd in zip(spds, input_spds):
            assert_array_equal(spd, input_spd)
        assert(is_spd(geo))

        # Invariance under reordering
        spds.reverse()
        spds.insert(0, spds[1])
        spds.pop(2)
        geo_new = geometric_mean(spds)
        assert_array_almost_equal(geo_new, geo)

        # Invariance under congruant transformation
        non_singular = random_non_singular(n_features)
        spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
        geo_new = geometric_mean(spds_cong)
        assert_array_almost_equal(geo_new,
                                  non_singular.dot(geo).dot(non_singular.T))

        # Invariance under inversion
        spds_inv = [linalg.inv(spd) for spd in spds]
        init = linalg.inv(np.mean(spds, axis=0))
        geo_new = geometric_mean(spds_inv, init=init)
        assert_array_almost_equal(geo_new, linalg.inv(geo))

        # Gradient norm is decreasing
        grad_norm = grad_geometric_mean(spds)
        difference = np.diff(grad_norm)
        assert(not(np.amax(difference) > 0.))

        # Check warning if gradient norm in the last step is less than
        # tolerance
        max_iter = 1
        tol = 1e-10
        with warnings.catch_warnings(record=True) as w:
            geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
            grad_norm = grad_geometric_mean(spds, max_iter=max_iter,
                                                     tol=tol)
        assert(grad_norm[-1] > tol)
        assert_equal(len(grad_norm), max_iter)
        assert_equal(len(w), 1)


def test_geometric_mean_checks():
    n_features = 5

    # Non square input matrix
    mat1 = np.ones((n_features, n_features + 1))
    assert_raises(ValueError, geometric_mean, [mat1])

    # Input matrices of different shapes
    mat1 = np.eye(n_features)
    mat2 = np.ones((n_features + 1, n_features + 1))
    assert_raises(ValueError, geometric_mean, [mat1, mat2])

    # Non spd input matrix
    assert_raises(ValueError, geometric_mean, [mat2])