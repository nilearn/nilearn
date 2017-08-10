import copy
import warnings
from math import sqrt, exp, log, cosh, sinh

import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_true
from sklearn.utils import check_random_state
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from nilearn._utils.extmath import is_spd
from nilearn._utils.testing import assert_raises_regex
from nilearn.tests.test_signal import generate_signals
from nilearn.connectome.connectivity_matrices import (
    _check_square, _check_spd, _map_eigenvalues, _form_symmetric,
    _geometric_mean, sym_matrix_to_vec, vec_to_sym_matrix, prec_to_partial,
    cov_to_corr, ConnectivityMeasure)


def grad_geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """Return the norm of the covariant derivative at each iteration step of
    geometric_mean. See its docstring for details.

    Norm is intrinsic norm on the tangent space of the manifold of symmetric
    positive definite matrices.

    Returns
    -------
    grad_norm : list of float
        Norm of the covariant derivative in the tangent space at each step.
    """
    mats = np.array(mats)

    # Initialization
    if init is None:
        gmean = np.mean(mats, axis=0)
    else:
        gmean = init
    norm_old = np.inf
    step = 1.
    grad_norm = []
    for n in range(max_iter):
        # Computation of the gradient
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_gmean, vecs_gmean)
        whitened_mats = [gmean_inv_sqrt.dot(mat).dot(gmean_inv_sqrt)
                         for mat in mats]
        logs = [_map_eigenvalues(np.log, w_mat) for w_mat in whitened_mats]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - gmean.dot(logms_mean)
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point gmean

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        gmean = gmean_sqrt.dot(
            _form_symmetric(np.exp, vals_log * step, vecs_log)).dot(gmean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old

        grad_norm.append(norm / gmean.size)
        if tol is not None and norm / gmean.size < tol:
            break

    return grad_norm


def test_check_square():
    non_square = np.ones((2, 3))
    assert_raises(ValueError, _check_square, non_square)


def test_check_spd():
    non_sym = np.array([[0, 1], [0, 0]])
    assert_raises(ValueError, _check_spd, non_sym)

    non_spd = np.ones((3, 3))
    assert_raises(ValueError, _check_spd, non_spd)


def test_map_eigenvalues():
    # Test on exp map
    sym = np.ones((2, 2))
    sym_exp = exp(1.) * np.array([[cosh(1.), sinh(1.)], [sinh(1.), cosh(1.)]])
    assert_array_almost_equal(_map_eigenvalues(np.exp, sym), sym_exp)

    # Test on sqrt map
    spd_sqrt = np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]])
    spd = spd_sqrt.dot(spd_sqrt)
    assert_array_almost_equal(_map_eigenvalues(np.sqrt, spd), spd_sqrt)

    # Test on log map
    spd = np.array([[1.25, 0.75], [0.75, 1.25]])
    spd_log = np.array([[0., log(2.)], [log(2.), 0.]])
    assert_array_almost_equal(_map_eigenvalues(np.log, spd), spd_log)


def test_geometric_mean_couple():
    n_features = 7
    spd1 = np.ones((n_features, n_features))
    spd1 = spd1.dot(spd1) + n_features * np.eye(n_features)
    spd2 = np.tril(np.ones((n_features, n_features)))
    spd2 = spd2.dot(spd2.T)
    vals_spd2, vecs_spd2 = np.linalg.eigh(spd2)
    spd2_sqrt = _form_symmetric(np.sqrt, vals_spd2, vecs_spd2)
    spd2_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_spd2, vecs_spd2)
    geo = spd2_sqrt.dot(_map_eigenvalues(np.sqrt, spd2_inv_sqrt.dot(spd1).dot(
                        spd2_inv_sqrt))).dot(spd2_sqrt)
    assert_array_almost_equal(_geometric_mean([spd1, spd2]), geo)


def test_geometric_mean_diagonal():
    n_matrices = 20
    n_features = 5
    diags = []
    for k in range(n_matrices):
        diag = np.eye(n_features)
        diag[k % n_features, k % n_features] = 1e4 + k
        diag[(n_features - 1) // (k + 1), (n_features - 1) // (k + 1)] = \
            (k + 1) * 1e-4
        diags.append(diag)
    geo = np.prod(np.array(diags), axis=0) ** (1 / float(len(diags)))
    assert_array_almost_equal(_geometric_mean(diags), geo)


def test_geometric_mean_geodesic():
    n_matrices = 10
    n_features = 6
    sym = np.arange(n_features) / np.linalg.norm(np.arange(n_features))
    sym = sym * sym[:, np.newaxis]
    times = np.arange(n_matrices)
    non_singular = np.eye(n_features)
    non_singular[1:3, 1:3] = np.array([[-1, -.5], [-.5, -1]])
    spds = []
    for time in times:
        spds.append(non_singular.dot(_map_eigenvalues(np.exp, time * sym)).dot(
            non_singular.T))
    gmean = non_singular.dot(_map_eigenvalues(np.exp, times.mean() * sym)).dot(
        non_singular.T)
    assert_array_almost_equal(_geometric_mean(spds), gmean)


def random_diagonal(p, v_min=1., v_max=2., random_state=0):
    """Generate a random diagonal matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    v_min : float, optional (default to 1.)
        Minimal element.

    v_max : float, optional (default to 2.)
        Maximal element.

    random_state : int or numpy.random.RandomState instance, optional
        random number generator, or seed.

    Returns
    -------
    output : numpy.ndarray, shape (p, p)
        A diagonal matrix with the given minimal and maximal elements.

    """
    random_state = check_random_state(random_state)
    diag = random_state.rand(p) * (v_max - v_min) + v_min
    diag[diag == np.amax(diag)] = v_max
    diag[diag == np.amin(diag)] = v_min
    return np.diag(diag)


def random_spd(p, eig_min, cond, random_state=0):
    """Generate a random symmetric positive definite matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    eig_min : float
        Minimal eigenvalue.

    cond : float
        Condition number, defined as the ratio of the maximum eigenvalue to the
        minimum one.

    random_state : int or numpy.random.RandomState instance, optional
        random number generator, or seed.

    Returns
    -------
    ouput : numpy.ndarray, shape (p, p)
        A symmetric positive definite matrix with the given minimal eigenvalue
        and condition number.
    """
    random_state = check_random_state(random_state)
    mat = random_state.randn(p, p)
    unitary, _ = linalg.qr(mat)
    diag = random_diagonal(p, v_min=eig_min, v_max=cond * eig_min,
                           random_state=random_state)
    return unitary.dot(diag).dot(unitary.T)


def random_non_singular(p, sing_min=1., sing_max=2., random_state=0):
    """Generate a random nonsingular matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    sing_min : float, optional (default to 1.)
        Minimal singular value.

    sing_max : float, optional (default to 2.)
        Maximal singular value.

    random_state : int or numpy.random.RandomState instance, optional
        random number generator, or seed.

    Returns
    -------
    output : numpy.ndarray, shape (p, p)
        A nonsingular matrix with the given minimal and maximal singular
        values.
    """
    random_state = check_random_state(random_state)
    diag = random_diagonal(p, v_min=sing_min, v_max=sing_max,
                           random_state=random_state)
    mat1 = random_state.randn(p, p)
    mat2 = random_state.randn(p, p)
    unitary1, _ = linalg.qr(mat1)
    unitary2, _ = linalg.qr(mat2)
    return unitary1.dot(diag).dot(unitary2.T)


def test_geometric_mean_properties():
    n_matrices = 40
    n_features = 15
    spds = []
    for k in range(n_matrices):
        spds.append(random_spd(n_features, eig_min=1., cond=10.,
                               random_state=0))
    input_spds = copy.copy(spds)
    gmean = _geometric_mean(spds)

    # Generic
    assert_true(isinstance(spds, list))
    for spd, input_spd in zip(spds, input_spds):
        assert_array_equal(spd, input_spd)
    assert(is_spd(gmean, decimal=7))

    # Invariance under reordering
    spds.reverse()
    spds.insert(0, spds[1])
    spds.pop(2)
    assert_array_almost_equal(_geometric_mean(spds), gmean)

    # Invariance under congruent transformation
    non_singular = random_non_singular(n_features, random_state=0)
    spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
    assert_array_almost_equal(_geometric_mean(spds_cong),
                              non_singular.dot(gmean).dot(non_singular.T))

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    init = linalg.inv(np.mean(spds, axis=0))
    assert_array_almost_equal(_geometric_mean(spds_inv, init=init),
                              linalg.inv(gmean))

    # Gradient norm is decreasing
    grad_norm = grad_geometric_mean(spds, tol=1e-20)
    difference = np.diff(grad_norm)
    assert_true(np.amax(difference) <= 0.)

    # Check warning if gradient norm in the last step is less than
    # tolerance
    max_iter = 1
    tol = 1e-20
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gmean = _geometric_mean(spds, max_iter=max_iter, tol=tol)
        assert_equal(len(w), 1)
    grad_norm = grad_geometric_mean(spds, max_iter=max_iter, tol=tol)
    assert_equal(len(grad_norm), max_iter)
    assert_true(grad_norm[-1] > tol)

    # Evaluate convergence. A warning is printed if tolerance is not reached
    for p in [.5, 1.]:  # proportion of badly conditionned matrices
        spds = []
        for k in range(int(p * n_matrices)):
            spds.append(random_spd(n_features, eig_min=1e-2, cond=1e6,
                                   random_state=0))
        for k in range(int(p * n_matrices), n_matrices):
            spds.append(random_spd(n_features, eig_min=1., cond=10.,
                                   random_state=0))
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        gmean = _geometric_mean(spds, max_iter=max_iter, tol=1e-5)


def test_geometric_mean_errors():
    n_features = 5

    # Non square input matrix
    mat1 = np.ones((n_features, n_features + 1))
    assert_raises(ValueError, _geometric_mean, [mat1])

    # Input matrices of different shapes
    mat1 = np.eye(n_features)
    mat2 = np.ones((n_features + 1, n_features + 1))
    assert_raises(ValueError, _geometric_mean, [mat1, mat2])

    # Non spd input matrix
    assert_raises(ValueError, _geometric_mean, [mat2])


def test_sym_matrix_to_vec():
    sym = np.ones((3, 3))
    sqrt2 = 1. / sqrt(2.)
    vec = np.array([sqrt2, 1., sqrt2, 1., 1., sqrt2])
    assert_array_almost_equal(sym_matrix_to_vec(sym), vec)

    vec = np.array([1., 1., 1.])
    assert_array_almost_equal(sym_matrix_to_vec(sym, discard_diagonal=True),
                              vec)

    # Check sym_matrix_to_vec is the inverse function of vec_to_sym_matrix
    n = 5
    p = n * (n + 1) // 2
    rand_gen = np.random.RandomState(0)
    # when diagonal is included
    vec = rand_gen.rand(p)
    sym = vec_to_sym_matrix(vec)
    assert_array_almost_equal(sym_matrix_to_vec(sym), vec)

    # when diagonal given separately
    diagonal = rand_gen.rand(n + 1)
    sym = vec_to_sym_matrix(vec, diagonal=diagonal)
    assert_array_almost_equal(sym_matrix_to_vec(sym, discard_diagonal=True),
                              vec)

    # multiple matrices case when diagonal is included
    vecs = np.asarray([vec, 2. * vec, 0.5 * vec])
    syms = vec_to_sym_matrix(vecs)
    assert_array_almost_equal(sym_matrix_to_vec(syms), vecs)

    # multiple matrices case when diagonal is given seperately
    diagonals = np.asarray([diagonal, 3. * diagonal, -diagonal])
    syms = vec_to_sym_matrix(vecs, diagonal=diagonals)
    assert_array_almost_equal(sym_matrix_to_vec(syms, discard_diagonal=True),
                              vecs)


def test_vec_to_sym_matrix():
    # Check error if unsuitable size
    vec = np.ones(31)
    assert_raises_regex(ValueError, 'Vector of unsuitable shape',
                        vec_to_sym_matrix, vec)

    # Check error if given diagonal shape incompatible with vec
    vec = np.ones(3)
    diagonal = np.zeros(4)
    assert_raises_regex(ValueError, 'incompatible with vector',
                        vec_to_sym_matrix, vec, diagonal)

    # Check output value is correct
    vec = np.ones(6, )
    sym = np.array([[sqrt(2), 1., 1.], [1., sqrt(2), 1.],
                    [1., 1., sqrt(2)]])
    assert_array_almost_equal(vec_to_sym_matrix(vec), sym)

    # Check output value is correct with seperate diagonal
    vec = np.ones(3, )
    diagonal = np.ones(3)
    assert_array_almost_equal(vec_to_sym_matrix(vec, diagonal=diagonal), sym)

    # Check vec_to_sym_matrix is the inverse function of sym_matrix_to_vec
    # when diagonal is included
    assert_array_almost_equal(vec_to_sym_matrix(sym_matrix_to_vec(sym)), sym)

    # when diagonal is discarded
    vec = sym_matrix_to_vec(sym, discard_diagonal=True)
    diagonal = np.diagonal(sym) / sqrt(2)
    assert_array_almost_equal(vec_to_sym_matrix(vec, diagonal=diagonal), sym)


def test_prec_to_partial():
    prec = np.array([[2., -1., 1.], [-1., 2., -1.], [1., -1., 1.]])
    partial = np.array([[1., .5, -sqrt(2.) / 2.], [.5, 1., sqrt(2.) / 2.],
                        [-sqrt(2.) / 2., sqrt(2.) / 2., 1.]])
    assert_array_almost_equal(prec_to_partial(prec), partial)


def test_connectivity_measure_errors():
    # Raising error for input subjects not iterable
    conn_measure = ConnectivityMeasure()
    assert_raises(ValueError, conn_measure.fit, 1.)

    # Raising error for input subjects not 2D numpy.ndarrays
    assert_raises(ValueError, conn_measure.fit, [np.ones((100, 40)),
                                                 np.ones((10,))])

    # Raising error for input subjects with different number of features
    assert_raises(ValueError, conn_measure.fit,
                  [np.ones((100, 40)), np.ones((100, 41))])


def test_connectivity_measure_outputs():
    n_subjects = 10
    n_features = 49

    # Generate signals and compute covariances
    emp_covs = []
    ledoit_covs = []
    signals = []
    ledoit_estimator = LedoitWolf()
    for k in range(n_subjects):
        n_samples = 200 + k
        signal, _, _ = generate_signals(n_features=n_features, n_confounds=5,
                                        length=n_samples, same_variance=False)
        signals.append(signal)
        signal -= signal.mean(axis=0)
        emp_covs.append((signal.T).dot(signal) / n_samples)
        ledoit_covs.append(ledoit_estimator.fit(signal).covariance_)

    kinds = ["covariance", "correlation", "tangent", "precision",
             "partial correlation"]

    # Check outputs properties
    for cov_estimator, covs in zip([EmpiricalCovariance(), LedoitWolf()],
                                   [emp_covs, ledoit_covs]):
        input_covs = copy.copy(covs)
        for kind in kinds:
            conn_measure = ConnectivityMeasure(kind=kind,
                                               cov_estimator=cov_estimator)
            connectivities = conn_measure.fit_transform(signals)

            # Generic
            assert_true(isinstance(connectivities, np.ndarray))
            assert_equal(len(connectivities), len(covs))

            for k, cov_new in enumerate(connectivities):
                assert_array_equal(input_covs[k], covs[k])
                assert(is_spd(covs[k], decimal=7))

                # Positive definiteness if expected and output value checks
                if kind == "tangent":
                    assert_array_almost_equal(cov_new, cov_new.T)
                    gmean_sqrt = _map_eigenvalues(np.sqrt,
                                                  conn_measure.mean_)
                    assert(is_spd(gmean_sqrt, decimal=7))
                    assert(is_spd(conn_measure.whitening_, decimal=7))
                    assert_array_almost_equal(conn_measure.whitening_.dot(
                        gmean_sqrt), np.eye(n_features))
                    assert_array_almost_equal(gmean_sqrt.dot(
                        _map_eigenvalues(np.exp, cov_new)).dot(gmean_sqrt),
                        covs[k])
                elif kind == "precision":
                    assert(is_spd(cov_new, decimal=7))
                    assert_array_almost_equal(cov_new.dot(covs[k]),
                                              np.eye(n_features))
                elif kind == "correlation":
                    assert(is_spd(cov_new, decimal=7))
                    d = np.sqrt(np.diag(np.diag(covs[k])))
                    if cov_estimator == EmpiricalCovariance():
                        assert_array_almost_equal(d.dot(cov_new).dot(d),
                                                  covs[k])
                    assert_array_almost_equal(np.diag(cov_new),
                                              np.ones((n_features)))
                elif kind == "partial correlation":
                    prec = linalg.inv(covs[k])
                    d = np.sqrt(np.diag(np.diag(prec)))
                    assert_array_almost_equal(d.dot(cov_new).dot(d), -prec +
                                              2 * np.diag(np.diag(prec)))

    # Check the mean_
    for kind in kinds:
        conn_measure = ConnectivityMeasure(kind=kind)
        conn_measure.fit_transform(signals)
        assert_equal((conn_measure.mean_).shape, (n_features, n_features))

    # Check vectorization option
    for kind in kinds:
        conn_measure = ConnectivityMeasure(kind=kind)
        connectivities = conn_measure.fit_transform(signals)
        conn_measure = ConnectivityMeasure(vectorize=True, kind=kind)
        vectorized_connectivities = conn_measure.fit_transform(signals)
        assert_array_almost_equal(vectorized_connectivities,
                                  sym_matrix_to_vec(connectivities))

    # Check not fitted error
    assert_raises_regex(
        ValueError, 'has not been fitted. ',
        ConnectivityMeasure().inverse_transform,
        vectorized_connectivities)

    # Check inverse transformation
    kinds.remove('tangent')
    for kind in kinds:
        # without vectorization: input matrices are returned with no change
        conn_measure = ConnectivityMeasure(kind=kind)
        connectivities = conn_measure.fit_transform(signals)
        assert_array_almost_equal(
            conn_measure.inverse_transform(connectivities), connectivities)

        # with vectorization: input vectors are reshaped into matrices
        # if diagonal has not been discarded
        conn_measure = ConnectivityMeasure(kind=kind, vectorize=True)
        vectorized_connectivities = conn_measure.fit_transform(signals)
        assert_array_almost_equal(
            conn_measure.inverse_transform(vectorized_connectivities),
            connectivities)

    # with vectorization if diagonal has been discarded
    for kind in ['correlation', 'partial correlation']:
        connectivities = ConnectivityMeasure(kind=kind).fit_transform(signals)
        conn_measure = ConnectivityMeasure(kind=kind, vectorize=True,
                                           discard_diagonal=True)
        vectorized_connectivities = conn_measure.fit_transform(signals)
        assert_array_almost_equal(
            conn_measure.inverse_transform(vectorized_connectivities),
            connectivities)

    for kind in ['covariance', 'precision']:
        connectivities = ConnectivityMeasure(kind=kind).fit_transform(signals)
        conn_measure = ConnectivityMeasure(kind=kind, vectorize=True,
                                           discard_diagonal=True)
        vectorized_connectivities = conn_measure.fit_transform(signals)
        diagonal = np.array([np.diagonal(conn) / sqrt(2) for conn in
                             connectivities])
        inverse_transformed = conn_measure.inverse_transform(
            vectorized_connectivities, diagonal=diagonal)
        assert_array_almost_equal(inverse_transformed, connectivities)
        assert_raises_regex(ValueError,
                            'can not reconstruct connectivity matrices',
                            conn_measure.inverse_transform,
                            vectorized_connectivities)

    # for 'tangent' kind, covariance matrices are reconstructed
    # without vectorization
    tangent_measure = ConnectivityMeasure(kind='tangent')
    displacements = tangent_measure.fit_transform(signals)
    covariances = ConnectivityMeasure(kind='covariance').fit_transform(
        signals)
    assert_array_almost_equal(
        tangent_measure.inverse_transform(displacements), covariances)

    # with vectorization
    # when diagonal has not been discarded
    tangent_measure = ConnectivityMeasure(kind='tangent', vectorize=True)
    vectorized_displacements = tangent_measure.fit_transform(signals)
    assert_array_almost_equal(
        tangent_measure.inverse_transform(vectorized_displacements),
        covariances)

    # when diagonal has been discarded
    tangent_measure = ConnectivityMeasure(kind='tangent', vectorize=True,
                                          discard_diagonal=True)
    vectorized_displacements = tangent_measure.fit_transform(signals)
    diagonal = np.array([np.diagonal(matrix) / sqrt(2) for matrix in
                         displacements])
    inverse_transformed = tangent_measure.inverse_transform(
        vectorized_displacements, diagonal=diagonal)
    assert_array_almost_equal(inverse_transformed, covariances)
    assert_raises_regex(ValueError,
                        'can not reconstruct connectivity matrices',
                        tangent_measure.inverse_transform,
                        vectorized_displacements)
