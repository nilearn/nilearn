import copy
import warnings
from math import cosh, exp, log, sinh, sqrt

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas import DataFrame
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.extmath import is_spd
from nilearn.connectome.connectivity_matrices import (
    ConnectivityMeasure,
    _check_spd,
    _check_square,
    _form_symmetric,
    _geometric_mean,
    _map_eigenvalues,
    prec_to_partial,
    sym_matrix_to_vec,
    vec_to_sym_matrix,
)
from nilearn.tests.test_signal import generate_signals

CONNECTIVITY_KINDS = (
    "covariance",
    "correlation",
    "tangent",
    "precision",
    "partial correlation",
)

N_FEATURES = 49

N_SUBJECTS = 5


@pytest.mark.parametrize(
    "estimator",
    [EmpiricalCovariance(), LedoitWolf()],
)
def test_check_estimator_cov_estimator(estimator):
    """Check compliance with sklearn estimators."""
    sklearn_check_estimator(estimator)


extra_valid_checks = [
    "check_parameters_default_constructible",
    "check_no_attributes_set_in_init",
    "check_estimators_unfitted",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_transformers_unfitted",
    "check_fit1d",
    "check_transformer_n_iter",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    (
        check_estimator(
            estimator=[
                ConnectivityMeasure(cov_estimator=EmpiricalCovariance())
            ],
            extra_valid_checks=extra_valid_checks,
        )
    ),
)
def test_check_estimator_group_sparse_covariance(
    estimator,
    check,
    name,  # noqa: ARG001
):
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[ConnectivityMeasure(cov_estimator=EmpiricalCovariance())],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid_group_sparse_covariance(
    estimator,
    check,
    name,  # noqa: ARG001
):
    """Check compliance with sklearn estimators."""
    check(estimator)


def random_diagonal(p, v_min=1.0, v_max=2.0, random_state=0):
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
    random_state = np.random.default_rng(random_state)
    diag = random_state.random(p) * (v_max - v_min) + v_min
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
    output : numpy.ndarray, shape (p, p)
        A symmetric positive definite matrix with the given minimal eigenvalue
        and condition number.
    """
    rand_gen = np.random.default_rng(random_state)
    mat = rand_gen.standard_normal((p, p))
    unitary, _ = linalg.qr(mat)
    diag = random_diagonal(
        p, v_min=eig_min, v_max=cond * eig_min, random_state=random_state
    )
    return unitary.dot(diag).dot(unitary.T)


def _signals(n_subjects=N_SUBJECTS):
    """Generate signals and compute covariances \
    and apply confounds while computing covariances.
    """
    n_features = N_FEATURES
    signals = []
    for k in range(n_subjects):
        n_samples = 200 + k
        signal, _, confounds = generate_signals(
            n_features=n_features,
            n_confounds=5,
            length=n_samples,
            same_variance=False,
        )
        signals.append(signal)
        signal -= signal.mean(axis=0)
    return signals, confounds


@pytest.fixture
def signals():
    return _signals(N_SUBJECTS)[0]


@pytest.fixture
def signals_and_covariances(cov_estimator):
    signals, _ = _signals()
    emp_covs = []
    ledoit_covs = []
    ledoit_estimator = LedoitWolf()
    for k, signal_ in enumerate(signals):
        n_samples = 200 + k
        signal_ -= signal_.mean(axis=0)
        emp_covs.append((signal_.T).dot(signal_) / n_samples)
        ledoit_covs.append(ledoit_estimator.fit(signal_).covariance_)

    if isinstance(cov_estimator, LedoitWolf):
        return signals, ledoit_covs
    elif isinstance(cov_estimator, EmpiricalCovariance):
        return signals, emp_covs


def test_check_square():
    non_square = np.ones((2, 3))
    with pytest.raises(ValueError, match="Expected a square matrix"):
        _check_square(non_square)


@pytest.mark.parametrize(
    "invalid_input",
    [np.array([[0, 1], [0, 0]]), np.ones((3, 3))],  # non symmetric
)  # non SPD
def test_check_spd(invalid_input):
    with pytest.raises(
        ValueError, match="Expected a symmetric positive definite matrix."
    ):
        _check_spd(invalid_input)


def test_map_eigenvalues_on_exp_map():
    sym = np.ones((2, 2))
    sym_exp = exp(1.0) * np.array(
        [[cosh(1.0), sinh(1.0)], [sinh(1.0), cosh(1.0)]]
    )
    assert_array_almost_equal(_map_eigenvalues(np.exp, sym), sym_exp)


def test_map_eigenvalues_on_sqrt_map():
    spd_sqrt = np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    )
    spd = spd_sqrt.dot(spd_sqrt)
    assert_array_almost_equal(_map_eigenvalues(np.sqrt, spd), spd_sqrt)


def test_map_eigenvalues_on_log_map():
    spd = np.array([[1.25, 0.75], [0.75, 1.25]])
    spd_log = np.array([[0.0, log(2.0)], [log(2.0), 0.0]])
    assert_array_almost_equal(_map_eigenvalues(np.log, spd), spd_log)


def test_geometric_mean_couple():
    n_features = 7
    spd1 = np.ones((n_features, n_features))
    spd1 = spd1.dot(spd1) + n_features * np.eye(n_features)
    spd2 = np.tril(np.ones((n_features, n_features)))
    spd2 = spd2.dot(spd2.T)
    vals_spd2, vecs_spd2 = np.linalg.eigh(spd2)
    spd2_sqrt = _form_symmetric(np.sqrt, vals_spd2, vecs_spd2)
    spd2_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_spd2, vecs_spd2)
    geo = spd2_sqrt.dot(
        _map_eigenvalues(np.sqrt, spd2_inv_sqrt.dot(spd1).dot(spd2_inv_sqrt))
    ).dot(spd2_sqrt)

    assert_array_almost_equal(_geometric_mean([spd1, spd2]), geo)


def test_geometric_mean_diagonal():
    n_matrices = 20
    n_features = 5
    diags = []
    for k in range(n_matrices):
        diag = np.eye(n_features)
        diag[k % n_features, k % n_features] = 1e4 + k
        diag[(n_features - 1) // (k + 1), (n_features - 1) // (k + 1)] = (
            k + 1
        ) * 1e-4
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
    non_singular[1:3, 1:3] = np.array([[-1, -0.5], [-0.5, -1]])
    spds = [
        non_singular.dot(_map_eigenvalues(np.exp, time * sym)).dot(
            non_singular.T
        )
        for time in times
    ]
    gmean = non_singular.dot(_map_eigenvalues(np.exp, times.mean() * sym)).dot(
        non_singular.T
    )
    assert_array_almost_equal(_geometric_mean(spds), gmean)


def test_geometric_mean_properties():
    n_matrices = 40
    n_features = 15
    spds = [
        random_spd(n_features, eig_min=1.0, cond=10.0, random_state=0)
        for _ in range(n_matrices)
    ]
    input_spds = copy.copy(spds)

    gmean = _geometric_mean(spds)

    # Generic
    assert isinstance(spds, list)
    for spd, input_spd in zip(spds, input_spds):
        assert_array_equal(spd, input_spd)
    assert is_spd(gmean, decimal=7)


def random_non_singular(p, sing_min=1.0, sing_max=2.0, random_state=0):
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
    rand_gen = np.random.default_rng(random_state)
    diag = random_diagonal(
        p, v_min=sing_min, v_max=sing_max, random_state=random_state
    )
    mat1 = rand_gen.standard_normal((p, p))
    mat2 = rand_gen.standard_normal((p, p))
    unitary1, _ = linalg.qr(mat1)
    unitary2, _ = linalg.qr(mat2)
    return unitary1.dot(diag).dot(unitary2.T)


def test_geometric_mean_properties_check_invariance():
    n_matrices = 40
    n_features = 15
    spds = [
        random_spd(n_features, eig_min=1.0, cond=10.0, random_state=0)
        for _ in range(n_matrices)
    ]

    gmean = _geometric_mean(spds)

    # Invariance under reordering
    spds.reverse()
    spds.insert(0, spds[1])
    spds.pop(2)
    assert_array_almost_equal(_geometric_mean(spds), gmean)

    # Invariance under congruent transformation
    non_singular = random_non_singular(n_features, random_state=0)
    spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
    assert_array_almost_equal(
        _geometric_mean(spds_cong), non_singular.dot(gmean).dot(non_singular.T)
    )

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    init = linalg.inv(np.mean(spds, axis=0))
    assert_array_almost_equal(
        _geometric_mean(spds_inv, init=init), linalg.inv(gmean)
    )


def grad_geometric_mean(mats, init=None, max_iter=10, tol=1e-7):
    """Return the norm of the covariant derivative at each iteration step \
       of geometric_mean. See its docstring for details.

    Norm is intrinsic norm on the tangent space of the manifold of symmetric
    positive definite matrices.

    Returns
    -------
    grad_norm : list of float
        Norm of the covariant derivative in the tangent space at each step.
    """
    mats = np.array(mats)

    # Initialization
    gmean = init or np.mean(mats, axis=0)

    norm_old = np.inf
    step = 1.0
    grad_norm = []
    for _ in range(max_iter):
        # Computation of the gradient
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_gmean, vecs_gmean)
        whitened_mats = [
            gmean_inv_sqrt.dot(mat).dot(gmean_inv_sqrt) for mat in mats
        ]
        logs = [_map_eigenvalues(np.log, w_mat) for w_mat in whitened_mats]

        # Covariant derivative is - gmean.dot(logs_mean)
        logs_mean = np.mean(logs, axis=0)

        # Norm of the covariant derivative on
        # the tangent space at point gmean
        norm = np.linalg.norm(logs_mean)

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        gmean = gmean_sqrt.dot(
            _form_symmetric(np.exp, vals_log * step, vecs_log)
        ).dot(gmean_sqrt)

        # Update the norm and the step size
        norm_old = min(norm, norm_old)
        if norm > norm_old:
            step = step / 2.0
            norm = norm_old

        grad_norm.append(norm / gmean.size)
        if tol is not None and norm / gmean.size < tol:
            break

    return grad_norm


def test_geometric_mean_properties_check_gradient():
    n_matrices = 40
    n_features = 15
    spds = [
        random_spd(n_features, eig_min=1.0, cond=10.0, random_state=0)
        for _ in range(n_matrices)
    ]

    grad_norm = grad_geometric_mean(spds, tol=1e-20)

    # Gradient norm is decreasing
    difference = np.diff(grad_norm)
    assert np.amax(difference) <= 0.0

    # Check warning if gradient norm in the last step is less than
    # tolerance
    max_iter = 1
    tol = 1e-20
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _geometric_mean(spds, max_iter=max_iter, tol=tol)
        assert len(w) == 1

    grad_norm = grad_geometric_mean(spds, max_iter=max_iter, tol=tol)

    assert len(grad_norm) == max_iter
    assert grad_norm[-1] > tol


# proportion of badly conditioned matrices
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_geometric_mean_properties_evaluate_convergence(p):
    n_matrices = 40
    n_features = 15
    # A warning is printed if tolerance is not reached
    spds = [
        random_spd(n_features, eig_min=1e-2, cond=1e6, random_state=0)
        for _ in range(int(p * n_matrices))
    ]
    spds.extend(
        random_spd(n_features, eig_min=1.0, cond=10.0, random_state=0)
        for _ in range(int(p * n_matrices), n_matrices)
    )
    max_iter = 30 if p < 1 else 60

    _geometric_mean(spds, max_iter=max_iter, tol=1e-5)


def test_geometric_mean_error_non_square_matrix():
    n_features = 5
    mat1 = np.ones((n_features, n_features + 1))

    with pytest.raises(ValueError, match="Expected a square matrix"):
        _geometric_mean([mat1])


def test_geometric_mean_error_input_matrices_have_different_shapes():
    n_features = 5
    mat1 = np.eye(n_features)
    mat2 = np.ones((n_features + 1, n_features + 1))

    with pytest.raises(
        ValueError, match="Matrices are not of the same shape."
    ):
        _geometric_mean([mat1, mat2])


def test_geometric_mean_error_non_spd_input_matrix():
    n_features = 5
    mat2 = np.ones((n_features + 1, n_features + 1))

    with pytest.raises(
        ValueError, match="Expected a symmetric positive definite matrix."
    ):
        _geometric_mean([mat2])


def test_sym_matrix_to_vec():
    sym = np.ones((3, 3))
    sqrt2 = 1.0 / sqrt(2.0)
    vec = np.array([sqrt2, 1.0, sqrt2, 1.0, 1.0, sqrt2])

    assert_array_almost_equal(sym_matrix_to_vec(sym), vec)

    vec = np.array([1.0, 1.0, 1.0])

    assert_array_almost_equal(
        sym_matrix_to_vec(sym, discard_diagonal=True), vec
    )


def test_sym_matrix_to_vec_is_the_inverse_of_vec_to_sym_matrix(rng):
    n = 5
    p = n * (n + 1) // 2

    # when diagonal is included
    vec = rng.random(p)
    sym = vec_to_sym_matrix(vec)

    assert_array_almost_equal(sym_matrix_to_vec(sym), vec)

    # when diagonal given separately
    diagonal = rng.random(n + 1)
    sym = vec_to_sym_matrix(vec, diagonal=diagonal)

    assert_array_almost_equal(
        sym_matrix_to_vec(sym, discard_diagonal=True), vec
    )

    # multiple matrices case when diagonal is included
    vecs = np.asarray([vec, 2.0 * vec, 0.5 * vec])
    syms = vec_to_sym_matrix(vecs)

    assert_array_almost_equal(sym_matrix_to_vec(syms), vecs)

    # multiple matrices case when diagonal is given separately
    diagonals = np.asarray([diagonal, 3.0 * diagonal, -diagonal])
    syms = vec_to_sym_matrix(vecs, diagonal=diagonals)

    assert_array_almost_equal(
        sym_matrix_to_vec(syms, discard_diagonal=True), vecs
    )


def test_vec_to_sym_matrix():
    # Check output value is correct
    vec = np.ones(6)
    sym = np.array(
        [[sqrt(2), 1.0, 1.0], [1.0, sqrt(2), 1.0], [1.0, 1.0, sqrt(2)]]
    )

    assert_array_almost_equal(vec_to_sym_matrix(vec), sym)

    # Check output value is correct with separate diagonal
    vec = np.ones(3)
    diagonal = np.ones(3)

    assert_array_almost_equal(vec_to_sym_matrix(vec, diagonal=diagonal), sym)
    # Check vec_to_sym_matrix is the inverse function of sym_matrix_to_vec
    # when diagonal is included
    assert_array_almost_equal(vec_to_sym_matrix(sym_matrix_to_vec(sym)), sym)

    # when diagonal is discarded
    vec = sym_matrix_to_vec(sym, discard_diagonal=True)
    diagonal = np.diagonal(sym) / sqrt(2)

    assert_array_almost_equal(vec_to_sym_matrix(vec, diagonal=diagonal), sym)


def test_vec_to_sym_matrix_errors():
    # Check error if unsuitable size
    vec = np.ones(31)

    with pytest.raises(ValueError, match="Vector of unsuitable shape"):
        vec_to_sym_matrix(vec)

    # Check error if given diagonal shape incompatible with vec
    vec = np.ones(3)
    diagonal = np.zeros(4)

    with pytest.raises(ValueError, match="incompatible with vector"):
        vec_to_sym_matrix(vec, diagonal)


def test_prec_to_partial():
    precision = np.array(
        [
            [2.0, -1.0, 1.0],
            [-1.0, 2.0, -1.0],
            [1.0, -1.0, 1.0],
        ]
    )
    partial = np.array(
        [
            [1.0, 0.5, -sqrt(2.0) / 2.0],
            [0.5, 1.0, sqrt(2.0) / 2.0],
            [-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 1.0],
        ]
    )

    assert_array_almost_equal(prec_to_partial(precision), partial)


def test_connectivity_measure_errors():
    # Raising error for input subjects not iterable
    conn_measure = ConnectivityMeasure()

    with pytest.raises(
        ValueError, match="'subjects' input argument must be an iterable"
    ):
        conn_measure.fit(1.0)

    # input subjects not 2D numpy.ndarrays
    with pytest.raises(
        ValueError, match="Each subject must be 2D numpy.ndarray."
    ):
        conn_measure.fit([np.ones((100, 40)), np.ones((10,))])

    # input subjects with different number of features
    with pytest.raises(
        ValueError, match="All subjects must have the same number of features."
    ):
        conn_measure.fit([np.ones((100, 40)), np.ones((100, 41))])

    # fit_transform with a single subject and kind=tangent
    conn_measure = ConnectivityMeasure(kind="tangent")

    with pytest.raises(
        ValueError,
        match="Tangent space parametrization .* only be .* group of subjects",
    ):
        conn_measure.fit_transform([np.ones((100, 40))])


@pytest.mark.parametrize(
    "cov_estimator", [EmpiricalCovariance(), LedoitWolf()]
)
@pytest.mark.parametrize("kind", CONNECTIVITY_KINDS)
def test_connectivity_measure_generic(
    kind, cov_estimator, signals_and_covariances
):
    signals, covs = signals_and_covariances

    # Check outputs properties
    input_covs = copy.copy(covs)
    conn_measure = ConnectivityMeasure(kind=kind, cov_estimator=cov_estimator)
    connectivities = conn_measure.fit_transform(signals)

    # Generic
    assert isinstance(connectivities, np.ndarray)
    assert len(connectivities) == len(covs)

    for k, _ in enumerate(connectivities):
        assert_array_equal(input_covs[k], covs[k])

        assert is_spd(covs[k], decimal=7)


def _assert_connectivity_tangent(connectivities, conn_measure, covs):
    """Check output value properties for tangent connectivity measure \
    that they have the expected relationship \
    to the input covariance matrices.

    - the geometric mean of the eigenvalues
        of the mean covariance matrix is positive-definite
    - the whitening matrix (used to transform the data \
        also produces a positive-definite matrix
    """
    for true_covariance_matrix, estimated_covariance_matrix in zip(
        covs, connectivities
    ):
        assert_array_almost_equal(
            estimated_covariance_matrix, estimated_covariance_matrix.T
        )

        assert is_spd(conn_measure.whitening_, decimal=7)

        gmean_sqrt = _map_eigenvalues(np.sqrt, conn_measure.mean_)
        assert is_spd(gmean_sqrt, decimal=7)
        assert_array_almost_equal(
            conn_measure.whitening_.dot(gmean_sqrt),
            np.eye(N_FEATURES),
        )
        assert_array_almost_equal(
            gmean_sqrt.dot(
                _map_eigenvalues(np.exp, estimated_covariance_matrix)
            ).dot(gmean_sqrt),
            true_covariance_matrix,
        )


def _assert_connectivity_precision(connectivities, covs):
    """Estimated precision matrix: \
    - is positive definite, \
    - its product with the true covariance matrix \
      is close to the identity matrix.
    """
    for true_covariance_matrix, estimated_covariance_matrix in zip(
        covs, connectivities
    ):
        assert is_spd(estimated_covariance_matrix, decimal=7)
        assert_array_almost_equal(
            estimated_covariance_matrix.dot(true_covariance_matrix),
            np.eye(N_FEATURES),
        )


def _assert_connectivity_correlation(connectivities, cov_estimator, covs):
    """Verify that the estimated covariance matrix: \
       - is symmetric and positive definite \
       - has values close to 1 on its diagonal.

    If the covariance estimator is EmpiricalCovariance,
    the product of:
        - the square root of the diagonal of the true covariance matrix,
        - the estimated covariance matrix,
        - the square root of the diagonal of the true covariance matrix,

    should be close to the true covariance matrix.
    """
    for true_covariance_matrix, estimated_covariance_matrix in zip(
        covs, connectivities
    ):
        assert is_spd(estimated_covariance_matrix, decimal=7)

        assert_array_almost_equal(
            np.diag(estimated_covariance_matrix), np.ones(N_FEATURES)
        )

        if cov_estimator == EmpiricalCovariance():
            # square root of the diagonal of the true covariance matrix
            d = np.sqrt(np.diag(np.diag(true_covariance_matrix)))

            assert_array_almost_equal(
                d.dot(estimated_covariance_matrix).dot(d),
                true_covariance_matrix,
            )


def _assert_connectivity_partial_correlation(connectivities, covs):
    for true_covariance_matrix, estimated_covariance_matrix in zip(
        covs, connectivities
    ):
        precision_matrix = linalg.inv(true_covariance_matrix)

        # square root of the diagonal elements of the precision matrix
        d = np.sqrt(np.diag(np.diag(precision_matrix)))

        # normalize the computed partial correlation matrix
        # necessary to ensure that the diagonal elements
        # of the partial correlation matrix are equal to 1
        normalized_partial_correlation_matrix = d.dot(
            estimated_covariance_matrix
        ).dot(d)

        # expected value
        partial_corrlelation_matrix = -precision_matrix + 2 * np.diag(
            np.diag(precision_matrix)
        )

        assert_array_almost_equal(
            normalized_partial_correlation_matrix,
            partial_corrlelation_matrix,
        )


@pytest.mark.parametrize(
    "kind",
    ["tangent", "precision", "correlation", "partial correlation"],
)
@pytest.mark.parametrize(
    "cov_estimator", [EmpiricalCovariance(), LedoitWolf()]
)
def test_connectivity_measure_specific_for_each_kind(
    kind, cov_estimator, signals_and_covariances
):
    signals, covs = signals_and_covariances

    conn_measure = ConnectivityMeasure(kind=kind, cov_estimator=cov_estimator)
    connectivities = conn_measure.fit_transform(signals)

    if kind == "tangent":
        _assert_connectivity_tangent(connectivities, conn_measure, covs)
    elif kind == "precision":
        _assert_connectivity_precision(connectivities, covs)
    elif kind == "correlation":
        _assert_connectivity_correlation(connectivities, cov_estimator, covs)
    elif kind == "partial correlation":
        _assert_connectivity_partial_correlation(connectivities, covs)


@pytest.mark.parametrize("kind", CONNECTIVITY_KINDS)
def test_connectivity_measure_check_mean(kind, signals):
    conn_measure = ConnectivityMeasure(kind=kind)
    conn_measure.fit_transform(signals)

    assert (conn_measure.mean_).shape == (N_FEATURES, N_FEATURES)

    if kind != "tangent":
        assert_array_almost_equal(
            conn_measure.mean_,
            np.mean(conn_measure.transform(signals), axis=0),
        )

    # Check that the mean isn't modified in transform
    conn_measure = ConnectivityMeasure(kind="covariance")
    conn_measure.fit(signals[:1])
    mean = conn_measure.mean_
    conn_measure.transform(signals[1:])

    assert_array_equal(mean, conn_measure.mean_)


@pytest.mark.parametrize("kind", CONNECTIVITY_KINDS)
def test_connectivity_measure_check_vectorization_option(kind, signals):
    conn_measure = ConnectivityMeasure(kind=kind)
    connectivities = conn_measure.fit_transform(signals)
    conn_measure = ConnectivityMeasure(vectorize=True, kind=kind)
    vectorized_connectivities = conn_measure.fit_transform(signals)

    assert_array_almost_equal(
        vectorized_connectivities, sym_matrix_to_vec(connectivities)
    )

    # Check not fitted error
    with pytest.raises(ValueError, match="has not been fitted. "):
        ConnectivityMeasure().inverse_transform(vectorized_connectivities)


@pytest.mark.parametrize(
    "kind",
    ["covariance", "correlation", "precision", "partial correlation"],
)
def test_connectivity_measure_check_inverse_transformation(kind, signals):
    # without vectorization: input matrices are returned with no change
    conn_measure = ConnectivityMeasure(kind=kind)
    connectivities = conn_measure.fit_transform(signals)

    assert_array_almost_equal(
        conn_measure.inverse_transform(connectivities), connectivities
    )

    # with vectorization: input vectors are reshaped into matrices
    # if diagonal has not been discarded
    conn_measure = ConnectivityMeasure(kind=kind, vectorize=True)
    vectorized_connectivities = conn_measure.fit_transform(signals)

    assert_array_almost_equal(
        conn_measure.inverse_transform(vectorized_connectivities),
        connectivities,
    )


@pytest.mark.parametrize(
    "kind",
    ["covariance", "correlation", "precision", "partial correlation"],
)
def test_connectivity_measure_check_inverse_transformation_discard_diag(
    kind, signals
):
    # with vectorization
    connectivities = ConnectivityMeasure(kind=kind).fit_transform(signals)
    conn_measure = ConnectivityMeasure(
        kind=kind, vectorize=True, discard_diagonal=True
    )
    vectorized_connectivities = conn_measure.fit_transform(signals)

    if kind in ["correlation", "partial correlation"]:
        assert_array_almost_equal(
            conn_measure.inverse_transform(vectorized_connectivities),
            connectivities,
        )
    elif kind in ["covariance", "precision"]:
        diagonal = np.array(
            [np.diagonal(conn) / sqrt(2) for conn in connectivities]
        )
        inverse_transformed = conn_measure.inverse_transform(
            vectorized_connectivities, diagonal=diagonal
        )

        assert_array_almost_equal(inverse_transformed, connectivities)
        with pytest.raises(
            ValueError, match="cannot reconstruct connectivity matrices"
        ):
            conn_measure.inverse_transform(vectorized_connectivities)


def test_connectivity_measure_inverse_transform_tangent(
    signals,
):
    """For 'tangent' kind, covariance matrices are reconstructed."""
    # Without vectorization
    tangent_measure = ConnectivityMeasure(kind="tangent")
    displacements = tangent_measure.fit_transform(signals)
    covariances = ConnectivityMeasure(kind="covariance").fit_transform(signals)

    assert_array_almost_equal(
        tangent_measure.inverse_transform(displacements), covariances
    )

    # with vectorization
    # when diagonal has not been discarded
    tangent_measure = ConnectivityMeasure(kind="tangent", vectorize=True)
    vectorized_displacements = tangent_measure.fit_transform(signals)

    assert_array_almost_equal(
        tangent_measure.inverse_transform(vectorized_displacements),
        covariances,
    )

    # When diagonal has been discarded
    tangent_measure = ConnectivityMeasure(
        kind="tangent", vectorize=True, discard_diagonal=True
    )
    vectorized_displacements = tangent_measure.fit_transform(signals)

    diagonal = np.array(
        [np.diagonal(matrix) / sqrt(2) for matrix in displacements]
    )
    inverse_transformed = tangent_measure.inverse_transform(
        vectorized_displacements, diagonal=diagonal
    )

    assert_array_almost_equal(inverse_transformed, covariances)
    with pytest.raises(
        ValueError, match="cannot reconstruct connectivity matrices"
    ):
        tangent_measure.inverse_transform(vectorized_displacements)


def test_confounds_connectome_measure():
    n_subjects = 10

    signals, confounds = _signals(n_subjects)

    correlation_measure = ConnectivityMeasure(
        kind="correlation", vectorize=True
    )

    # Clean confounds on 10 subjects with confounds filtered to 10 subjects in
    # length
    cleaned_vectors = correlation_measure.fit_transform(
        signals, confounds=confounds[:10]
    )

    zero_matrix = np.zeros((confounds.shape[1], cleaned_vectors.shape[1]))
    assert_array_almost_equal(
        np.dot(confounds[:10].T, cleaned_vectors), zero_matrix
    )
    assert isinstance(cleaned_vectors, np.ndarray)

    # Confounds as pandas DataFrame
    confounds_df = DataFrame(confounds[:10])
    correlation_measure.fit_transform(signals, confounds=confounds_df)


def test_confounds_connectome_measure_errors(signals):
    # Generate signals and compute covariances and apply confounds while
    # computing covariances
    signals, confounds = _signals()

    # Raising error for input confounds are not iterable
    conn_measure = ConnectivityMeasure(vectorize=True)
    msg = "'confounds' input argument must be an iterable"

    with pytest.raises(ValueError, match=msg):
        conn_measure._check_input(X=signals, confounds=1.0)

    with pytest.raises(ValueError, match=msg):
        conn_measure._fit_transform(
            X=signals, do_fit=True, do_transform=True, confounds=1.0
        )

    with pytest.raises(ValueError, match=msg):
        conn_measure.fit_transform(X=signals, y=None, confounds=1.0)

    # Raising error for input confounds are given but not vectorize=True
    conn_measure = ConnectivityMeasure(vectorize=False)
    with pytest.raises(
        ValueError, match="'confounds' are provided but vectorize=False"
    ):
        conn_measure.fit_transform(signals, None, confounds[:10])


def test_connectivity_measure_standardize(signals):
    """Check warning is raised and then suppressed with setting standardize."""
    match = "default strategy for standardize"

    with pytest.deprecated_call(match=match):
        ConnectivityMeasure(kind="correlation").fit_transform(signals)

    with warnings.catch_warnings(record=True) as record:
        ConnectivityMeasure(
            kind="correlation", standardize="zscore_sample"
        ).fit_transform(signals)
        for m in record:
            assert match not in m.message
