import numpy as np
import scipy.linalg

from nose.tools import assert_equal, assert_true, assert_raises

from ..group_sparse_covariance import (group_sparse_covariance,
                                       group_sparse_score,
                                       GroupSparseCovariance,
                                       GroupSparseCovarianceCV)


def generate_group_sparse_gaussian_graphs(
        n_subjects=5, n_features=30, min_n_samples=30, max_n_samples=50,
        density=0.1, random_state=np.random.RandomState(0)):
    """Generate signals drawn from a sparse Gaussian graphical model.

    Parameters
    ==========
    n_subjects : int
        number of subjects

    n_features : int
        number of signals per subject to generate

    density : float
        density of edges in graph topology

    min_n_samples, max_n_samples: int
        Each subject have a different number of samples, between these two
        numbers. All signals for a given subject have the same number of
        samples.

    Returns
    =======
    subjects : list of numpy.ndarray
        subjects[n] is the signals for subject n. They are provided as a numpy
        array with shape (sample number, n_features).
        len(subjects) == n_subjects

    precisions : list of numpy.ndarray
        precision matrices.

    topology : numpy.ndarray
        binary array giving the graph topology used for generating covariances
        and signals.
    """

    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.empty((n_features, n_features))
    topology[:, :] = np.triu((
        random_state.randint(0, high=int(1. / density),
                         size=n_features * n_features)
        ).reshape(n_features, n_features) == 0, k=1)

    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    for _ in range(n_subjects):

        # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
        prec = topology.copy()
        prec[mask] = random_state.uniform(low=0.1, high=.9, size=(mask.sum()))
        prec += -np.eye(prec.shape[0])
        prec = np.dot(prec.T, prec)

        np.testing.assert_almost_equal(prec, prec.T)
        eigenvalues = np.linalg.eigvalsh(prec)
        if eigenvalues.min() < 0:
            raise ValueError
        precisions.append(prec)

    # Returns the topology matrix of precision matrices.
    topology += np.eye(*topology.shape)
    topology = np.dot(topology.T, topology)
    topology = topology > 0
    assert(np.all(topology == topology.T))
    print("Sparsity: {0:f}".format(
        1. * topology.sum() / (topology.shape[0] ** 2)))

    # Generate temporal signals
    signals = []
    mean = np.zeros(topology.shape[0])
    n_samples = random_state.randint(min_n_samples, high=max_n_samples,
                                 size=len(precisions))

    for n, prec in zip(n_samples, precisions):
        signals.append(random_state.multivariate_normal(
            mean, -scipy.linalg.inv(prec), (n,)))

    return signals, precisions, topology


def test_group_sparse_covariance():
    # run in debug mode. Should not fail
    # without debug mode: cost must decrease.

    signals, _, _ = generate_group_sparse_gaussian_graphs(
        density=0.1, n_subjects=5, n_features=10,
        min_n_samples=100, max_n_samples=151,
        random_state=np.random.RandomState(0))

    alpha = 0.1

    # These executions must hit the tolerance limit
    emp_covs, omega = group_sparse_covariance(signals, alpha, max_iter=20,
                                              tol=1e-3, verbose=10, debug=True)
    emp_covs, omega2 = group_sparse_covariance(signals, alpha, max_iter=20,
                                               tol=1e-3, verbose=0,
                                               debug=True)
    np.testing.assert_almost_equal(omega, omega2, decimal=4)

    class Probe(object):
        def __init__(self):
            self.objective = []

        def __call__(self, emp_covs, n_samples, alpha, max_iter, tol, n, omega,
                     omega_diff):
            if n >= 0:
                _, objective = group_sparse_score(omega, n_samples, emp_covs,
                                                  alpha)
                self.objective.append(objective)
    # Use a probe to test for number of iterations and decreasing objective.
    probe = Probe()
    emp_covs, omega = group_sparse_covariance(
        signals, alpha, max_iter=7, tol=None, verbose=0, probe_function=probe)
    objective = probe.objective
    ## # check number of iterations
    assert_equal(len(objective), 7)

    ## np.testing.assert_array_less is a strict comparison.
    ## Zeros can occur in np.diff(objective).
    assert_true(np.all(np.diff(objective) <= 0))
    assert_equal(omega.shape, (10, 10, 5))

    # Test input argument checking
    assert_raises(ValueError, group_sparse_covariance, signals, "")
    assert_raises(ValueError, group_sparse_covariance, 1, alpha)
    assert_raises(ValueError, group_sparse_covariance,
                  [np.ones((2, 2)), np.ones((2, 3))], alpha)

    # Check consistency between classes
    gsc1 = GroupSparseCovarianceCV(alphas=4, tol=1e-1, max_iter=40, verbose=0,
                                   assume_centered=False, n_jobs=3)
    gsc1.fit(signals)

    gsc2 = GroupSparseCovariance(alpha=gsc1.alpha_, tol=1e-1, max_iter=40,
                                 verbose=0, assume_centered=False)
    gsc2.fit(signals)

    np.testing.assert_almost_equal(gsc1.precisions_, gsc2.precisions_,
                                   decimal=4)
