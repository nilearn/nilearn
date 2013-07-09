import numpy as np

from nose.tools import assert_equal, assert_true, assert_raises

from .. group_sparse_covariance import (group_sparse_covariance,
                                        GroupSparseCovariance,
                                        GroupSparseCovarianceCV)


def generate_multi_task_gg_model(n_tasks=5, n_var=30, density=0.1,
                                 min_samples=30, max_samples=50,
                                 rand_gen=np.random.RandomState(0)):
    """Generate signals drawn from a sparse gaussian graphical models.

    Parameters
    ==========
    n_tasks: int
        number of tasks

    n_var: int
        number of signals per task to generate

    density: float
        density of edges in graph topology

    min_samples, max_samples: int
        Each task have a different number of samples, between these two
        numbers. All signals in a given task have the same number of samples.

    Returns
    =======
    tasks: list of numpy.ndarray
        tasks[n] is the signals for task n. They are provided as a numpy array
        with shape (sample number, n_var). len(tasks) == n_tasks

    precisions: list of numpy.ndarray
        precision matrices.

    topology: numpy.ndarray
        binary array giving the graph topology used for generating covariances
        and signals.
    """

    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.ndarray((n_var, n_var))
    topology[:, :] = np.triu((
        rand_gen.randint(0, high=int(1. / density), size=n_var * n_var)
        ).reshape(n_var, n_var) == 0, k=1)

    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    for _ in xrange(n_tasks):

        # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
        prec = topology.copy()
        prec[mask] = rand_gen.uniform(low=0.1, high=.9, size=(mask.sum()))
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
    n_samples = rand_gen.randint(min_samples, high=max_samples,
                                 size=len(precisions))

    for n, prec in zip(n_samples, precisions):
        signals.append(rand_gen.multivariate_normal(mean, -np.linalg.inv(prec),
                                                    (n,)))

    return signals, precisions, topology


def test_group_sparse_covariance():
    # run in debug mode. Should not fail
    # without debug mode: cost must decrease.

    signals, _, _ = generate_multi_task_gg_model(
        density=0.1, n_tasks=5, n_var=10, min_samples=100, max_samples=151,
        rand_gen=np.random.RandomState(0))

    rho = 0.1

    # These executions must hit the tolerance limit
    emp_covs, omega = group_sparse_covariance(signals, rho, max_iter=20,
                                              tol=1e-5, verbose=10, debug=True,
                                              return_costs=False)
    emp_covs, omega2 = group_sparse_covariance(signals, rho, max_iter=20,
                                               tol=1e-5, verbose=0,
                                               debug=True,
                                               return_costs=False)
    np.testing.assert_almost_equal(omega, omega2, decimal=4)

    emp_covs, omega, costs = group_sparse_covariance(
        signals, rho, max_iter=10, tol=None, verbose=0, return_costs=True)

    # check number of iterations
    assert_equal(len(costs), 10)

    ## np.testing.assert_array_less is a strict comparison.
    ## Zeros can occur in 'objective'.
    objective, _ = zip(*costs)
    assert_true(np.all(np.diff(objective) <= 0))
    assert_equal(omega.shape, (10, 10, 5))

    # Test input argument checking
    assert_raises(ValueError, group_sparse_covariance, signals, "")
    assert_raises(ValueError, group_sparse_covariance, 1, rho)
    assert_raises(ValueError, group_sparse_covariance,
                  [np.ones((2, 2)), np.ones((2, 3))], rho)

    # Check consistency between classes
    gsc1 = GroupSparseCovarianceCV(rhos=4, tol=1e-5, max_iter=40, verbose=0,
                                   assume_centered=False, n_jobs=3)
    gsc1.fit(signals)

    gsc2 = GroupSparseCovariance(rho=gsc1.rho_, tol=1e-5, max_iter=20,
                                 verbose=0, assume_centered=False)
    gsc2.fit(signals)

    np.testing.assert_almost_equal(gsc1.precisions_, gsc2.precisions_,
                                   decimal=4)
