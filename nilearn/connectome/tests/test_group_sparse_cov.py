from nose.tools import assert_equal, assert_true, assert_raises

import numpy as np
from nilearn._utils.testing import generate_group_sparse_gaussian_graphs
from nilearn.connectome.group_sparse_cov import (group_sparse_covariance,
                                                        group_sparse_scores)
from nilearn.connectome import GroupSparseCovariance, GroupSparseCovarianceCV


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
                                              tol=1e-2, debug=True, verbose=0)
    emp_covs, omega2 = group_sparse_covariance(signals, alpha, max_iter=20,
                                               tol=1e-2, debug=True, verbose=0)

    np.testing.assert_almost_equal(omega, omega2, decimal=4)

    class Probe(object):

        def __init__(self):
            self.objective = []

        def __call__(self, emp_covs, n_samples, alpha, max_iter, tol, n, omega,
                     omega_diff):
            if n >= 0:
                _, objective = group_sparse_scores(omega, n_samples, emp_covs,
                                                   alpha)
                self.objective.append(objective)

    # Use a probe to test for number of iterations and decreasing objective.
    probe = Probe()
    emp_covs, omega = group_sparse_covariance(
        signals, alpha, max_iter=4, tol=None, verbose=0, probe_function=probe)
    objective = probe.objective
    # check number of iterations
    assert_equal(len(objective), 4)

    # np.testing.assert_array_less is a strict comparison.
    # Zeros can occur in np.diff(objective).
    assert_true(np.all(np.diff(objective) <= 0))
    assert_equal(omega.shape, (10, 10, 5))

    # Test input argument checking
    assert_raises(ValueError, group_sparse_covariance, signals, "")
    assert_raises(ValueError, group_sparse_covariance, 1, alpha)
    assert_raises(ValueError, group_sparse_covariance,
                  [np.ones((2, 2)), np.ones((2, 3))], alpha)

    # Check consistency between classes
    gsc1 = GroupSparseCovarianceCV(alphas=4, tol=1e-1, max_iter=20, verbose=0,
                                   early_stopping=True)
    gsc1.fit(signals)

    gsc2 = GroupSparseCovariance(alpha=gsc1.alpha_, tol=1e-1, max_iter=20,
                                 verbose=0)
    gsc2.fit(signals)

    np.testing.assert_almost_equal(gsc1.precisions_, gsc2.precisions_,
                                   decimal=4)
