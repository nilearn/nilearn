import numpy as np
import pytest

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.data_gen import generate_group_sparse_gaussian_graphs
from nilearn.connectome import GroupSparseCovariance, GroupSparseCovarianceCV
from nilearn.connectome.group_sparse_cov import (
    group_sparse_covariance,
    group_sparse_scores,
)

extra_valid_checks = [
    "check_parameters_default_constructible",
    "check_no_attributes_set_in_init",
    "check_estimators_unfitted",
    "check_do_not_raise_errors_in_init_or_set_params",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    (
        check_estimator(
            estimator=[GroupSparseCovarianceCV(), GroupSparseCovariance()],
            extra_valid_checks=extra_valid_checks,
        )
    ),
)
def test_check_estimator_group_sparse_covariance(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[GroupSparseCovarianceCV(), GroupSparseCovariance()],
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


def test_group_sparse_covariance(rng):
    # run in debug mode. Should not fail
    # without debug mode: cost must decrease.

    signals, _, _ = generate_group_sparse_gaussian_graphs(
        density=0.1,
        n_subjects=5,
        n_features=10,
        min_n_samples=100,
        max_n_samples=151,
        random_state=rng,
    )

    alpha = 0.1

    # These executions must hit the tolerance limit
    _, omega = group_sparse_covariance(
        signals, alpha, max_iter=20, tol=1e-2, debug=True, verbose=1
    )
    _, omega2 = group_sparse_covariance(
        signals, alpha, max_iter=20, tol=1e-2, debug=True, verbose=0
    )

    np.testing.assert_almost_equal(omega, omega2, decimal=4)


@pytest.mark.parametrize("duality_gap", [True, False])
def test_group_sparse_covariance_with_probe_function(rng, duality_gap):
    signals, _, _ = generate_group_sparse_gaussian_graphs(
        density=0.1,
        n_subjects=5,
        n_features=10,
        min_n_samples=100,
        max_n_samples=151,
        random_state=rng,
    )

    alpha = 0.1

    class Probe:
        def __init__(self):
            self.objective = []

        def __call__(
            self,
            emp_covs,
            n_samples,
            alpha,
            max_iter,  # noqa: ARG002
            tol,  # noqa: ARG002
            n,
            omega,
            omega_diff,  # noqa: ARG002
        ):
            if n >= 0:
                if duality_gap:
                    _, objective, _ = group_sparse_scores(
                        omega,
                        n_samples,
                        emp_covs,
                        alpha,
                        duality_gap=duality_gap,
                    )
                else:
                    _, objective = group_sparse_scores(
                        omega,
                        n_samples,
                        emp_covs,
                        alpha,
                        duality_gap=duality_gap,
                    )
                self.objective.append(objective)

    # Use a probe to test for number of iterations and decreasing objective.
    probe = Probe()
    _, omega = group_sparse_covariance(
        signals, alpha, max_iter=4, tol=None, verbose=0, probe_function=probe
    )
    objective = probe.objective
    # check number of iterations
    assert len(objective) == 4

    # np.testing.assert_array_less is a strict comparison.
    # Zeros can occur in np.diff(objective).
    assert np.all(np.diff(objective) <= 0)
    assert omega.shape == (10, 10, 5)


def test_group_sparse_covariance_check_consistency_between_classes(rng):
    signals, _, _ = generate_group_sparse_gaussian_graphs(
        density=0.1,
        n_subjects=5,
        n_features=10,
        min_n_samples=100,
        max_n_samples=151,
        random_state=rng,
    )

    # Check consistency between classes
    gsc1 = GroupSparseCovarianceCV(
        alphas=4, tol=1e-1, max_iter=20, verbose=0, early_stopping=True
    )
    gsc1.fit(signals)

    gsc2 = GroupSparseCovariance(
        alpha=gsc1.alpha_, tol=1e-1, max_iter=20, verbose=0
    )
    gsc2.fit(signals)

    np.testing.assert_almost_equal(
        gsc1.precisions_, gsc2.precisions_, decimal=4
    )


def test_group_sparse_covariance_errors(rng):
    signals, _, _ = generate_group_sparse_gaussian_graphs(
        density=0.1,
        n_subjects=5,
        n_features=10,
        min_n_samples=100,
        max_n_samples=151,
        random_state=rng,
    )

    alpha = 0.1

    # Test input argument checking
    with pytest.raises(ValueError, match="must be a positive number"):
        group_sparse_covariance(signals, "")
    with pytest.raises(ValueError, match="subjects' .* must be .* iterable"):
        group_sparse_covariance(1, alpha)
    with pytest.raises(
        ValueError, match="All subjects must have the same number of features."
    ):
        group_sparse_covariance([np.ones((2, 2)), np.ones((2, 3))], alpha)
