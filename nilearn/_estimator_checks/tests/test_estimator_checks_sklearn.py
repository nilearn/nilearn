import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._estimator_checks.sklearn_expected_failed_checks import (
    return_expected_failed_checks,
)
from nilearn.utils.discovery import all_estimators


@pytest.mark.slow
@parametrize_with_checks(
    estimators=[est() for _, est in all_estimators()],
    expected_failed_checks=return_expected_failed_checks,
)
def test_check_estimator_sklearn(estimator, check):
    """Check compliance with sklearn estimators."""
    check(estimator)
