import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn.maskers import MultiSurfaceLabelsMasker
from nilearn.maskers.tests.conftest import sklearn_surf_label_img

ESTIMATORS_TO_CHECK = [
    MultiSurfaceLabelsMasker(sklearn_surf_label_img()),
    MultiSurfaceLabelsMasker(sklearn_surf_label_img(n_regions=1)),
]


@parametrize_with_checks(
    estimators=ESTIMATORS_TO_CHECK,
    expected_failed_checks=return_expected_failed_checks,
)
def test_check_estimator_sklearn(estimator, check):
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)
