import pytest

from nilearn._utils.estimator_checks import nilearn_check_estimator
from nilearn.maskers import MultiSurfaceMasker


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=[MultiSurfaceMasker()]),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)
