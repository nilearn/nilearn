import pytest

from nilearn._utils.estimator_checks import nilearn_check_estimator
from nilearn.maskers import MultiSurfaceLabelsMasker
from nilearn.maskers.tests.conftest import sklearn_surf_label_img


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            MultiSurfaceLabelsMasker(sklearn_surf_label_img()),
            MultiSurfaceLabelsMasker(sklearn_surf_label_img(n_regions=1)),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)
