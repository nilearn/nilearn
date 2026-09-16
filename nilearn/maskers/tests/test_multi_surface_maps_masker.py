import pytest

from nilearn._utils.estimator_checks import nilearn_check_estimator
from nilearn.conftest import _surf_maps_img
from nilearn.maskers import MultiSurfaceMapsMasker


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            MultiSurfaceMapsMasker(_surf_maps_img()),
            MultiSurfaceMapsMasker(_surf_maps_img(n_regions=1)),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)
