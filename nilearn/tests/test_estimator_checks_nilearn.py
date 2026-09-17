import pytest

from nilearn._utils.estimator_checks import nilearn_check_estimator
from nilearn.conftest import _img_labels, _img_maps, _surf_maps_img
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    MultiSurfaceLabelsMasker,
    MultiSurfaceMapsMasker,
    MultiSurfaceMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)
from nilearn.maskers.tests.conftest import sklearn_surf_label_img
from nilearn.utils.discovery import all_estimators

ESTIMATORS_TO_CHECK = [
    NiftiMasker(),
    NiftiLabelsMasker(labels_img=_img_labels()),
    NiftiLabelsMasker(labels_img=_img_labels(n_regions=1)),
    NiftiMapsMasker(maps_img=_img_maps(n_regions=2)),
    NiftiMapsMasker(maps_img=_img_maps(n_regions=1)),
    NiftiSpheresMasker(seeds=[(1, 1, 1)]),
    NiftiSpheresMasker(seeds=[(1, 1, 1), (1, 2, 3)]),
    SurfaceMasker(),
    SurfaceMapsMasker(_surf_maps_img()),
    SurfaceMapsMasker(_surf_maps_img(n_regions=1)),
    SurfaceLabelsMasker(sklearn_surf_label_img()),
    SurfaceLabelsMasker(sklearn_surf_label_img(n_regions=1)),
    MultiNiftiMasker(),
    MultiNiftiLabelsMasker(labels_img=_img_labels()),
    MultiNiftiLabelsMasker(labels_img=_img_labels(n_regions=1)),
    MultiNiftiMapsMasker(_img_maps(n_regions=2)),
    MultiNiftiMapsMasker(_img_maps(n_regions=1)),
    MultiSurfaceMasker(),
    MultiSurfaceLabelsMasker(sklearn_surf_label_img()),
    MultiSurfaceLabelsMasker(sklearn_surf_label_img(n_regions=1)),
    MultiSurfaceMapsMasker(_surf_maps_img()),
    MultiSurfaceMapsMasker(_surf_maps_img(n_regions=1)),
]


def test_check_estimator_count():
    """Test if all estimators provided by nilearn are covered by
    ESTIMATORS_TO_CHECK.
    """
    assert len({est.__class__ for est in ESTIMATORS_TO_CHECK}) == len(
        all_estimators()
    )


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)
