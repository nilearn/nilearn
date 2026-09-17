import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.covariance import EmpiricalCovariance

from nilearn._utils.data_gen import generate_maps
from nilearn._utils.estimator_checks import nilearn_check_estimator
from nilearn.conftest import (
    _affine_eye,
    _img_3d_mni,
    _img_labels,
    _img_maps,
    _shape_3d_large,
    _surf_maps_img,
)
from nilearn.connectome import (
    ConnectivityMeasure,
    GroupSparseCovariance,
    GroupSparseCovarianceCV,
)
from nilearn.decoding import (
    Decoder,
    DecoderRegressor,
    FREMClassifier,
    FREMRegressor,
    SearchLight,
    SpaceNetClassifier,
    SpaceNetRegressor,
)
from nilearn.decomposition import CanICA, DictLearning
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
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
from nilearn.maskers.tests.conftest import sklearn_surf_label_img  # TODO move
from nilearn.regions import (
    HierarchicalKMeans,
    Parcellations,
    RegionExtractor,
    ReNA,
)

# this to here
from nilearn.utils.discovery import all_estimators

ESTIMATORS_TO_CHECK = [
    ConnectivityMeasure(cov_estimator=EmpiricalCovariance()),
    ConnectivityMeasure(),
    GroupSparseCovarianceCV(),
    GroupSparseCovariance(),
    Decoder(
        screening_percentile=100,
        estimator_args={"random_state": 0},
    ),
    DecoderRegressor(screening_percentile=100),
    FREMClassifier(
        screening_percentile=100,
        estimator_args={"random_state": 0},
    ),
    FREMRegressor(screening_percentile=100),
    SpaceNetClassifier(),
    SpaceNetRegressor(),
    SearchLight(
        mask_img=Nifti1Image(
            np.ones((5, 5, 5), dtype=bool).astype("uint8"), np.eye(4)
        )
    ),
    DictLearning(),
    CanICA(),
    FirstLevelModel(),
    SecondLevelModel(),
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
    RegionExtractor(
        maps_img=generate_maps(
            shape=_shape_3d_large(),
            n_regions=2,
            rand_gen=42,
            affine=_affine_eye(),
        )[0]
    ),
    HierarchicalKMeans(n_clusters=2),
    ReNA(mask_img=_img_3d_mni(), n_clusters=2),
    Parcellations(method="kmeans", n_parcels=5),
    Parcellations(method="ward", n_parcels=5),
    Parcellations(method="rena", n_parcels=5),
]


def test_check_estimator_count():
    """Test if all estimators provided by nilearn are covered by
    ESTIMATORS_TO_CHECK.
    """
    assert len({est.__class__ for est in ESTIMATORS_TO_CHECK}) == len(
        all_estimators()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)
