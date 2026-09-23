from pathlib import Path

import numpy as np
import pandas as pd
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
from nilearn.utils.discovery import all_estimators

CONNECTOME = [
    ConnectivityMeasure(cov_estimator=EmpiricalCovariance()),
    ConnectivityMeasure(),
    GroupSparseCovarianceCV(),
    GroupSparseCovariance(),
]


DECODING = [
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
]

DECOMPOSITION = [
    DictLearning(random_state=0),
    CanICA(random_state=0),
]

GLM = [
    FirstLevelModel(),
    SecondLevelModel(),
]

MASKERS = [
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


REGIONS = [
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


ESTIMATORS_TO_CHECK = (
    CONNECTOME + DECODING + DECOMPOSITION + GLM + MASKERS + REGIONS
)


def _estimators():
    """Create list of estimators to be used for nilearn checks.

    Nilearn estimator checks should be run only for the estimators whose
    package is modified. The list of modified packages is taken from from
    ``tests_to_run.txt`` file. This file is generated only when tests are run
    in CI. To generate it locally,
    `` python build_tools/github/restrict_tests_to_run.py`` command must be run
    in command line before running tests.
    """
    path = Path(__file__).resolve().parent.parent.parent / "tests_to_run.txt"
    if path.exists():
        data = pd.read_csv(path, sep=" ")
        packages = data.columns
        estimator_list = []

        if packages is not None:
            for package in packages:
                if "connectome" in package:
                    estimator_list.extend(CONNECTOME)
                elif "decoding" in package:
                    estimator_list.extend(DECODING)
                elif "decomposition" in package:
                    estimator_list.extend(DECOMPOSITION)
                elif "glm" in package:
                    estimator_list.extend(GLM)
                elif "maskers" in package:
                    estimator_list.extend(MASKERS)
                elif "regions" in package:
                    estimator_list.extend(REGIONS)
    else:
        estimator_list = ESTIMATORS_TO_CHECK

    return estimator_list


def test_check_estimator_count():
    """Test if all estimators provided by nilearn are covered by
    ESTIMATORS_TO_CHECK.
    """
    assert len({est.__class__ for est in ESTIMATORS_TO_CHECK}) == len(
        all_estimators()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, name, check",
    nilearn_check_estimator(estimators=_estimators()),
)
def test_check_estimator_nilearn(estimator, name, check):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)
