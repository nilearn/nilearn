import pytest

from nilearn._estimator_checks.nilearn_checks import (
    CACHE_MIXIN_CHECKS,
    COMMON_CHECKS,
    DECOMPOSITION_CHECKS,
    GLM_CHECKS,
    IMG_INPUT_CLAS_REG_COMMON_CHECKS,
    IMG_INPUT_COMMON_CHECKS,
    IMG_INPUT_INVERSE_TRANSFORM_CHECKS,
    IMG_INPUT_REG_CHECKS,
    IMG_INPUT_REQUIRES_Y,
    IMG_INPUT_TRANSFORM_DTYPE_CHECKS,
    MULTI_MASKER_CHECKS,
    MULTINIFTIMASKER_CHECKS,
    NIFTIMASKER_CHECKS,
    NON_MULTI_MASKER_CHECKS,
    SURFACE_INPUT_MASKER_CHECKS,
    VOLUME_INPUT_MASKER_CHECKS,
    nilearn_check_generator,
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
from nilearn.regions import (
    HierarchicalKMeans,
    Parcellations,
    RegionExtractor,
    ReNA,
)

CONNECTOME = [
    ConnectivityMeasure,
    GroupSparseCovariance,
    GroupSparseCovarianceCV,
]

DECODING_CLASSIFIERS = [
    Decoder,
    FREMClassifier,
    SpaceNetClassifier,
]

DECODING_REGRESSORS = [
    DecoderRegressor,
    FREMRegressor,
    SpaceNetRegressor,
]

DECODING_SEARCH_LIGHT = [SearchLight]

DECOMPOSITION = [
    DictLearning,
    CanICA,
]

GLM = [
    FirstLevelModel,
    SecondLevelModel,
]

VOLUME_MASKERS = [
    NiftiMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiSpheresMasker,
]

VOLUME_MULTI_MASKERS = [
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
]

SURFACE_MASKERS = [
    SurfaceMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
]

SURFACE_MULTI_MASKERS = [
    MultiSurfaceLabelsMasker,
    MultiSurfaceMapsMasker,
    MultiSurfaceMasker,
]

REGIONS = [HierarchicalKMeans, RegionExtractor, ReNA, Parcellations]


@pytest.mark.parametrize(
    "estimator, expected_checks",
    # GLM estimators
    [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + GLM_CHECKS,
        )
        for e in GLM
    ]
    # Decomposition estimators
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + DECOMPOSITION_CHECKS,
        )
        for e in DECOMPOSITION
    ]
    # Connectome estimators except GroupSparseCovariance
    + [(e, COMMON_CHECKS) for e in CONNECTOME if e != GroupSparseCovariance]
    # GroupSparseCovariance
    + [(GroupSparseCovariance, COMMON_CHECKS + CACHE_MIXIN_CHECKS)]
    # Nifti maskers (non multi) except NiftiMasker
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + VOLUME_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + NON_MULTI_MASKER_CHECKS,
        )
        for e in VOLUME_MASKERS
        if e != NiftiMasker
    ]
    # NiftiMasker
    + [
        (
            NiftiMasker,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + VOLUME_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + NON_MULTI_MASKER_CHECKS
            + NIFTIMASKER_CHECKS,
        )
    ]
    # Multi Nifti maskers except MultiNiftiMasker
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + VOLUME_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + MULTI_MASKER_CHECKS,
        )
        for e in VOLUME_MULTI_MASKERS
        if e != MultiNiftiMasker
    ]
    # MultiNiftiMasker
    + [
        (
            MultiNiftiMasker,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + VOLUME_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + MULTI_MASKER_CHECKS
            + NIFTIMASKER_CHECKS
            + MULTINIFTIMASKER_CHECKS,
        )
    ]
    # Surface maskers
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + SURFACE_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + NON_MULTI_MASKER_CHECKS,
        )
        for e in SURFACE_MASKERS
    ]
    # Multi surface maskers
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + SURFACE_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + MULTI_MASKER_CHECKS,
        )
        for e in SURFACE_MULTI_MASKERS
    ]
    # Decoding estimators which are classifiers
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + IMG_INPUT_CLAS_REG_COMMON_CHECKS
            + IMG_INPUT_REQUIRES_Y,
        )
        for e in DECODING_CLASSIFIERS
    ]
    # Decoding estimators which are regressors
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + IMG_INPUT_REG_CHECKS
            + IMG_INPUT_REQUIRES_Y,
        )
        for e in DECODING_REGRESSORS
    ]
    # Region estimators: RegionExtractor
    + [
        (
            RegionExtractor,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + VOLUME_INPUT_MASKER_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + NON_MULTI_MASKER_CHECKS,
        )
    ]
    +
    # Region estimators: Parcellations
    [
        (
            Parcellations,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMG_INPUT_COMMON_CHECKS
            + IMG_INPUT_TRANSFORM_DTYPE_CHECKS
            + IMG_INPUT_INVERSE_TRANSFORM_CHECKS
            + DECOMPOSITION_CHECKS,
        )
    ]
    +
    # Region estimators: ReNA, HierarchicalKMeans
    [(e, COMMON_CHECKS) for e in [ReNA, HierarchicalKMeans]],
)
def test_nilearn_check_generator_common_checks(estimator, expected_checks):
    checks_found = 0
    checks_total = 0
    for check in nilearn_check_generator(estimator()):
        checks_total += 1
        checks_found += 1 if check in expected_checks else 0

    assert checks_found == len(expected_checks)
    assert checks_total == checks_found
