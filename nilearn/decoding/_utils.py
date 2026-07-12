"""Utilities to check for decoders."""

import inspect
import warnings
from typing import Any, Literal, TypedDict

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
)
from sklearn.linear_model import (
    LassoCV,
    LogisticRegressionCV,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.svm import SVR, LinearSVC

from nilearn._utils import logger
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import _get_data
from nilearn._utils.versions import SKLEARN_GTE_1_8
from nilearn.exceptions import MaskWarning
from nilearn.image import get_data
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data

MAX_ITER = 10000

# Volume of a standard (MNI152) brain mask in mm^3
MNI152_BRAIN_VOLUME = 1882989.0

kwarg_logistic_regression_cv = {}
if SKLEARN_GTE_1_8:
    # TODO (sklearn 1.8) remove if
    # TODO (sklearn 1.10) remove 'use_legacy_attributes'
    kwarg_logistic_regression_cv = {
        "use_legacy_attributes": False,
        "scoring": "neg_log_loss",
    }


class EstimatorConfig(TypedDict):
    estimator: Any
    params: dict[str, Any]
    extra_params: dict[str, Any]


SUPPORTED_ESTIMATORS: dict[
    Literal["classifier", "regressor"], dict[str, EstimatorConfig]
] = {
    "classifier": {
        # "params" cannot be overridden
        # "extra_params" can be overridden by parameters passed by user
        "svc_l1": {
            "estimator": LinearSVC,
            "params": {
                "penalty": "l1",
            },
            "extra_params": {"max_iter": MAX_ITER, "random_state": 0},
        },
        "svc_l2": {
            "estimator": LinearSVC,
            "params": {"penalty": "l2"},
            "extra_params": {"max_iter": MAX_ITER, "random_state": 0},
        },
        "svc": {
            "estimator": LinearSVC,
            "params": {"penalty": "l2"},
            "extra_params": {"max_iter": MAX_ITER, "random_state": 0},
        },
        "logistic_l1": {
            "estimator": LogisticRegressionCV,
            "params": {
                "l1_ratios": (1,),
                "solver": "liblinear",
                **kwarg_logistic_regression_cv,
            },
            "extra_params": {},
        },
        "logistic_l2": {
            "estimator": LogisticRegressionCV,
            "params": {
                "l1_ratios": (0,),
                "solver": "liblinear",
                **kwarg_logistic_regression_cv,
            },
            "extra_params": {},
        },
        "logistic": {
            "estimator": LogisticRegressionCV,
            "params": {
                "l1_ratios": (0,),
                "solver": "liblinear",
                **kwarg_logistic_regression_cv,
            },
            "extra_params": {},
        },
        "ridge_classifier": {
            "estimator": RidgeClassifierCV,
            "params": {},
            "extra_params": {},
        },
        "dummy_classifier": {
            "estimator": DummyClassifier,
            "params": {"strategy": "stratified"},
            "extra_params": {"random_state": 0},
        },
    },
    "regressor": {
        "ridge_regressor": {
            "estimator": RidgeCV,
            "params": {},
            "extra_params": {},
        },
        "ridge": {"estimator": RidgeCV, "params": {}, "extra_params": {}},
        "lasso": {"estimator": LassoCV, "params": {}, "extra_params": {}},
        "lasso_regressor": {
            "estimator": LassoCV,
            "params": {},
            "extra_params": {},
        },
        "svr": {
            "estimator": SVR,
            "params": {"kernel": "linear"},
            "extra_params": {"max_iter": MAX_ITER},
        },
        "dummy_regressor": {
            "estimator": DummyRegressor,
            "params": {"strategy": "mean"},
            "extra_params": {},
        },
    },
}


def _get_mask_extent(mask_img):
    """Compute the extent of the provided brain mask.
    The extent is the volume of the mask in mm^3 if mask_img is a Nifti1Image
    or the number of vertices if mask_img is a SurfaceImage.

    Parameters
    ----------
    mask_img : Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    Returns
    -------
    mask_extent : float
        The computed volume in mm^3 (if mask_img is a Nifti1Image) or the
        number of vertices (if mask_img is a SurfaceImage).

    """
    if not hasattr(mask_img, "affine"):
        # sum number of True values in both hemispheres
        return (
            mask_img.data.parts["left"].sum()
            + mask_img.data.parts["right"].sum()
        )
    affine = mask_img.affine
    prod_vox_dims = 1.0 * np.abs(np.linalg.det(affine[:3, :3]))
    return prod_vox_dims * _get_data(mask_img).astype(bool).sum()


@fill_doc
def adjust_screening_percentile(screening_percentile, mask_img, verbose=0):
    """Adjust the screening percentile according to the MNI152 template or
    the number of vertices of the provided standard brain mesh.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img :  Nifti1Image or SurfaceImage
        The Nifti1Image whose voxel dimensions or the SurfaceImage whose
        number of vertices are to be computed.

    %(verbose0)s

    Returns
    -------
    screening_percentile : float in the interval [0, 100]
        Percentile value for ANOVA univariate feature selection.

    """
    original_screening_percentile = screening_percentile
    # correct screening_percentile according to the volume of the data mask
    # or the number of vertices of the reference mesh
    mask_extent = _get_mask_extent(mask_img)
    # if mask_img is a surface mesh, reference is the number of vertices
    # in the standard mesh otherwise it is the volume of the MNI152 brain
    # template
    reference_extent = (
        mask_img.mesh.n_vertices
        if isinstance(mask_img, SurfaceImage)
        else MNI152_BRAIN_VOLUME
    )
    if mask_extent > 1.1 * reference_extent:
        unit = "mm^3"
        if hasattr(mask_img, "mesh"):
            unit = "vertices"
        warnings.warn(
            f"Brain mask ({mask_extent} {unit}) is bigger than the standard "
            f"human brain ({reference_extent} {unit})."
            "This object is probably not tuned to be used on such data.",
            stacklevel=find_stack_level(),
            category=MaskWarning,
        )
    elif mask_extent < 0.005 * reference_extent:
        warnings.warn(
            "Brain mask is smaller than .5% of the size of the standard "
            "human brain. This object is probably not tuned to "
            "be used on such data.",
            stacklevel=find_stack_level(),
            category=MaskWarning,
        )

    if screening_percentile < 100.0:
        screening_percentile = screening_percentile * (
            reference_extent / mask_extent
        )
        screening_percentile = min(screening_percentile, 100.0)
    # if screening_percentile is 100, we don't do anything

    if hasattr(mask_img, "mesh"):
        log_mask = f"Mask n_vertices = {mask_extent:g}"
    else:
        log_mask = (
            f"Mask volume = {mask_extent:g}mm^3 = {mask_extent / 1000.0:g}cm^3"
        )
    logger.log(
        log_mask,
        verbose=verbose,
        msg_level=1,
    )
    if hasattr(mask_img, "mesh"):
        log_ref = f"Reference mesh n_vertices = {reference_extent:g}"
    else:
        log_ref = f"Standard brain volume = {MNI152_BRAIN_VOLUME:g}mm^3"
    logger.log(
        log_ref,
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Original screening-percentile: {original_screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    logger.log(
        f"Corrected screening-percentile: {screening_percentile:g}",
        verbose=verbose,
        msg_level=1,
    )
    return screening_percentile


@fill_doc
def check_feature_screening(
    screening_percentile,
    mask_img,
    is_classification,
    screening_n_features=None,
    verbose=0,
):
    """Check feature screening method.

    Turns floats between 1 and 100 into SelectPercentile objects.

    Parameters
    ----------
    %(screening_percentile)s

    mask_img : nibabel image object
        Input image whose :term:`voxel` dimensions are to be computed.

    is_classification : bool
        If is_classification is True, it indicates that a classification task
        is performed. Otherwise, a regression task is performed.

    %(verbose0)s

    Returns
    -------
    selector : SelectPercentile instance
       Used to perform the :term:`ANOVA` univariate feature selection.

    """
    f_test = f_classif if is_classification else f_regression

    if screening_percentile is None and screening_n_features is not None:
        if mask_img is not None:
            if isinstance(mask_img, SurfaceImage):
                data = get_surface_data(mask_img)
            else:
                data = get_data(mask_img)
            n_features_in_mask = np.sum(data != 0)

            if screening_n_features > n_features_in_mask:
                raise ValueError(
                    f"{screening_n_features=} is larger "
                    "the number of features in the mask "
                    f"({n_features_in_mask})."
                )
        return SelectKBest(f_test, k=screening_n_features)

    if screening_percentile == 100 or screening_percentile is None:
        return None

    elif not (0.0 <= screening_percentile <= 100.0):
        raise ValueError(
            "screening_percentile should be in the interval"
            f" [0, 100], got {screening_percentile:g}"
        )

    else:
        # correct screening_percentile according to the volume or the number of
        # vertices in the data mask
        effective_screening_percentile = adjust_screening_percentile(
            screening_percentile,
            mask_img,
            verbose=verbose,
        )

        if effective_screening_percentile == 100:
            warnings.warn(
                f"screening_percentile set to '100' despite "
                f"requesting '{screening_percentile=}'. "
                "\nAll elements in the mask will be included. "
                "\nThis usually occurs when the mask image "
                "is too small compared to full brain mask.",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )

        return SelectPercentile(
            f_test, percentile=int(effective_screening_percentile)
        )


def validate_estimator(
    estimator,
    owning_class_type: Literal["classifier", "regressor", None] = None,
    estimator_args=None,
    verbose=0,
):
    """Check requested estimator.

    If an actual estimator instance was passed, we allow it but warn the user.

    Otherwise we instantiate one
    from the config defined in supported_estimators.

    Parameters
    ----------
    estimator : Any
        estimator to validate: can be a string
        or ideally a sklearn compatible object

    owning_class_type : "classifier" or "regressor" or None
        estimator type of the class
        in which the estimator to validate
        is embedded

    estimator_args: dict or None
        extra args to pass when instantiating the embedded estimator

    verbose:
        used to adjust the verbosity of the embedded estimator
    """
    if not isinstance(estimator, str):
        # The following tries to make sure that the estimator_type
        # matches that of the owning class
        # The user may not have defined estimator_type
        # so we have to be a bit lenient here.

        # TODO (sklearn >= 1.8) _estimator_type will be removed
        estimator_type = getattr(estimator, "_estimator_type", None)

        # TODO test with sklearn sklearn_version == 1.5.0
        if estimator_type is None and hasattr(estimator, "__sklearn_tags__"):
            estimator_type = getattr(
                estimator.__sklearn_tags__(), "estimator_type", None
            )

        if (
            owning_class_type is not None
            and estimator_type is not None
            and owning_class_type != estimator_type
        ):
            raise ValueError(
                f"The estimator '{estimator.__class__.__name__}' "
                f"is of type '{estimator_type}' "
                f"and should be of type '{owning_class_type}'."
            )

        warnings.warn(
            "Use a custom estimator at your own risk of the process not "
            "working as intended. Nilearn cannot define a default tuning "
            "param_grid for custom estimators; when using a Decoder, provide "
            "param_grid to tune its hyperparameters.",
            stacklevel=find_stack_level(),
        )

        return estimator

    if owning_class_type is None:
        tmp = (
            SUPPORTED_ESTIMATORS["classifier"]
            | SUPPORTED_ESTIMATORS["regressor"]
        )
    else:
        tmp = SUPPORTED_ESTIMATORS[owning_class_type]
    estimator_config = tmp.get(estimator)

    if estimator_config is None:
        raise ValueError(
            "Invalid estimator. Known estimators are: "
            f"{list(tmp.keys())}. "
            f"Got: {estimator}"
        )

    # "extra_params" can be overridden by parameters passed by user
    params = estimator_config["extra_params"]
    if estimator_args is not None:
        params |= estimator_args

    # "params" cannot be overridden so we use them last
    # to update the parameter of the estimator
    params |= estimator_config["params"]

    sig = inspect.signature(estimator_config["estimator"]).parameters
    if "verbose" in sig:
        params["verbose"] = (verbose - 1) > 0

    estimator = estimator_config["estimator"](**params)

    return estimator
