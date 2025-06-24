"""Checks for nilearn estimators.

Most of those estimators have pytest dependencies
and importing them will fail if pytest is not installed.
"""

import inspect
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _is_public_parameter,
    check_is_fitted,
)
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)

from nilearn._utils.exceptions import DimensionError, MeshDimensionError
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.niimg_conversions import check_imgs_equal
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import (
    _affine_eye,
    _affine_mni,
    _drop_surf_img_part,
    _flip_surf_img,
    _img_3d_mni,
    _img_3d_ones,
    _img_3d_rand,
    _img_3d_zeros,
    _img_4d_rand_eye,
    _img_4d_rand_eye_medium,
    _img_mask_mni,
    _make_mesh,
    _make_surface_img,
    _make_surface_img_and_design,
    _make_surface_mask,
    _rng,
    _shape_3d_default,
    _shape_3d_large,
    _surf_mask_1d,
)
from nilearn.connectome import GroupSparseCovariance, GroupSparseCovarianceCV
from nilearn.connectome.connectivity_matrices import ConnectivityMeasure
from nilearn.decoding.decoder import _BaseDecoder
from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.decomposition._base import _BaseDecomposition
from nilearn.maskers import (
    MultiNiftiMapsMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)
from nilearn.masking import load_mask_img
from nilearn.regions import RegionExtractor
from nilearn.regions.hierarchical_kmeans_clustering import HierarchicalKMeans
from nilearn.regions.rena_clustering import ReNA
from nilearn.reporting.tests.test_html_report import _check_html
from nilearn.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data
from nilearn.surface.utils import (
    assert_surface_image_equal,
)

SKLEARN_MINOR = parse(sklearn_version).release[1]


def nilearn_dir() -> Path:
    return Path(__file__).parents[1]


def check_estimator(estimators: list[BaseEstimator], valid: bool = True):
    """Yield a valid or invalid scikit-learn estimators check.

    ONLY USED FOR sklearn<1.6

    As some of Nilearn estimators do not comply
    with sklearn recommendations
    (cannot fit Numpy arrays, do input validation in the constructor...)
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made generator that yields an estimator instance
    along with a
    - valid check from sklearn: those should stay valid
    - or an invalid check that is known to fail.

    See this section rolling-your-own-estimator in
    the scikit-learn doc for more info:
    https://scikit-learn.org/stable/developers/develop.html

    Parameters
    ----------
    estimators : list of estimator object
        Estimator instance to check.

    valid : bool, default=True
        Whether to return only the valid checks or not.
    """
    # TODO remove this function when dropping sklearn 1.5
    if not SKLEARN_LT_1_6:  # pragma: no cover
        raise RuntimeError(
            "Use dedicated sklearn utilities to test estimators."
        )

    if not isinstance(estimators, list):  # pragma: no cover
        raise TypeError(
            "'estimators' should be a list. "
            f"Got {estimators.__class__.__name__}."
        )

    for est in estimators:
        expected_failed_checks = return_expected_failed_checks(est)

        for e, check in sklearn_check_estimator(
            estimator=est, generate_only=True
        ):
            if not valid and check.func.__name__ in expected_failed_checks:
                yield e, check, check.func.__name__
            if valid and check.func.__name__ not in expected_failed_checks:
                yield e, check, check.func.__name__


# some checks would fail on sklearn 1.6.1 on older python
# see https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1131
IS_SKLEARN_1_6_1_on_py_3_9 = (
    SKLEARN_MINOR == 6
    and parse(sklearn_version).release[2] == 1
    and sys.version_info[1] < 10
)


def return_expected_failed_checks(
    estimator: BaseEstimator,
) -> dict[str, str]:
    """Return the expected failures for a given estimator.

    This is where all the "expected_failed_checks" for all Nilearn estimators
    are centralized.

    "expected_failed_checks" is first created to make sure that all checks
    with the oldest supported sklearn versions pass.

    After the function may tweak the "expected_failed_checks" depending
    on the estimator and sklearn version.

    Returns
    -------
    expected_failed_checks : dict[str, str]
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.
    """
    expected_failed_checks: dict[str, str] = {}

    if isinstance(estimator, ConnectivityMeasure):
        expected_failed_checks = {
            "check_estimator_sparse_data": "remove when dropping sklearn 1.4",
            "check_fit2d_predict1d": "not applicable",
            "check_estimator_sparse_array": "TODO",
            "check_estimator_sparse_matrix": "TODO",
            "check_methods_sample_order_invariance": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_n_features_in": "TODO",
            "check_n_features_in_after_fitting": "TODO",
            "check_readonly_memmap_input": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
        }
        if SKLEARN_MINOR > 4:
            expected_failed_checks.pop("check_estimator_sparse_data")
            expected_failed_checks |= {
                "check_transformer_preserve_dtypes": "TODO",
            }

        return expected_failed_checks

    elif isinstance(estimator, HierarchicalKMeans):
        return expected_failed_checks_clustering()

    elif isinstance(estimator, ReNA):
        expected_failed_checks = {
            "check_estimator_sparse_array": "remove when dropping sklearn 1.4",
            "check_estimator_sparse_matrix": (
                "remove when dropping sklearn 1.4"
            ),
            "check_clustering": "TODO",
            "check_dict_unchanged": "TODO",
            "check_dtype_object": "TODO",
            "check_dont_overwrite_parameters": "TODO",
            "check_estimators_dtypes": "TODO",
            "check_estimators_fit_returns_self": "TODO",
            "check_estimators_nan_inf": "TODO",
            "check_estimators_overwrite_params": "TODO",
            "check_estimators_pickle": "TODO",
            "check_f_contiguous_array_estimator": "TODO",
            "check_fit_idempotent": "TODO",
            "check_fit_check_is_fitted": "TODO",
            "check_fit_score_takes_y": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_methods_sample_order_invariance": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_n_features_in": "TODO",
            "check_n_features_in_after_fitting": "TODO",
            "check_pipeline_consistency": "TODO",
            "check_positive_only_tag_during_fit": "TODO",
            "check_readonly_memmap_input": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

        if SKLEARN_MINOR >= 5:
            expected_failed_checks.pop("check_estimator_sparse_matrix")
            expected_failed_checks.pop("check_estimator_sparse_array")

        return expected_failed_checks

    elif isinstance(
        estimator, (GroupSparseCovariance, GroupSparseCovarianceCV)
    ):
        return {
            "check_fit_score_takes_y": "not applicable",
            "check_fit_check_is_fitted": "handled by nilearn checks",
            "check_dict_unchanged": "TODO",
            "check_dont_overwrite_parameters": "TODO",
            "check_dtype_object": "TODO",
            "check_estimator_sparse_array": "TODO",
            "check_estimator_sparse_data": "TODO",
            "check_estimator_sparse_matrix": "TODO",
            "check_estimators_dtypes": "TODO",
            "check_estimators_empty_data_messages": "TODO",
            "check_estimators_fit_returns_self": "TODO",
            "check_estimators_nan_inf": "TODO",
            "check_estimators_overwrite_params": "TODO",
            "check_estimators_pickle": "TODO",
            "check_f_contiguous_array_estimator": "TODO",
            "check_fit_idempotent": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_1sample": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_methods_sample_order_invariance": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_n_features_in": "TODO",
            "check_n_features_in_after_fitting": "TODO",
            "check_pipeline_consistency": "TODO",
            "check_positive_only_tag_during_fit": "TODO",
            "check_readonly_memmap_input": "TODO",
        }

    # below this point we should only deal with estimators
    # that accept images as input
    assert accept_niimg_input(estimator) or accept_surf_img_input(estimator)

    if isinstance(estimator, (_BaseDecoder, SearchLight)):
        return expected_failed_checks_decoders(estimator)

    # keeping track of some of those in
    # https://github.com/nilearn/nilearn/issues/4538
    expected_failed_checks = {
        # the following are skipped
        # because there is nilearn specific replacement
        "check_dict_unchanged": "replaced by check_masker_dict_unchanged",
        "check_dont_overwrite_parameters": (
            "replaced by check_img_estimator_dont_overwrite_parameters"
        ),
        "check_estimators_dtypes": ("replaced by check_masker_dtypes"),
        "check_estimators_empty_data_messages": (
            "replaced by check_masker_empty_data_messages "
            "for surface maskers and not implemented for nifti maskers "
            "for performance reasons."
        ),
        "check_estimators_fit_returns_self": (
            "replaced by check_fit_returns_self"
        ),
        "check_fit_check_is_fitted": ("replaced by check_masker_fitted"),
        "check_fit_score_takes_y": (
            "replaced by check_masker_fit_score_takes_y"
        ),
        # Those are skipped for now they fail
        # for unknown reasons
        # most often because sklearn inputs expect a numpy array
        # that errors with maskers,
        # or because a suitable nilearn replacement
        # has not yet been created.
        "check_estimators_pickle": "TODO",
        "check_estimators_nan_inf": "TODO",
        "check_estimators_overwrite_params": "TODO",
        "check_fit_idempotent": "TODO",
        "check_methods_sample_order_invariance": "TODO",
        "check_methods_subset_invariance": "TODO",
        "check_positive_only_tag_during_fit": "TODO",
        "check_pipeline_consistency": "TODO",
        "check_readonly_memmap_input": "TODO",
    }

    expected_failed_checks |= unapplicable_checks()

    if hasattr(estimator, "transform"):
        expected_failed_checks |= {
            "check_transformer_data_not_an_array": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_general": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_preserve_dtypes": (
                "replaced by check_masker_transformer"
            ),
        }

    # Adapt some checks for some estimators

    # not entirely sure why some of them pass
    # e.g check_estimator_sparse_data passes for SurfaceLabelsMasker
    # but not SurfaceMasker ????

    if is_glm(estimator):
        expected_failed_checks.pop("check_estimator_sparse_data")
        if SKLEARN_MINOR >= 5:
            expected_failed_checks.pop("check_estimator_sparse_matrix")
            expected_failed_checks.pop("check_estimator_sparse_array")
        if SKLEARN_MINOR >= 6:
            expected_failed_checks.pop("check_estimator_sparse_tag")

        expected_failed_checks |= {
            # have nilearn replacements
            "check_estimators_dtypes": ("replaced by check_glm_dtypes"),
            "check_estimators_empty_data_messages": (
                "not implemented for nifti data for performance reasons"
            ),
            "check_estimators_fit_returns_self": (
                "replaced by check_glm_fit_returns_self"
            ),
            "check_fit_check_is_fitted": ("replaced by check_glm_is_fitted"),
            "check_transformer_data_not_an_array": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_general": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_preserve_dtypes": (
                "replaced by check_masker_transformer"
            ),
            # nilearn replacements required
            "check_dict_unchanged": "TODO",
            "check_fit_score_takes_y": "TODO",
        }

    if isinstance(estimator, (_BaseDecomposition,)):
        if SKLEARN_MINOR >= 6:
            expected_failed_checks.pop("check_estimator_sparse_tag")
        if not IS_SKLEARN_1_6_1_on_py_3_9 and SKLEARN_MINOR >= 5:
            expected_failed_checks.pop("check_estimator_sparse_array")

    if is_masker(estimator):
        if accept_niimg_input(estimator):
            # TODO remove when bumping to nilearn 0.13.0
            expected_failed_checks |= {
                "check_do_not_raise_errors_in_init_or_set_params": (
                    "Deprecation cycle started to fix."
                ),
                "check_no_attributes_set_in_init": (
                    "Deprecation cycle started to fix."
                ),
            }

        if isinstance(estimator, (NiftiMasker)) and SKLEARN_MINOR >= 5:
            if not IS_SKLEARN_1_6_1_on_py_3_9:
                expected_failed_checks.pop("check_estimator_sparse_array")

            expected_failed_checks.pop("check_estimator_sparse_tag")

        if isinstance(estimator, (RegionExtractor)) and SKLEARN_MINOR >= 6:
            expected_failed_checks.pop(
                "check_do_not_raise_errors_in_init_or_set_params"
            )

    return expected_failed_checks


def unapplicable_checks() -> dict[str, str]:
    """Return sklearn checks that do not apply for nilearn estimators \
       when they take images as input.
    """
    return dict.fromkeys(
        [
            "check_complex_data",
            "check_dtype_object",
            "check_estimator_sparse_array",
            "check_estimator_sparse_data",
            "check_estimator_sparse_matrix",
            "check_estimator_sparse_tag",
            "check_f_contiguous_array_estimator",
            "check_fit1d",
            "check_fit2d_1feature",
            "check_fit2d_1sample",
            "check_fit2d_predict1d",
            "check_n_features_in",
            "check_n_features_in_after_fitting",
        ],
        "not applicable for image input",
    )


def expected_failed_checks_clustering() -> dict[str, str]:
    expected_failed_checks = {
        "check_estimator_sparse_array": "remove when dropping sklearn 1.4",
        "check_estimator_sparse_matrix": "remove when dropping sklearn 1.4",
        "check_clustering": "TODO",
        "check_estimators_nan_inf": "TODO",
        "check_fit2d_predict1d": "TODO",
        "check_n_features_in": "TODO",
        "check_n_features_in_after_fitting": "TODO",
        "check_transformer_data_not_an_array": "TODO",
        "check_transformer_general": "TODO",
    }

    if SKLEARN_MINOR >= 5:
        expected_failed_checks.pop("check_estimator_sparse_matrix")
        expected_failed_checks.pop("check_estimator_sparse_array")

    return expected_failed_checks


def expected_failed_checks_decoders(estimator) -> dict[str, str]:
    """Return expected failed sklearn checks for nilearn decoders."""
    expected_failed_checks = {
        # the following are have nilearn replacement for masker and/or glm
        # but not for decoders
        "check_estimators_empty_data_messages": (
            "not implemented for nifti data performance reasons"
        ),
        "check_dont_overwrite_parameters": (
            "replaced by check_img_estimator_dont_overwrite_parameters"
        ),
        "check_estimators_fit_returns_self": (
            "replaced by check_fit_returns_self"
        ),
        "check_requires_y_none": (
            "replaced by check_image_estimator_requires_y_none"
        ),
        "check_supervised_y_no_nan": (
            "replaced by check_image_supervised_estimator_y_no_nan"
        ),
        # Those are skipped for now they fail
        # for unknown reasons
        # most often because sklearn inputs expect a numpy array
        # that errors with maskers,
        # or because a suitable nilearn replacement
        # has not yet been created.
        "check_dict_unchanged": "TODO",
        "check_estimators_dtypes": "TODO",
        "check_estimators_pickle": "TODO",
        "check_estimators_nan_inf": "TODO",
        "check_estimators_overwrite_params": "TODO",
        "check_fit_check_is_fitted": "TODO",
        "check_fit_idempotent": "TODO",
        "check_fit_score_takes_y": "TODO",
        "check_methods_sample_order_invariance": "TODO",
        "check_methods_subset_invariance": "TODO",
        "check_positive_only_tag_during_fit": "TODO",
        "check_pipeline_consistency": "TODO",
        "check_readonly_memmap_input": "TODO",
        "check_supervised_y_2d": "TODO",
    }

    if is_classifier(estimator):
        expected_failed_checks |= {
            "check_classifier_data_not_an_array": (
                "not applicable for image input"
            ),
            "check_classifiers_classes": "TODO",
            "check_classifiers_one_label": "TODO",
            "check_classifiers_regression_target": "TODO",
            "check_classifiers_train": "TODO",
        }

    if is_regressor(estimator):
        expected_failed_checks |= {
            "check_regressor_data_not_an_array": (
                "not applicable for image input"
            ),
            "check_regressor_multioutput": "TODO",
            "check_regressors_int": "TODO",
            "check_regressors_train": "TODO",
            "check_regressors_no_decision_function": "TODO",
        }

    if hasattr(estimator, "transform"):
        expected_failed_checks |= {
            "check_transformer_data_not_an_array": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_general": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_preserve_dtypes": (
                "replaced by check_masker_transformer"
            ),
        }

    expected_failed_checks |= unapplicable_checks()

    if isinstance(estimator, SearchLight):
        return expected_failed_checks

    if not IS_SKLEARN_1_6_1_on_py_3_9:
        expected_failed_checks.pop("check_estimator_sparse_tag")

    return expected_failed_checks


def nilearn_check_estimator(estimators: list[BaseEstimator]):
    if not isinstance(estimators, list):  # pragma: no cover
        raise TypeError(
            "'estimators' should be a list. "
            f"Got {estimators.__class__.__name__}."
        )
    for est in estimators:
        for e, check in nilearn_check_generator(estimator=est):
            yield e, check, check.__name__


def nilearn_check_generator(estimator: BaseEstimator):
    """Yield (estimator, check) tuples.

    Each nilearn check can be run on an initialized estimator.
    """
    if SKLEARN_LT_1_6:  # pragma: no cover
        tags = estimator._more_tags()
    else:
        tags = estimator.__sklearn_tags__()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        requires_y = isinstance(estimator, _BaseDecoder)
    else:
        requires_y = getattr(tags.target_tags, "required", False)

    yield (clone(estimator), check_estimator_has_sklearn_is_fitted)
    yield (clone(estimator), check_fit_returns_self)
    yield (clone(estimator), check_transformer_set_output)

    if accept_niimg_input(estimator) or accept_surf_img_input(estimator):
        if requires_y:
            yield (clone(estimator), check_image_estimator_requires_y_none)

        if is_classifier(estimator) or is_regressor(estimator):
            yield (clone(estimator), check_image_supervised_estimator_y_no_nan)
            yield (clone(estimator), check_decoder_empty_data_messages)

        if (
            is_classifier(estimator)
            or is_regressor(estimator)
            or is_masker(estimator)
            or is_glm(estimator)
        ):
            yield (
                clone(estimator),
                check_img_estimator_dont_overwrite_parameters,
            )
            yield (clone(estimator), check_img_estimators_overwrite_params)

    if is_masker(estimator):
        yield (clone(estimator), check_masker_clean_kwargs)
        yield (clone(estimator), check_masker_compatibility_mask_image)
        yield (clone(estimator), check_masker_dict_unchanged)
        yield (clone(estimator), check_masker_dtypes)
        yield (clone(estimator), check_masker_empty_data_messages)
        yield (clone(estimator), check_masker_fit_score_takes_y)
        yield (clone(estimator), check_masker_fit_with_empty_mask)
        yield (
            clone(estimator),
            check_masker_fit_with_non_finite_in_mask,
        )
        yield (clone(estimator), check_masker_fitted)
        yield (clone(estimator), check_masker_generate_report)
        yield (clone(estimator), check_masker_generate_report_false)
        yield (clone(estimator), check_masker_inverse_transform)
        yield (clone(estimator), check_masker_transform_resampling)
        yield (clone(estimator), check_masker_mask_img)
        yield (clone(estimator), check_masker_mask_img_from_imgs)
        yield (clone(estimator), check_masker_no_mask_no_img)
        yield (clone(estimator), check_masker_refit)
        yield (clone(estimator), check_masker_smooth)
        yield (clone(estimator), check_masker_transformer)
        yield (
            clone(estimator),
            check_masker_transformer_high_variance_confounds,
        )

        if not is_multimasker(estimator):
            yield (clone(estimator), check_masker_clean)
            yield (clone(estimator), check_masker_detrending)
            yield (clone(estimator), check_masker_transformer_sample_mask)
            yield (clone(estimator), check_masker_with_confounds)

        if accept_niimg_input(estimator):
            yield (clone(estimator), check_nifti_masker_clean_error)
            yield (clone(estimator), check_nifti_masker_clean_warning)
            yield (clone(estimator), check_nifti_masker_dtype)
            yield (clone(estimator), check_nifti_masker_fit_transform)
            yield (clone(estimator), check_nifti_masker_fit_transform_5d)
            yield (clone(estimator), check_nifti_masker_fit_transform_files)
            yield (clone(estimator), check_nifti_masker_fit_with_3d_mask)
            yield (
                clone(estimator),
                check_nifti_masker_generate_report_after_fit_with_only_mask,
            )

            if is_multimasker(estimator):
                yield (
                    clone(estimator),
                    check_multi_nifti_masker_generate_report_4d_fit,
                )
                yield (
                    clone(estimator),
                    check_multi_masker_transformer_high_variance_confounds,
                )
                yield (
                    clone(estimator),
                    check_multi_masker_transformer_sample_mask,
                )
                yield (clone(estimator), check_multi_masker_with_confounds)

        if accept_surf_img_input(estimator):
            yield (clone(estimator), check_surface_masker_fit_with_mask)
            yield (clone(estimator), check_surface_masker_list_surf_images)

    if is_glm(estimator):
        yield (clone(estimator), check_glm_dtypes)
        yield (clone(estimator), check_glm_empty_data_messages)
        yield (clone(estimator), check_glm_is_fitted)


def get_tag(estimator: BaseEstimator, tag: str) -> bool:
    tags = estimator.__sklearn_tags__()
    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return tag in tags["X_types"]
    else:
        return getattr(tags.input_tags, tag, False)


def is_masker(estimator: BaseEstimator) -> bool:
    return get_tag(estimator, "masker")


def is_multimasker(estimator: BaseEstimator) -> bool:
    return get_tag(estimator, "multi_masker")


def is_glm(estimator: BaseEstimator) -> bool:
    return get_tag(estimator, "glm")


def accept_niimg_input(estimator: BaseEstimator) -> bool:
    return get_tag(estimator, "niimg_like")


def accept_surf_img_input(estimator: BaseEstimator) -> bool:
    return get_tag(estimator, "surf_img")


def _not_fitted_error_message(estimator):
    return (
        f"This {type(estimator).__name__} instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator."
    )


def fit_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Fit on a nilearn estimator with appropriate input and return it."""
    assert accept_niimg_input(estimator) or accept_surf_img_input(estimator)

    if is_glm(estimator):
        data, design_matrices = _make_surface_img_and_design()
        # FirstLevel
        if hasattr(estimator, "hrf_model"):
            return estimator.fit(data, design_matrices=design_matrices)
        # SecondLevel
        else:
            return estimator.fit(data, design_matrix=design_matrices)

    elif isinstance(estimator, SearchLight):
        n_samples = 30
        data = _rng().random((5, 5, 5, n_samples))
        # Create a condition array, with balanced classes
        y = np.arange(n_samples, dtype=int) >= (n_samples // 2)

        data[2, 2, 2, :] = 0
        data[2, 2, 2, y] = 2
        X = Nifti1Image(data, np.eye(4))

        return estimator.fit(X, y)

    elif is_classifier(estimator):
        dim = 5
        X, y = make_classification(
            n_samples=30,
            n_features=dim**3,
            scale=3.0,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X, _ = to_niimgs(X, [dim, dim, dim])
        return estimator.fit(X, y)

    elif is_regressor(estimator):
        dim = 5
        X, y = make_regression(
            n_samples=30,
            n_features=dim**3,
            n_informative=dim,
            noise=1.5,
            bias=1.0,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X)
        X, _ = to_niimgs(X, [dim, dim, dim])
        return estimator.fit(X, y)

    elif is_masker(estimator):
        if accept_niimg_input(estimator):
            imgs = Nifti1Image(_rng().random(_shape_3d_large()), _affine_eye())
        else:
            imgs = _make_surface_img(10)
        return estimator.fit(imgs)

    else:
        imgs = Nifti1Image(_rng().random(_shape_3d_large()), _affine_eye())
        return estimator.fit(imgs)


# ------------------ GENERIC CHECKS ------------------


def _check_mask_img_(estimator):
    if accept_niimg_input(estimator):
        assert isinstance(estimator.mask_img_, Nifti1Image)
    else:
        assert isinstance(estimator.mask_img_, SurfaceImage)
    load_mask_img(estimator.mask_img_)


def check_estimator_has_sklearn_is_fitted(estimator):
    """Check appropriate response to check_fitted from sklearn before fitting.

    check that before fitting
    - estimator has a __sklearn_is_fitted__ method
    - running sklearn check_is_fitted on estimator throws an error
    """
    if not hasattr(estimator, "__sklearn_is_fitted__"):
        raise TypeError(
            "All nilearn estimators must have __sklearn_is_fitted__ method."
        )

    if estimator.__sklearn_is_fitted__() is True:
        raise ValueError(
            "Estimator __sklearn_is_fitted__ must return False before fit."
        )

    with pytest.raises(ValueError, match=_not_fitted_error_message(estimator)):
        check_is_fitted(estimator)


def check_transformer_set_output(estimator):
    """Check that set_ouput throws a not implemented error."""
    if hasattr(estimator, "transform"):
        with pytest.raises(NotImplementedError):
            estimator.set_output(transform="default")


def check_fit_returns_self(estimator) -> None:
    """Check maskers return itself after fit.

    Replace sklearn check_estimators_fit_returns_self
    """
    # TODO make sure the following estimator pass this check
    if isinstance(
        estimator,
        (
            _BaseDecomposition,
            ReNA,
            HierarchicalKMeans,
            GroupSparseCovariance,
            GroupSparseCovarianceCV,
            ConnectivityMeasure,
        ),
    ):
        return None

    fitted_estimator = fit_estimator(estimator)

    assert fitted_estimator is estimator


def check_img_estimator_dont_overwrite_parameters(estimator) -> None:
    """Check that fit method only changes or sets private attributes.

    Only for estimator that work with images.

    Replaces check_dont_overwrite_parameters from sklearn.
    """
    estimator = clone(estimator)

    set_random_state(estimator, 1)

    dict_before_fit = estimator.__dict__.copy()

    fitted_estimator = fit_estimator(estimator)

    dict_after_fit = fitted_estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit
    ]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
        f"Estimator {estimator.__class__.__name__} "
        "adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        f" with _ but [{', '.join(attrs_added_by_fit)}] added"
    )

    # check that fit doesn't change any public attribute

    # nifti_maps_masker, nifti_maps_masker, nifti_spheres_masker
    # change memory parameters on fit if it's None
    keys_to_ignore = ["memory"]

    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
        and key not in keys_to_ignore
    ]

    assert not attrs_changed_by_fit, (
        f"Estimator {estimator.__class__.__name__} "
        "changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        f" [{', '.join(attrs_changed_by_fit)}] changed"
    )


def check_img_estimators_overwrite_params(estimator) -> None:
    """Check that we do not change or mutate the internal state of input.

    Replaces sklearn check_estimators_overwrite_params
    """
    estimator = clone(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    fitted_estimator = fit_estimator(estimator)

    # Compare the state of the model parameters with the original parameters
    new_params = fitted_estimator.get_params()

    # nifti_maps_masker, nifti_maps_masker, nifti_spheres_masker
    # change memory parameters on fit if it's None
    param_to_ignore = ["memory"]

    for param_name, original_value in original_params.items():
        if param_name in param_to_ignore:
            continue

        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            f"Estimator {estimator.__class__.__name__} "
            "should not change or mutate "
            f"the parameter {param_name} from {original_value} "
            f"to {new_value} during fit."
        )


# ------------------ DECODERS CHECKS ------------------


def check_image_estimator_requires_y_none(estimator) -> None:
    """Check estimator with requires_y=True fails gracefully for y=None.

    Replaces sklearn check_requires_y_none
    """
    expected_err_msgs = "requires y to be passed, but the target y is None"
    shape = (5, 5, 5) if isinstance(estimator, SearchLight) else (30, 31, 32)
    input_img = Nifti1Image(_rng().random(shape), _affine_eye())
    try:
        estimator.fit(input_img, None)
    except ValueError as ve:
        if all(msg not in str(ve) for msg in expected_err_msgs):
            raise ve


def check_image_supervised_estimator_y_no_nan(estimator) -> None:
    """Check estimator fails if y contains nan or inf.

    Replaces sklearn check_supervised_y_no_nan
    """
    dim = 5
    if isinstance(estimator, SearchLight):
        n_samples = 30
        # Create a condition array, with balanced classes
        y = np.arange(n_samples, dtype=int) >= (n_samples // 2)

        data = _rng().random((dim, dim, dim, n_samples))
        data[2, 2, 2, :] = 0
        data[2, 2, 2, y] = 2
        X = Nifti1Image(data, np.eye(4))

    else:
        # we can use classification data even for regressors
        # because fit should fail early
        X, y = make_classification(
            n_samples=20,
            n_features=dim**3,
            scale=3.0,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X, _ = to_niimgs(X, [dim, dim, dim])

    y = _rng().random(y.shape)

    for value in [np.inf, np.nan]:
        y[5,] = value
        with pytest.raises(ValueError, match="Input .*contains"):
            estimator.fit(X, y)


def check_decoder_empty_data_messages(estimator):
    """Check that empty images are caught properly.

    Replaces sklearn check_estimators_empty_data_messages.

    Not implemented for nifti data for performance reasons.
    See : https://github.com/nilearn/nilearn/pull/5293#issuecomment-2977170723
    """
    n_samples = 30
    if isinstance(estimator, SearchLight):
        # SearchLight do not support surface data directly
        return None

    else:
        # we can use classification data even for regressors
        # because fit should fail early
        dim = 5
        _, y = make_classification(
            n_samples=20,
            n_features=dim**3,
            scale=3.0,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )

    imgs = _make_surface_img(n_samples)
    data = {
        part: np.empty(0).reshape((imgs.data.parts[part].shape[0], 0))
        for part in imgs.data.parts
    }
    X = SurfaceImage(imgs.mesh, data)

    y = _rng().random(y.shape)

    with pytest.raises(ValueError, match="empty"):
        estimator.fit(X, y)


# ------------------ MASKER CHECKS ------------------


def check_masker_dict_unchanged(estimator):
    """Replace check_dict_unchanged from sklearn.

    transform() should not changed the dict of the object.
    """
    if accept_niimg_input(estimator):
        # We use a different shape here to force some maskers
        # to perform a resampling.
        shape = (30, 31, 32)
        input_img = Nifti1Image(_rng().random(shape), _affine_eye())
    else:
        input_img = _make_surface_img(10)

    estimator = estimator.fit(input_img)

    dict_before = estimator.__dict__.copy()

    estimator.transform(input_img)

    dict_after = estimator.__dict__

    # TODO NiftiLabelsMasker is modified at transform time
    # see issue https://github.com/nilearn/nilearn/issues/2720
    if isinstance(estimator, (NiftiLabelsMasker)):
        with pytest.raises(AssertionError):
            assert dict_after == dict_before
    else:
        # The following try / except is mostly
        # to give more informative error messages when this check fails.
        try:
            assert dict_after == dict_before
        except AssertionError as e:
            unmatched_keys = set(dict_after.keys()) ^ set(dict_before.keys())
            if len(unmatched_keys) > 0:
                raise ValueError(
                    "Estimator changes '__dict__' keys during transform.\n"
                    f"{unmatched_keys} \n"
                )

            difference = {}
            for x in dict_before:
                if type(dict_before[x]) is not type(dict_after[x]):
                    difference[x] = {
                        "before": dict_before[x],
                        "after": dict_after[x],
                    }
                    continue
                if (
                    isinstance(dict_before[x], np.ndarray)
                    and not np.array_equal(dict_before[x], dict_after[x])
                    and not check_imgs_equal(dict_before[x], dict_after[x])
                ) or (
                    not isinstance(dict_before[x], (np.ndarray, Nifti1Image))
                    and dict_before[x] != dict_after[x]
                ):
                    difference[x] = {
                        "before": dict_before[x],
                        "after": dict_after[x],
                    }
                    continue
            if difference:
                raise ValueError(
                    "Estimator changes the following '__dict__' keys \n"
                    "during transform.\n"
                    f"{difference}"
                )
            else:
                raise e
        except Exception as e:
            raise e


def check_masker_fitted(estimator):
    """Check appropriate response of maskers to check_fitted from sklearn.

    Should act as a replacement in the case of the maskers
    for sklearn's check_fit_check_is_fitted

    check that before fitting
    - transform() and inverse_transform() \
      throw same error

    check that after fitting
    - __sklearn_is_fitted__ returns true
    - running sklearn check_fitted throws no error
    - masker have a n_elements_ attribute that is positive int
    """
    # Failure should happen before the input type is determined
    # so we can pass nifti image to surface maskers.
    with pytest.raises(ValueError, match=_not_fitted_error_message(estimator)):
        estimator.transform(_img_3d_rand())
    with pytest.raises(ValueError, match=_not_fitted_error_message(estimator)):
        estimator.transform_single_imgs(_img_3d_rand())
    if is_multimasker(estimator):
        with pytest.raises(
            ValueError, match=_not_fitted_error_message(estimator)
        ):
            estimator.transform_imgs([_img_3d_rand()])

    # Failure should happen before the size of the input type is determined
    # so we can pass any array here.
    signals = np.ones((10, 11))
    with pytest.raises(ValueError, match=_not_fitted_error_message(estimator)):
        estimator.inverse_transform(signals)

    # NiftiMasker and SurfaceMasker cannot accept None on fit
    if accept_niimg_input(estimator):
        estimator.fit(_img_3d_rand())
    else:
        estimator.fit(_make_surface_img(10))

    assert estimator.__sklearn_is_fitted__()

    check_is_fitted(estimator)

    assert isinstance(estimator.n_elements_, int) and estimator.n_elements_ > 0


def check_masker_clean_kwargs(estimator):
    """Check attributes for cleaning.

    Maskers accept a clean_args dict
    and store in clean_args and contains parameters to pass to clean.
    """
    assert estimator.clean_args is None


def check_masker_detrending(estimator):
    """Check detrending does something.

    Fit transform on same input should give different results
    if detrend is true or false.
    """
    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(100)

    signal = estimator.fit_transform(input_img)

    estimator.detrend = True
    detrended_signal = estimator.fit_transform(input_img)

    assert_raises(AssertionError, assert_array_equal, detrended_signal, signal)


def check_masker_compatibility_mask_image(estimator):
    """Check compatibility of the mask_img and images to masker.

    Compatibility should be check at fit and transform time.

    For nifti maskers this is handled by one the check_nifti functions.
    For surface maskers, check_compatibility_mask_and_images does it.
    But this means we do not have exactly the same error messages.
    """
    if accept_niimg_input(estimator):
        mask_img = _img_mask_mni()
        input_img = _make_surface_img()
    else:
        mask_img = _make_surface_mask()
        input_img = _img_3d_mni()

    estimator.mask_img = mask_img
    with pytest.raises(TypeError):
        estimator.fit(input_img)

    if accept_niimg_input(estimator):
        # using larger images to be compatible
        # with regions extraction tests
        mask = np.zeros(_shape_3d_large(), dtype=np.int8)
        mask[1:-1, 1:-1, 1:-1] = 1
        mask_img = Nifti1Image(mask, _affine_eye())
        image_to_transform = _make_surface_img()
    else:
        mask_img = _make_surface_mask()
        image_to_transform = _img_3d_mni()

    estimator = clone(estimator)
    estimator.mask_img = mask_img
    estimator.fit()
    with pytest.raises(TypeError):
        estimator.transform(image_to_transform)

    _check_mask_img_(estimator)


def check_masker_no_mask_no_img(estimator):
    """Check maskers mask_img_ when no mask passed at init or imgs at fit.

    For (Multi)NiftiMasker and SurfaceMasker fit should raise ValueError.
    For all other maskers mask_img_ should be None after fit.
    """
    assert not hasattr(estimator, "mask_img_")

    if isinstance(estimator, (NiftiMasker, SurfaceMasker)):
        with pytest.raises(
            ValueError, match="Parameter 'imgs' must be provided to "
        ):
            estimator.fit()
    else:
        estimator.fit()
        assert estimator.mask_img_ is None


def check_masker_mask_img_from_imgs(estimator):
    """Check maskers mask_img_ inferred from imgs when no mask is provided.

    For (Multi)NiftiMasker and SurfaceMasker:
    they must have a valid mask_img_ after fit.
    For all other maskers mask_img_ should be None after fit.
    """
    if accept_niimg_input(estimator):
        # Small image with shape=(7, 8, 9) would fail with MultiNiftiMasker
        # giving mask_img_that mask all the data : do not know why!!!
        input_img = Nifti1Image(
            _rng().random(_shape_3d_large()), _affine_mni()
        )

    else:
        input_img = _make_surface_img(2)

    # Except for (Multi)NiftiMasker and SurfaceMasker,
    # maskers have mask_img_ = None after fitting some input image
    # when no mask was passed at construction
    estimator = clone(estimator)
    assert not hasattr(estimator, "mask_img_")

    estimator.fit(input_img)

    if isinstance(estimator, (NiftiMasker, SurfaceMasker)):
        _check_mask_img_(estimator)
    else:
        assert estimator.mask_img_ is None


def check_masker_mask_img(estimator):
    """Check maskers mask_img_ post fit is valid.

    If a mask is passed at construction,
    then mask_img_ should be a valid mask after fit.

    Maskers should be fittable
    even when passing a non-binary image
    with multiple samples (4D for volume, 2D for surface) as mask.
    Resulting mask_img_ should be binary and have a single sample.
    """
    if accept_niimg_input(estimator):
        # Small image with shape=(7, 8, 9) would fail with MultiNiftiMasker
        # giving mask_img_that mask all the data : do not know why!!!
        mask_data = np.zeros(_shape_3d_large(), dtype="int8")
        mask_data[2:-2, 2:-2, 2:-2] = 1
        binary_mask_img = Nifti1Image(mask_data, _affine_eye())

        input_img = Nifti1Image(
            _rng().random(_shape_3d_large()), _affine_eye()
        )

        non_binary_mask_img = Nifti1Image(
            _rng().random((*_shape_3d_large(), 2)), _affine_eye()
        )

    else:
        binary_mask_img = _make_surface_mask()
        non_binary_mask_img = _make_surface_img()

        input_img = _make_surface_img(2)

    # happy path
    estimator = clone(estimator)
    estimator.mask_img = binary_mask_img
    assert not hasattr(estimator, "mask_img_")

    estimator.fit()

    _check_mask_img_(estimator)

    # use non binary multi-sample image as mask
    estimator = clone(estimator)
    estimator.mask_img = non_binary_mask_img
    assert not hasattr(estimator, "mask_img_")

    estimator.fit()

    _check_mask_img_(estimator)

    # use mask at init and imgs at fit
    # mask at init should prevail
    estimator = clone(estimator)
    estimator.mask_img = binary_mask_img

    estimator.fit()
    ref_mask_img_ = estimator.mask_img_

    estimator = clone(estimator)
    estimator.mask_img = binary_mask_img

    assert not hasattr(estimator, "mask_img_")

    if isinstance(estimator, (NiftiMasker, SurfaceMasker)):
        with pytest.warns(
            UserWarning,
            match=(
                "Generation of a mask has been requested .* "
                "while a mask was given at masker creation."
            ),
        ):
            estimator.fit(input_img)
    else:
        estimator.fit(input_img)

    _check_mask_img_(estimator)
    if accept_niimg_input(estimator):
        assert_array_equal(
            ref_mask_img_.get_fdata(), estimator.mask_img_.get_fdata()
        )
    else:
        assert_array_equal(
            get_surface_data(ref_mask_img_),
            get_surface_data(estimator.mask_img_),
        )


def check_masker_clean(estimator):
    """Check that cleaning does something on fit transform.

    Fit transform on same input should give different results
    if some cleaning parameters are passed.
    """
    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(100)

    signal = estimator.fit_transform(input_img)

    estimator.t_r = 2.0
    estimator.high_pass = 1 / 128
    estimator.clean_args = {"filter": "cosine"}
    detrended_signal = estimator.fit_transform(input_img)

    assert_raises(AssertionError, assert_array_equal, detrended_signal, signal)


def check_masker_transformer(estimator):
    """Replace sklearn _check_transformer for maskers.

    - for maskers transform is in the base class and
      implemented via a transform_single_imgs
    - checks that "imgs" (and not X) is the parameter
      for input for fit / transform
    - fit_transform method should work on non fitted estimator
    - fit_transform should give same result as fit then transform
    """
    # transform_single_imgs should not be an abstract method anymore
    assert not getattr(
        estimator.transform_single_imgs, "__isabstractmethod__", False
    )

    for attr in ["fit", "transform", "fit_transform"]:
        tmp = dict(**inspect.signature(getattr(estimator, attr)).parameters)
        assert next(iter(tmp)) == "imgs"
        assert "X" not in tmp

    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(100)

    signal_1 = estimator.fit_transform(input_img)

    estimator = clone(estimator)
    signal_2 = estimator.fit(input_img).transform(input_img)

    assert_array_equal(signal_1, signal_2)


def check_masker_transformer_high_variance_confounds(estimator):
    """Check high_variance_confounds use in maskers.

    Make sure that using high_variance_confounds returns different result.

    Ensure that high_variance_confounds can be used with regular confounds,
    and that results are different than when just using the confounds alone.
    """
    length = 10

    if accept_niimg_input(estimator):
        data = _rng().random((*_shape_3d_default(), length))
        input_img = Nifti1Image(data, _affine_eye())
    else:
        input_img = _make_surface_img(length)

    estimator.high_variance_confounds = False

    signal = estimator.fit_transform(input_img)

    estimator = clone(estimator)
    estimator.high_variance_confounds = True

    signal_hvc = estimator.fit_transform(input_img)

    assert_raises(AssertionError, assert_array_equal, signal, signal_hvc)

    with TemporaryDirectory() as tmp_dir:
        array = _rng().random((length, 3))

        dataframe = pd.DataFrame(array)

        tmp_dir = Path(tmp_dir)
        dataframe.to_csv(tmp_dir / "confounds.csv")

        for c in [array, dataframe, tmp_dir / "confounds.csv"]:
            confounds = [c] if is_multimasker(estimator) else c

            estimator = clone(estimator)
            estimator.high_variance_confounds = False
            signal_c = estimator.fit_transform(input_img, confounds=confounds)

            estimator = clone(estimator)
            estimator.high_variance_confounds = True
            signal_c_hvc = estimator.fit_transform(
                input_img, confounds=confounds
            )

            assert_raises(
                AssertionError, assert_array_equal, signal_c, signal_c_hvc
            )


def check_masker_transformer_sample_mask(estimator):
    """Check sample_mask use in maskers.

    Make sure that using sample_mask returns different result
    compare to when it's not used.

    Try different types of sample_mask
    that always keep the same samples (sample 1, 2 and 4)
    that should all return the same thing.
    """
    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye()
    else:
        input_img = _make_surface_img(5)

    estimator.fit(input_img)
    signal_1 = estimator.transform(input_img, sample_mask=None)

    assert signal_1.ndim == 2

    # index sample to keep
    sample_mask = np.asarray([1, 2, 4])

    signal_2 = estimator.transform(input_img, sample_mask=sample_mask)

    assert signal_2.shape[0] == 3

    assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

    # logical indexing
    n_sample = signal_1.shape[0]
    sample_mask = np.full((n_sample,), True)
    np.put(sample_mask, [0, 3], [False, False])

    signal_3 = estimator.transform(input_img, sample_mask=sample_mask)

    assert_array_equal(signal_2, signal_3)

    # list of explicit index
    sample_mask = [[1, 2, 4]]

    signal_4 = estimator.transform(input_img, sample_mask=sample_mask)

    assert_array_equal(signal_2, signal_4)

    # list of logical index
    sample_mask = [[False, True, True, False, True]]

    signal_5 = estimator.transform(input_img, sample_mask=sample_mask)

    assert_array_equal(signal_2, signal_5)


def check_masker_with_confounds(estimator):
    """Test fit_transform with confounds.

    Check different types of confounds
    (array, dataframe, str or path to txt, csv, tsv)
    and ensure results is different
    than when not using confounds.

    Check proper errors are raised if file is not found
    or if confounds do not match signal length.

    For more tests see those of signal.clean.
    """
    length = 20
    if accept_niimg_input(estimator):
        input_img = Nifti1Image(
            _rng().random((4, 5, 6, length)), affine=_affine_eye()
        )
    else:
        input_img = _make_surface_img(length)

    signal_1 = estimator.fit_transform(input_img, confounds=None)

    array = _rng().random((length, 3))

    dataframe = pd.DataFrame(array)

    confounds_path = nilearn_dir() / "tests" / "data" / "spm_confounds.txt"

    for confounds in [array, dataframe, confounds_path, str(confounds_path)]:
        signal_2 = estimator.fit_transform(input_img, confounds=confounds)

        assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        dataframe.to_csv(tmp_dir / "confounds.csv")
        signal_2 = estimator.fit_transform(
            input_img, confounds=tmp_dir / "confounds.csv"
        )

        assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

        dataframe.to_csv(tmp_dir / "confounds.tsv", sep="\t")
        signal_2 = estimator.fit_transform(
            input_img, confounds=tmp_dir / "confounds.tsv"
        )

        assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

    with pytest.raises(FileNotFoundError):
        estimator.fit_transform(input_img, confounds="not_a_file.txt")

    with pytest.raises(
        ValueError, match="Confound signal has an incorrect length"
    ):
        estimator.fit_transform(
            input_img, confounds=_rng().random((length * 2, 3))
        )


def check_masker_refit(estimator):
    """Check masker can be refitted and give different results."""
    if accept_niimg_input(estimator):
        # using larger images to be compatible
        # with regions extraction tests
        mask = np.zeros(_shape_3d_large(), dtype=np.int8)
        mask[1:-1, 1:-1, 1:-1] = 1
        mask_img_1 = Nifti1Image(mask, _affine_eye())

        mask = np.zeros(_shape_3d_large(), dtype=np.int8)
        mask[3:-3, 3:-3, 3:-3] = 1
        mask_img_2 = Nifti1Image(mask, _affine_eye())
    else:
        mask_img_1 = _make_surface_mask()
        data = {
            part: np.ones(mask_img_1.data.parts[part].shape)
            for part in mask_img_1.data.parts
        }
        mask_img_2 = SurfaceImage(mask_img_1.mesh, data)

    estimator.mask_img = mask_img_1
    estimator.fit()
    fitted_mask_1 = estimator.mask_img_

    estimator.mask_img = mask_img_2
    estimator.fit()
    fitted_mask_2 = estimator.mask_img_

    if accept_niimg_input(estimator):
        with pytest.raises(AssertionError):
            assert_array_equal(
                fitted_mask_1.get_fdata(), fitted_mask_2.get_fdata()
            )
    else:
        with pytest.raises(ValueError):
            assert_surface_image_equal(fitted_mask_1, fitted_mask_2)


def check_masker_empty_data_messages(estimator):
    """Check that empty images are caught properly.

    Replaces sklearn check_estimators_empty_data_messages.

    Not implemented for nifti maskers for performance reasons.
    See : https://github.com/nilearn/nilearn/pull/5293#issuecomment-2977170723
    """
    if accept_niimg_input(estimator):
        return None

    else:
        imgs = _make_surface_img()
        data = {
            part: np.empty(0).reshape((imgs.data.parts[part].shape[0], 0))
            for part in imgs.data.parts
        }
        imgs = SurfaceImage(imgs.mesh, data)

        mask_img = _make_surface_mask()

    with pytest.raises(ValueError, match="empty"):
        estimator.fit(imgs)

    estimator.mask_img = mask_img
    estimator.fit()
    with pytest.raises(ValueError, match="empty"):
        estimator.transform(imgs)


# ------------------ SURFACE MASKER CHECKS ------------------


def check_masker_fit_with_empty_mask(estimator):
    """Check mask that excludes all voxels raise an error."""
    if accept_niimg_input(estimator):
        mask_img = _img_3d_zeros()
        imgs = [_img_3d_rand()]
    else:
        mask_img = _make_surface_mask()
        for k, v in mask_img.data.parts.items():
            mask_img.data.parts[k] = np.zeros(v.shape)
        imgs = _make_surface_img(1)

    estimator.mask_img = mask_img
    with pytest.raises(
        ValueError,
        match="The mask is invalid as it is empty: it masks all data",
    ):
        estimator.fit(imgs)


def check_masker_fit_with_non_finite_in_mask(estimator):
    """Check mask with non finite values can be used with maskers.

    - Warning is thrown.
    - Output of transform must contain only finite values.
    """
    if accept_niimg_input(estimator):
        # _shape_3d_large() is used,
        # this test would fail for RegionExtractor otherwise
        mask = np.ones(_shape_3d_large())
        mask[:, :, 7] = np.nan
        mask[:, :, 4] = np.inf
        mask_img = Nifti1Image(mask, affine=_affine_eye())

        imgs = _img_3d_rand()

    else:
        mask_img = _make_surface_mask()
        for k, v in mask_img.data.parts.items():
            mask_img.data.parts[k] = np.zeros(v.shape)
        mask_img.data.parts["left"][0:3, 0] = [np.nan, np.inf, 1]
        mask_img.data.parts["right"][0:3, 0] = [np.nan, np.inf, 1]

        imgs = _make_surface_img(1)

    estimator.mask_img = mask_img
    with pytest.warns(UserWarning, match="Non-finite values detected."):
        estimator.fit()

    signal = estimator.transform(imgs)
    assert np.all(np.isfinite(signal))


def check_masker_dtypes(estimator):
    """Check masker can fit/transform with inputs of varying dtypes.

    Replacement for sklearn check_estimators_dtypes.

    np.int64 not tested: see no_int64_nifti in nilearn/conftest.py
    """
    length = 20
    for dtype in [np.float32, np.float64, np.int32]:
        estimator = clone(estimator)

        if accept_niimg_input(estimator):
            data = np.zeros((*_shape_3d_large(), length))
            data[1:28, 1:28, 1:28, ...] = (
                _rng().random((27, 27, 27, length)) + 2.0
            )
            imgs = Nifti1Image(data.astype(dtype), affine=_affine_eye())

        else:
            imgs = _make_surface_img(length)
            for k, v in imgs.data.parts.items():
                imgs.data.parts[k] = v.astype(dtype)

        estimator.fit(imgs)
        estimator.transform(imgs)


def check_masker_smooth(estimator):
    """Check that masker can smooth data when extracting.

    Check that masker instance has smoothing_fwhm attribute.
    Check that output is different with and without smoothing.

    For Surface maskers:
    - Check smoothing on surface maskers raises NotImplemented warning.
    - Check that output is the same with and without smoothing.
    TODO: update once smoothing is implemented.
    """
    assert hasattr(estimator, "smoothing_fwhm")

    if accept_niimg_input(estimator):
        imgs = _img_3d_rand()
    else:
        n_sample = 1
        imgs = _make_surface_img(n_sample)

    signal = estimator.fit_transform(imgs)

    estimator.smoothing_fwhm = 3
    estimator.fit(imgs)

    if accept_niimg_input(estimator):
        smoothed_signal = estimator.transform(imgs)

        assert_raises(
            AssertionError, assert_array_equal, smoothed_signal, signal
        )

    else:
        with pytest.warns(UserWarning, match="not yet supported"):
            smoothed_signal = estimator.transform(imgs)

        assert_array_equal(smoothed_signal, signal)


def check_masker_inverse_transform(estimator) -> None:
    """Check output of inverse_transform.

    For signal with 1 or more samples.

    For nifti maskers:
        - 1D arrays -> 3D images
        - 2D arrays -> 4D images

    For surface maskers:
        - 1D arrays -> 1D images
        - 2D arrays -> 2D images

    Check that running transform() is not required to run inverse_transform().

    Check that running inverse_transform() before and after running transform()
    give same result.

    Check that the proper error is thrown, if signal has the wrong shape.
    """
    if accept_niimg_input(estimator):
        # using different shape for imgs, mask
        # to force resampling
        input_shape = (28, 29, 30)
        imgs = Nifti1Image(_rng().random(input_shape), _affine_eye())

        mask_shape = (15, 16, 17)
        mask_img = Nifti1Image(np.ones(mask_shape), _affine_eye())

        if isinstance(estimator, NiftiSpheresMasker):
            tmp = mask_img.shape
        else:
            tmp = input_shape
        expected_shapes = [tmp, (*tmp, 1), (*tmp, 10)]

    else:
        imgs = _make_surface_img(1)

        mask_img = _make_surface_mask()

        expected_shapes = [
            (imgs.shape[0],),
            (imgs.shape[0], 1),
            (imgs.shape[0], 10),
        ]

    for i, expected_shape in enumerate(
        expected_shapes,
    ):
        estimator = clone(estimator)

        if isinstance(estimator, (NiftiSpheresMasker)):
            estimator.mask_img = mask_img

        estimator.fit(imgs)

        if i == 0:
            signals = _rng().random((estimator.n_elements_,))
        elif i == 1:
            signals = _rng().random((1, estimator.n_elements_))
        elif i == 2:
            signals = _rng().random((10, estimator.n_elements_))

        new_imgs = estimator.inverse_transform(signals)

        if accept_niimg_input(estimator):
            actual_shape = new_imgs.shape
            assert_array_almost_equal(imgs.affine, new_imgs.affine)
        else:
            actual_shape = new_imgs.data.shape
        assert actual_shape == expected_shape

        # same result before and after running transform()
        estimator.transform(imgs)

        new_imgs_2 = estimator.inverse_transform(signals)

        if accept_niimg_input(estimator):
            assert check_imgs_equal(new_imgs, new_imgs_2)
        else:
            assert_surface_image_equal(new_imgs, new_imgs_2)

    signals = _rng().random((1, estimator.n_elements_ + 1))
    with pytest.raises(
        ValueError, match="Input to 'inverse_transform' has wrong shape."
    ):
        estimator.inverse_transform(signals)


def check_masker_transform_resampling(estimator) -> None:
    """Check transform / inverse_transform for maskers with resampling.

    Similar to check_masker_inverse_transform
    but for nifti masker that can do some resampling
    (labels and maps maskers).

    Check that output has the shape of the data or the labels/maps image
    depending on which resampling_target was requested at init.

    Check that using a mask does not affect shape of output.

    Check that running transform() is not required to run inverse_transform().

    Check that running inverse_transform() before and after running transform()
    give same result.

    Check that running transform on images with different fov
    than those used at fit is possible.
    """
    if not hasattr(estimator, "resampling_target"):
        return None

    # using different shape for imgs, mask
    # to force resampling
    n_sample = 10
    input_shape = (28, 29, 30, n_sample)
    imgs = Nifti1Image(_rng().random(input_shape), _affine_eye())

    imgs2 = Nifti1Image(_rng().random((20, 21, 22)), _affine_eye())

    mask_shape = (15, 16, 17)
    mask_img = Nifti1Image(np.ones(mask_shape), _affine_eye())

    for resampling_target in ["data", "labels"]:
        expected_shape = input_shape
        if resampling_target == "labels":
            if isinstance(estimator, NiftiMapsMasker):
                expected_shape = (*estimator.maps_img.shape[:3], n_sample)
                resampling_target = "maps"
            else:
                expected_shape = (*estimator.labels_img.shape, n_sample)

        for mask in [None, mask_img]:
            estimator = clone(estimator)
            estimator.resampling_target = resampling_target
            estimator.mask_img = mask

            # no resampling warning at fit time
            with warnings.catch_warnings(record=True) as warning_list:
                estimator.fit(imgs)
            assert all(
                "at transform time" not in str(x.message) for x in warning_list
            )

            signals = _rng().random((n_sample, estimator.n_elements_))

            new_imgs = estimator.inverse_transform(signals)

            assert_array_almost_equal(imgs.affine, new_imgs.affine)
            actual_shape = new_imgs.shape
            assert actual_shape == expected_shape

            # no resampling warning when using same imgs as for fit()
            with warnings.catch_warnings(record=True) as warning_list:
                estimator.transform(imgs)
            assert all(
                "at transform time" not in str(x.message) for x in warning_list
            )

            # same result before and after running transform()
            new_imgs_2 = estimator.inverse_transform(signals)

            assert check_imgs_equal(new_imgs, new_imgs_2)

            # no error transforming an image with different fov
            # than the one used at fit time,
            # but there should be a resampling warning
            # we are resampling to data
            with warnings.catch_warnings(record=True) as warning_list:
                estimator.transform(imgs2)
            if resampling_target == "data":
                assert any(
                    "at transform time" in str(x.message) for x in warning_list
                )
            else:
                assert all(
                    "at transform time" not in str(x.message)
                    for x in warning_list
                )


def check_masker_fit_score_takes_y(estimator):
    """Replace sklearn check_fit_score_takes_y for maskers.

    Check that all estimators accept an optional y
    in fit and score so they can be used in pipelines.
    """
    for attr in ["fit", "fit_transform"]:
        tmp = {
            k: v.default
            for k, v in inspect.signature(
                getattr(estimator, attr)
            ).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        if "y" not in tmp:
            raise ValueError(
                f"{estimator.__class__.__name__} "
                f"is missing 'y=None' for the method '{attr}'."
            )
        assert tmp["y"] is None


# ------------------ SURFACE MASKER CHECKS ------------------


def check_surface_masker_fit_with_mask(estimator):
    """Check fit / transform with mask provided at init.

    Check with 2D and 1D images.

    1D image -> 1D array
    2D image -> 2D array

    Also check 'shape' errors between images to fit and mask.
    """
    mask_img = _make_surface_mask()

    # 1D image
    mesh = _make_mesh()
    data = {}
    for k, v in mesh.parts.items():
        data_shape = (v.n_vertices,)
        data[k] = _rng().random(data_shape)
    imgs = SurfaceImage(mesh, data)
    assert imgs.shape == (9,)
    estimator.fit(imgs)

    signal = estimator.transform(imgs)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (estimator.n_elements_,)

    # 2D image with 1 sample
    imgs = _make_surface_img(1)
    estimator.mask_img = mask_img
    estimator.fit(imgs)

    signal = estimator.transform(imgs)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (1, estimator.n_elements_)

    # 2D image with several samples
    imgs = _make_surface_img(5)
    estimator = clone(estimator)
    estimator.mask_img = mask_img
    estimator.fit(imgs)

    signal = estimator.transform(imgs)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (5, estimator.n_elements_)

    # errors
    with pytest.raises(
        MeshDimensionError,
        match="Number of vertices do not match for between meshes.",
    ):
        estimator.fit(_flip_surf_img(imgs))
    with pytest.raises(
        MeshDimensionError,
        match="Number of vertices do not match for between meshes.",
    ):
        estimator.transform(_flip_surf_img(imgs))

    with pytest.raises(
        MeshDimensionError, match="PolyMeshes do not have the same keys."
    ):
        estimator.fit(_drop_surf_img_part(imgs))
    with pytest.raises(
        MeshDimensionError, match="PolyMeshes do not have the same keys."
    ):
        estimator.transform(_drop_surf_img_part(imgs))


def check_surface_masker_list_surf_images(estimator):
    """Test transform / inverse_transform on list of surface images.

    Check that 1D or 2D mask work.

    transform
    - list of 1D -> 2D array
    - list of 2D -> 2D array
    """
    n_sample = 5
    images_to_transform = [
        [_make_surface_img()] * 5,
        [_make_surface_img(2), _make_surface_img(3)],
    ]
    for imgs in images_to_transform:
        for mask_img in [None, _surf_mask_1d(), _make_surface_mask()]:
            estimator.mask_img = mask_img

            estimator = estimator.fit(imgs)

            signals = estimator.transform(imgs)

            assert signals.shape == (n_sample, estimator.n_elements_)

            img = estimator.inverse_transform(signals)

            assert img.shape == (_make_surface_img().mesh.n_vertices, n_sample)


# ------------------ NIFTI MASKER CHECKS ------------------


def check_nifti_masker_fit_transform(estimator):
    """Run several checks on maskers.

    - can fit 3D / 4D image
    - fitted maskers can transform:
      - 3D image
      - list of 3D images with same affine
    - array from transformed 3D images should have 1D
    - array from transformed 4D images should have 2D
    """
    estimator.fit(_img_3d_rand())

    # 3D images
    signal = estimator.transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (estimator.n_elements_,)

    signal_2 = estimator.fit_transform(_img_3d_rand())

    assert_array_equal(signal, signal_2)

    # list of 3D images
    signal = estimator.transform([_img_3d_rand(), _img_3d_rand()])

    if is_multimasker(estimator):
        assert isinstance(signal, list)
        assert len(signal) == 2
        for x in signal:
            assert isinstance(x, np.ndarray)
            assert x.ndim == 1
            assert x.shape == (estimator.n_elements_,)
    else:
        assert isinstance(signal, np.ndarray)
        assert signal.ndim == 2
        assert signal.shape[1] == estimator.n_elements_

    # 4D images
    signal = estimator.transform(_img_4d_rand_eye())

    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 2
    assert signal.shape == (_img_4d_rand_eye().shape[3], estimator.n_elements_)


def check_nifti_masker_fit_transform_5d(estimator):
    """Run checks on nifti maskers for transforming 5D images.

    - multi masker should be fine
      and return a list of 2D numpy arrays
    - non multimasker should fail
    """
    n_subject = 3

    estimator.fit(_img_3d_rand())

    input_5d_img = [_img_4d_rand_eye() for _ in range(n_subject)]

    if not is_multimasker(estimator):
        with pytest.raises(
            DimensionError,
            match="Input data has incompatible dimensionality: "
            "Expected dimension is 4D and you provided "
            "a list of 4D images \\(5D\\).",
        ):
            estimator.transform(input_5d_img)

        with pytest.raises(
            DimensionError,
            match="Input data has incompatible dimensionality: "
            "Expected dimension is 4D and you provided "
            "a list of 4D images \\(5D\\).",
        ):
            estimator.fit_transform(input_5d_img)

    else:
        signal = estimator.transform(input_5d_img)

        assert isinstance(signal, list)
        assert all(isinstance(x, np.ndarray) for x in signal)
        assert len(signal) == n_subject
        assert all(x.ndim == 2 for x in signal)

        signal = estimator.fit_transform(input_5d_img)

        assert isinstance(signal, list)
        assert all(isinstance(x, np.ndarray) for x in signal)
        assert len(signal) == n_subject
        assert all(x.ndim == 2 for x in signal)


def check_nifti_masker_clean_error(estimator):
    """Nifti maskers cannot be given cleaning parameters \
        via both clean_args and kwargs simultaneously.

    TODO remove after nilearn 0.13.0
    """
    input_img = _img_4d_rand_eye_medium()

    estimator.t_r = 2.0
    estimator.high_pass = 1 / 128
    estimator.clean_kwargs = {"clean__filter": "cosine"}
    estimator.clean_args = {"filter": "cosine"}

    error_msg = (
        "Passing arguments via 'kwargs' "
        "is mutually exclusive with using 'clean_args'"
    )
    with pytest.raises(ValueError, match=error_msg):
        estimator.fit(input_img)


def check_nifti_masker_clean_warning(estimator):
    """Nifti maskers raise warning if cleaning parameters \
        passed via kwargs.

        But this still affects the transformed signal.

    TODO remove after nilearn 0.13.0
    """
    input_img = _img_4d_rand_eye_medium()

    signal = estimator.fit_transform(input_img)

    estimator.t_r = 2.0
    estimator.high_pass = 1 / 128
    estimator.clean_kwargs = {"clean__filter": "cosine"}

    with pytest.warns(DeprecationWarning, match="You passed some kwargs"):
        estimator.fit(input_img)

    detrended_signal = estimator.transform(input_img)

    assert_raises(AssertionError, assert_array_equal, detrended_signal, signal)


def check_nifti_masker_fit_transform_files(estimator):
    """Check that nifti maskers can work directly on files."""
    with TemporaryDirectory() as tmp_dir:
        filename = write_imgs_to_path(
            _img_3d_rand(),
            file_path=Path(tmp_dir),
            create_files=True,
        )

        estimator.fit(filename)
        estimator.transform(filename)
        estimator.fit_transform(filename)


def check_nifti_masker_dtype(estimator):
    """Check dtype of output of maskers."""
    data_32 = _rng().random(_shape_3d_default(), dtype=np.float32)
    affine_32 = np.eye(4, dtype=np.float32)
    img_32 = Nifti1Image(data_32, affine_32)

    data_64 = _rng().random(_shape_3d_default(), dtype=np.float64)
    affine_64 = np.eye(4, dtype=np.float64)
    img_64 = Nifti1Image(data_64, affine_64)

    for img in [img_32, img_64]:
        estimator = clone(estimator)
        estimator.dtype = "auto"
        assert estimator.fit_transform(img).dtype == np.float32

    for img in [img_32, img_64]:
        estimator = clone(estimator)
        estimator.dtype = "float64"
        assert estimator.fit_transform(img).dtype == np.float64


def check_nifti_masker_fit_with_3d_mask(estimator):
    """Check 3D mask can be used with nifti maskers.

    Mask can have different shape than fitted image.
    """
    # _shape_3d_large() is used
    # this test would fail for RegionExtractor otherwise
    mask = np.ones(_shape_3d_large())
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit([_img_3d_rand()])

    assert hasattr(estimator, "mask_img_")


# ------------------ MULTI NIFTI MASKER CHECKS ------------------


def check_multi_masker_with_confounds(estimator):
    """Test multi maskers with a list of confounds.

    Ensure results is different than when not using confounds.

    Check that confounds are applied when passing a 4D image (not iterable)
    to transform.

    Check that error is raised if number of confounds
    does not match number of images.
    """
    length = _img_4d_rand_eye_medium().shape[3]

    array = _rng().random((length, 3))

    signals_list_1 = estimator.fit_transform(
        [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
    )
    signals_list_2 = estimator.fit_transform(
        [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
        confounds=[array, array],
    )

    for signal_1, signal_2 in zip(signals_list_1, signals_list_2):
        assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

    # should also work with a single 4D image (has no __iter__ )
    signals_list_1 = estimator.fit_transform(_img_4d_rand_eye_medium())
    signals_list_2 = estimator.fit_transform(
        _img_4d_rand_eye_medium(),
        confounds=[array],
    )
    for signal_1, signal_2 in zip(signals_list_1, signals_list_2):
        assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)

    # Mismatch n imgs and n confounds
    with pytest.raises(
        ValueError, match="number of confounds .* unequal to number of images"
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            confounds=[array],
        )

    with pytest.raises(
        TypeError, match="'confounds' must be a None or a list."
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            confounds=1,
        )


def check_multi_masker_transformer_sample_mask(estimator):
    """Test multi maskers with a list of "sample_mask".

    "sample_mask" was directly sent as input to the parallel calls of
    "transform_single_imgs" instead of sending iterations.
    See https://github.com/nilearn/nilearn/issues/3967 for more details.
    """
    length = _img_4d_rand_eye_medium().shape[3]

    n_scrub1 = 3
    n_scrub2 = 2

    sample_mask1 = np.arange(length - n_scrub1)
    sample_mask2 = np.arange(length - n_scrub2)

    signals_list = estimator.fit_transform(
        [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
        sample_mask=[sample_mask1, sample_mask2],
    )

    for ts, n_scrub in zip(signals_list, [n_scrub1, n_scrub2]):
        assert ts.shape[0] == length - n_scrub

    # should also work with a single 4D image (has no __iter__ )
    signals_list = estimator.fit_transform(
        _img_4d_rand_eye_medium(),
        sample_mask=[sample_mask1],
    )

    assert signals_list.shape[0] == length - n_scrub1

    with pytest.raises(
        ValueError,
        match="number of sample_mask .* unequal to number of images",
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            sample_mask=[sample_mask1],
        )

    with pytest.raises(
        TypeError, match="'sample_mask' must be a None or a list."
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            sample_mask=1,
        )


def check_multi_masker_transformer_high_variance_confounds(estimator):
    """Check high_variance_confounds use in multi maskers with 5D data.

    Make sure that using high_variance_confounds returns different result.

    Ensure that high_variance_confounds can be used with regular confounds,
    and that results are different than when just using the confounds alone.
    """
    length = 20

    data = _rng().random((*_shape_3d_default(), length))
    input_img = Nifti1Image(data, _affine_eye())

    estimator.high_variance_confounds = False

    signal = estimator.fit_transform([input_img, input_img])

    estimator = clone(estimator)
    estimator.high_variance_confounds = True

    signal_hvc = estimator.fit_transform([input_img, input_img])

    for s1, s2 in zip(signal, signal_hvc):
        assert_raises(AssertionError, assert_array_equal, s1, s2)

    with TemporaryDirectory() as tmp_dir:
        array = _rng().random((length, 3))

        dataframe = pd.DataFrame(array)

        tmp_dir = Path(tmp_dir)
        dataframe.to_csv(tmp_dir / "confounds.csv")

        for c in [array, dataframe, tmp_dir / "confounds.csv"]:
            confounds = [c, c]

            estimator = clone(estimator)
            estimator.high_variance_confounds = False
            signal_c = estimator.fit_transform(
                [input_img, input_img], confounds=confounds
            )

            estimator = clone(estimator)
            estimator.high_variance_confounds = True
            signal_c_hvc = estimator.fit_transform(
                [input_img, input_img], confounds=confounds
            )

            for s1, s2 in zip(signal_c, signal_c_hvc):
                assert_raises(AssertionError, assert_array_equal, s1, s2)


# ------------------ GLM CHECKS ------------------


def check_glm_empty_data_messages(estimator: BaseEstimator) -> None:
    """Check that empty images are caught properly.

    Replaces sklearn check_estimators_empty_data_messages.

    Not implemented for nifti data for performance reasons.
    See : https://github.com/nilearn/nilearn/pull/5293#issuecomment-2977170723
    """
    imgs, design_matrices = _make_surface_img_and_design()

    data = {
        part: np.empty(0).reshape((imgs.data.parts[part].shape[0], 0))
        for part in imgs.data.parts
    }
    imgs = SurfaceImage(imgs.mesh, data)

    with pytest.raises(ValueError, match="empty"):
        # FirstLevel
        if hasattr(estimator, "hrf_model"):
            estimator.fit(imgs, design_matrices=design_matrices)
        # SecondLevel
        else:
            estimator.fit(imgs, design_matrix=design_matrices)


def check_glm_is_fitted(estimator):
    """Check glm throws proper error when not fitted."""
    with pytest.raises(ValueError, match=_not_fitted_error_message(estimator)):
        estimator.compute_contrast([])

    data, design_matrices = _make_surface_img_and_design()
    # FirstLevel
    if hasattr(estimator, "hrf_model"):
        estimator.fit(data, design_matrices=design_matrices)
    # SecondLevel
    else:
        estimator.fit(data, design_matrix=design_matrices)

    assert estimator.__sklearn_is_fitted__()

    check_is_fitted(estimator)


def check_glm_dtypes(estimator):
    """Check glm can fit with inputs of varying dtypes.

    Replacement for sklearn check_estimators_dtypes.

    np.int64 not tested: see no_int64_nifti in nilearn/conftest.py
    """
    imgs, design_matrices = _make_surface_img_and_design()

    for dtype in [np.float32, np.float64, np.int32]:
        estimator = clone(estimator)

        for k, v in imgs.data.parts.items():
            imgs.data.parts[k] = v.astype(dtype)

        # FirstLevel
        if hasattr(estimator, "hrf_model"):
            estimator.fit(imgs, design_matrices=design_matrices)
        # SecondLevel
        else:
            estimator.fit(imgs, design_matrix=design_matrices)


# ------------------ REPORT GENERATION CHECKS ------------------


def _generate_report_with_no_warning(estimator):
    """Check that report generation throws no warning."""
    with warnings.catch_warnings(record=True) as warning_list:
        report = _generate_report(estimator)

        # TODO
        # RegionExtractor, SurfaceMapsMasker still throws too many warnings
        warnings_to_ignore = [
            # only thrown with older dependencies
            "No contour levels were found within the data range.",
        ]
        unknown_warnings = [
            str(x.message)
            for x in warning_list
            if str(x.message) not in warnings_to_ignore
        ]
        if not isinstance(estimator, (RegionExtractor, SurfaceMapsMasker)):
            assert not unknown_warnings, unknown_warnings

    _check_html(report)

    return report


def _generate_report(estimator):
    """Adapt the call to generate_report to limit warnings.

    For example by only passing the number of displayed maps
    that a map masker contains.
    """
    if isinstance(
        estimator,
        (NiftiMapsMasker, MultiNiftiMapsMasker, SurfaceMapsMasker),
    ) and hasattr(estimator, "n_elements_"):
        return estimator.generate_report(displayed_maps=estimator.n_elements_)
    else:
        return estimator.generate_report()


def check_masker_generate_report(estimator):
    """Check that maskers can generate report.

    - check that we get a warning:
      - when matplotlib is not installed
      - when generating reports before fit
    - check content of report before fit and after fit

    """
    if not is_matplotlib_installed():
        with warnings.catch_warnings(record=True) as warning_list:
            report = _generate_report(estimator)

        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, ImportWarning)
        assert report == [None]

        return

    with warnings.catch_warnings(record=True) as warning_list:
        report = _generate_report(estimator)
        assert len(warning_list) == 1

    _check_html(report, is_fit=False)
    assert "Make sure to run `fit`" in str(report)

    if accept_niimg_input(estimator):
        input_img = _img_3d_rand()
    else:
        input_img = _make_surface_img(2)

    estimator.fit(input_img)

    assert estimator._report_content["warning_message"] is None

    # TODO
    # SurfaceMapsMasker, RegionExtractor still throws a warning
    report = _generate_report_with_no_warning(estimator)
    report = _generate_report(estimator)
    _check_html(report)

    with TemporaryDirectory() as tmp_dir:
        report.save_as_html(Path(tmp_dir) / "report.html")
        assert (Path(tmp_dir) / "report.html").is_file()


def check_nifti_masker_generate_report_after_fit_with_only_mask(estimator):
    """Check 3D mask is enough to run with fit and generate report."""
    mask = np.ones(_shape_3d_large())
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit()

    assert estimator._report_content["warning_message"] is None

    if not is_matplotlib_installed():
        return

    with pytest.warns(UserWarning, match="No image provided to fit."):
        report = _generate_report(estimator)
    _check_html(report)

    input_img = _img_4d_rand_eye_medium()

    estimator.fit(input_img)

    # TODO
    # NiftiSpheresMasker still throws a warning
    if isinstance(estimator, NiftiSpheresMasker):
        return
    report = _generate_report_with_no_warning(estimator)
    _check_html(report)


def check_masker_generate_report_false(estimator):
    """Test with reports set to False."""
    if not is_matplotlib_installed():
        return

    estimator.reports = False

    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(2)

    estimator.fit(input_img)

    assert estimator._reporting_data is None
    assert estimator._reporting() == [None]
    with pytest.warns(
        UserWarning,
        match=("No visual outputs created."),
    ):
        report = _generate_report(estimator)

    _check_html(report, reports_requested=False)

    assert "Empty Report" in str(report)


def check_multi_nifti_masker_generate_report_4d_fit(estimator):
    """Test calling generate report on multiple subjects raises warning."""
    if not is_matplotlib_installed():
        return

    estimator.maps_img = _img_3d_ones()
    estimator.fit([_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()])
    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        _generate_report(estimator)
