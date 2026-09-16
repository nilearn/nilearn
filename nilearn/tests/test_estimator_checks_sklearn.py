import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._base import NilearnBaseEstimator
from nilearn._utils.tags import (
    accept_niimg_input,
    accept_surf_img_input,
    is_glm,
    is_masker,
)
from nilearn.connectome import GroupSparseCovariance, GroupSparseCovarianceCV
from nilearn.connectome.connectivity_matrices import ConnectivityMeasure
from nilearn.decoding.decoder import _BaseDecoder
from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.space_net import BaseSpaceNet
from nilearn.decomposition._base import _BaseDecomposition
from nilearn.regions import HierarchicalKMeans, ReNA
from nilearn.utils.discovery import all_estimators


def return_expected_failed_checks(
    estimator: NilearnBaseEstimator,
) -> dict[str, str]:
    """Return the expected failures for a given estimator.

    This will say which of the sklearn checks are expected to fail
    for a given nilearn estimator,
    with the reason why or saying what home made check replaces it.

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
            "check_fit2d_predict1d": "not applicable",
            "check_estimator_sparse_array": "TODO",
            "check_estimator_sparse_matrix": "TODO",
            "check_methods_sample_order_invariance": "TODO",
            "check_methods_subset_invariance": "TODO",
            "check_readonly_memmap_input": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

        return expected_failed_checks

    elif isinstance(estimator, (HierarchicalKMeans, ReNA)):
        return expected_failed_checks_clustering()

    elif isinstance(
        estimator, (GroupSparseCovariance, GroupSparseCovarianceCV)
    ):
        expected_failed_checks = {
            "check_estimator_sparse_data": "removed when dropping sklearn 1.4",
            "check_estimator_sparse_array": "TODO",
            "check_estimator_sparse_matrix": "TODO",
            "check_estimator_sparse_tag": "TODO",
        }
        if isinstance(estimator, GroupSparseCovarianceCV):
            expected_failed_checks |= {
                "check_estimators_dtypes": "TODO",
                "check_dtype_object": "TODO",
            }
        return expected_failed_checks

    # below this point we should only deal with estimators
    # that accept images as input
    assert accept_niimg_input(estimator) or accept_surf_img_input(estimator)

    if isinstance(estimator, (_BaseDecoder, SearchLight, BaseSpaceNet)):
        return expected_failed_checks_decoders(estimator)

    # keeping track of some of those in
    # https://github.com/nilearn/nilearn/issues/4538
    expected_failed_checks = {
        # the following are skipped
        # because there is nilearn specific replacement
        "check_dict_unchanged": (
            "replaced by check_img_estimator_dict_unchanged"
        ),
        "check_dont_overwrite_parameters": (
            "replaced by check_img_estimator_dont_overwrite_parameters"
        ),
        "check_estimators_dtypes": "replaced by check_img_estimator_dtypes",
        "check_estimators_empty_data_messages": (
            "replaced by check_*_empty_data_messages "
        ),
        "check_estimators_fit_returns_self": (
            "replaced by check_fit_returns_self"
        ),
        "check_estimators_overwrite_params": (
            "replaced by check_img_estimator_overwrite_params"
        ),
        "check_estimators_pickle": "replaced by check_img_estimator_pickle",
        "check_fit_check_is_fitted": (
            "replaced by check_img_estimator_fit_check_is_fitted"
        ),
        "check_fit_idempotent": (
            "replaced by check_img_estimator_fit_idempotent"
        ),
        "check_fit_score_takes_y": (
            "replaced by check_img_estimator_fit_score_takes_y"
        ),
        "check_methods_sample_order_invariance": (
            "replaced by check_nilearn_methods_sample_order_invariance"
        ),
        "check_n_features_in": "replaced by check_img_estimator_n_elements",
        "check_n_features_in_after_fitting": (
            "replaced by check_img_estimator_n_elements"
        ),
        "check_pipeline_consistency": (
            "replaced by check_img_estimator_pipeline_consistency"
        ),
        # Those are skipped for now they fail
        # for unknown reasons
        # most often because sklearn inputs expect a numpy array
        # that errors with maskers,
        # or because a suitable nilearn replacement
        # has not yet been created.
        "check_estimators_nan_inf": "TODO",
        "check_methods_subset_invariance": "TODO",
        "check_positive_only_tag_during_fit": "TODO",
        "check_readonly_memmap_input": "TODO",
    }

    expected_failed_checks |= inapplicable_checks()

    if hasattr(estimator, "transform"):
        expected_failed_checks |= {
            "check_transformer_data_not_an_array": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_general": (
                "replaced by check_masker_transformer"
            ),
            "check_transformer_preserve_dtypes": (
                "replaced by check_img_estimator_dtypes"
            ),
        }

    # Adapt some checks for some estimators

    # not entirely sure why some of them pass
    # e.g check_estimator_sparse_data passes for SurfaceLabelsMasker
    # but not SurfaceMasker ????

    if is_glm(estimator):
        expected_failed_checks.pop("check_estimator_sparse_data")
        expected_failed_checks.pop("check_estimator_sparse_matrix")
        expected_failed_checks.pop("check_estimator_sparse_array")
        expected_failed_checks.pop("check_estimator_sparse_tag")

        expected_failed_checks |= {
            # have nilearn replacements
            "check_estimators_dtypes": (
                "replaced by check_img_estimator_dtypes"
            ),
            "check_dict_unchanged": "does not apply - no transform method",
            "check_methods_sample_order_invariance": (
                "does not apply - no relevant method"
            ),
            "check_estimators_empty_data_messages": (
                "not implemented for nifti data for performance reasons"
            ),
            "check_estimators_fit_returns_self": (
                "replaced by check_glm_fit_returns_self"
            ),
            "check_fit_check_is_fitted": (
                "replaced by check_img_estimator_fit_check_is_fitted"
            ),
        }

    if isinstance(estimator, (_BaseDecomposition,)):
        expected_failed_checks |= {
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }
        expected_failed_checks.pop("check_estimator_sparse_tag")

    if is_masker(estimator):
        expected_failed_checks |= {
            "check_n_features_in": (
                "replaced by check_img_estimator_n_elements"
            ),
            "check_n_features_in_after_fitting": (
                "replaced by check_img_estimator_n_elements"
            ),
        }

    return expected_failed_checks


def inapplicable_checks() -> dict[str, str]:
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
        ],
        "not applicable for image input",
    )


def expected_failed_checks_clustering() -> dict[str, str]:
    """Return expected failed checks for clustering."""
    expected_failed_checks = {
        "check_clustering": "TODO",
    }
    return expected_failed_checks


def expected_failed_checks_decoders(estimator) -> dict[str, str]:
    """Return expected failed sklearn checks for nilearn decoders."""
    expected_failed_checks = {
        # the following are have nilearn replacement for masker and/or glm
        # but not for decoders
        "check_dict_unchanged": (
            "replaced by check_img_estimator_dict_unchanged"
        ),
        "check_dont_overwrite_parameters": (
            "replaced by check_img_estimator_dont_overwrite_parameters"
        ),
        "check_estimators_empty_data_messages": (
            "replaced by check_*_empty_data_messages "
        ),
        "check_estimators_fit_returns_self": (
            "replaced by check_fit_returns_self"
        ),
        "check_estimators_overwrite_params": (
            "replaced by check_img_estimator_overwrite_params"
        ),
        "check_estimators_pickle": "replaced by check_img_estimator_pickle",
        "check_fit_check_is_fitted": (
            "replaced by check_img_estimator_fit_check_is_fitted"
        ),
        "check_fit_idempotent": (
            "replaced by check_img_estimator_fit_idempotent"
        ),
        "check_fit_score_takes_y": (
            "replaced by check_img_estimator_fit_score_takes_y"
        ),
        "check_methods_sample_order_invariance": (
            "replaced by check_nilearn_methods_sample_order_invariance"
        ),
        "check_n_features_in": "replaced by check_img_estimator_n_elements",
        "check_n_features_in_after_fitting": (
            "replaced by check_img_estimator_n_elements"
        ),
        "check_pipeline_consistency": (
            "replaced by check_img_estimator_pipeline_consistency"
        ),
        "check_requires_y_none": (
            "replaced by check_img_estimator_requires_y_none"
        ),
        "check_supervised_y_no_nan": (
            "replaced by check_supervised_img_estimator_y_no_nan"
        ),
        # Those are skipped for now they fail
        # for unknown reasons
        # most often because sklearn inputs expect a numpy array
        # that errors with maskers,
        # or because a suitable nilearn replacement
        # has not yet been created.
        "check_estimators_dtypes": "replaced by check_img_estimator_dtypes",
        "check_estimators_nan_inf": "TODO",
        "check_methods_subset_invariance": "TODO",
        "check_positive_only_tag_during_fit": "TODO",
        "check_readonly_memmap_input": "TODO",
        "check_supervised_y_2d": "TODO",
    }

    if isinstance(estimator, BaseSpaceNet):
        expected_failed_checks |= {
            "check_non_transformer_estimators_n_iter": ("TODO")
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
        if isinstance(estimator, BaseSpaceNet):
            expected_failed_checks |= {
                "check_classifier_multioutput": ("TODO")
            }

    if is_regressor(estimator):
        expected_failed_checks |= {
            "check_regressors_no_decision_function": (
                "replaced by check_img_regressor_no_decision_function"
            ),
            "check_regressor_data_not_an_array": (
                "not applicable for image input"
            ),
            "check_regressor_multioutput": "TODO",
            "check_regressors_int": "TODO",
            "check_regressors_train": "TODO",
        }

    if hasattr(estimator, "transform"):
        expected_failed_checks |= {
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": (
                "replaced by check_img_estimator_dtypes"
            ),
        }

    expected_failed_checks |= inapplicable_checks()

    return expected_failed_checks


@pytest.mark.slow
@parametrize_with_checks(
    estimators=[est() for _, est in all_estimators()],
    expected_failed_checks=return_expected_failed_checks,
)
def test_check_estimator_sklearn(estimator, check):
    """Check compliance with sklearn estimators."""
    check(estimator)
