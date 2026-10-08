from sklearn.base import is_classifier, is_regressor

from nilearn._base import NilearnBaseEstimator
from nilearn._estimator_checks.nilearn_checks import (
    _clone_estimator,
    _requires_y,
    check_decoder_compatibility_mask_image,
    check_decoder_empty_data_messages,
    check_decoder_estimator_args,
    check_decoder_screening_n_features,
    check_decoder_with_arrays,
    check_decoder_with_surface_data,
    check_doc_attributes_after_fit,
    check_doc_link,
    check_doc_parameters_at_init,
    check_fit_returns_self,
    check_glm_empty_data_messages,
    check_img_estimator_cache_warning,
    check_img_estimator_clean_dtype,
    check_img_estimator_dict_unchanged,
    check_img_estimator_dont_overwrite_parameters,
    check_img_estimator_dtype_bool,
    check_img_estimator_dtypes,
    check_img_estimator_dtypes_inverse_transform,
    check_img_estimator_dtypes_transform,
    check_img_estimator_fit_check_is_fitted,
    check_img_estimator_fit_idempotent,
    check_img_estimator_fit_score_takes_y,
    check_img_estimator_n_elements,
    check_img_estimator_overwrite_params,
    check_img_estimator_pickle,
    check_img_estimator_pipeline_consistency,
    check_img_estimator_requires_y_none,
    check_img_estimator_standardization,
    check_img_estimator_verbose,
    check_img_regressor_no_decision_function,
    check_inputs_length,
    check_masker_clean,
    check_masker_clean_kwargs,
    check_masker_compatibility_mask_image,
    check_masker_detrending,
    check_masker_empty_data_messages,
    check_masker_fit_with_empty_mask,
    check_masker_fit_with_non_finite_in_mask,
    check_masker_generate_report,
    check_masker_generate_report_constant,
    check_masker_generate_report_false,
    check_masker_inverse_transform,
    check_masker_joblib_cache,
    check_masker_mask_img,
    check_masker_mask_img_from_imgs,
    check_masker_no_mask_no_img,
    check_masker_refit,
    check_masker_shelving,
    check_masker_smooth,
    check_masker_standardization,
    check_masker_transform_resampling,
    check_masker_transformer,
    check_masker_transformer_high_variance_confounds,
    check_masker_transformer_sample_mask,
    check_masker_verbose,
    check_masker_with_confounds,
    check_multi_nifti_masker_shelving,
    check_multimasker_generate_report,
    check_multimasker_transformer_high_variance_confounds,
    check_multimasker_transformer_sample_mask,
    check_multimasker_with_confounds,
    check_nifti_masker_dtype,
    check_nifti_masker_fit_transform,
    check_nifti_masker_fit_transform_5d,
    check_nifti_masker_fit_transform_files,
    check_nifti_masker_fit_with_3d_mask,
    check_nifti_masker_generate_report_after_fit_with_only_mask,
    check_nilearn_methods_sample_order_invariance,
    check_set_output,
    check_set_output_accepts_surface,
    check_supervised_img_estimator_y_no_nan,
    check_surface_masker_fit_transform_errors,
    check_surface_masker_list_surf_images_no_mask,
    check_surface_masker_list_surf_images_with_mask,
    check_verbose,
    check_verbosity_embedded_masker,
    check_warning_embedded_masker,
)
from nilearn.decomposition._base import _BaseDecomposition
from nilearn.maskers import NiftiMasker
from nilearn.maskers._mixin import _MultiMixin
from nilearn.utils.tags import (
    accepts_surface,
    accepts_volume,
    is_glm,
    is_masker,
)

# Checks that apply to all estimators
COMMON_CHECKS = [
    check_doc_parameters_at_init,
    check_doc_attributes_after_fit,
    check_doc_link,
    check_set_output,
    check_set_output_accepts_surface,
    check_verbose,
    check_img_estimator_cache_warning,
]

# Checks that apply to all estimators that accept volume or surface image as
# input
IMG_INPUT_COMMON_CHECKS = [
    check_fit_returns_self,
    check_img_estimator_dtypes,
    check_img_estimator_dtypes_transform,
    check_img_estimator_dtype_bool,
    check_img_estimator_dict_unchanged,
    check_img_estimator_dont_overwrite_parameters,
    check_img_estimator_fit_check_is_fitted,
    check_img_estimator_fit_idempotent,
    check_img_estimator_overwrite_params,
    check_img_estimator_pickle,
    check_img_estimator_fit_score_takes_y,
    check_img_estimator_n_elements,
    check_img_estimator_pipeline_consistency,
    check_img_estimator_standardization,
    check_img_estimator_verbose,
    check_nilearn_methods_sample_order_invariance,
    check_img_estimator_clean_dtype,
    check_img_estimator_dtypes_inverse_transform,
]

# Checks that apply to all classifiers and regressors which accept volume or
# surface image as input
IMG_INPUT_CLAS_REG_COMMON_CHECKS = [
    check_supervised_img_estimator_y_no_nan,
    check_decoder_empty_data_messages,
    check_decoder_compatibility_mask_image,
    check_decoder_screening_n_features,
    check_decoder_with_surface_data,
    check_decoder_with_arrays,
    check_decoder_estimator_args,
    check_verbosity_embedded_masker,
    check_warning_embedded_masker,
]

# Checks for regressors which accept volume or surface image as input
IMG_INPUT_REG_CHECKS = [
    *IMG_INPUT_CLAS_REG_COMMON_CHECKS,
    check_img_regressor_no_decision_function,
]

# Checks for estimators that accept volume or surface image as input and
# require y parameter for fit
IMG_INPUT_REQUIRES_Y = [
    check_img_estimator_requires_y_none,
    check_inputs_length,
]

# Checks that apply to all maskers
COMMON_MASKER_CHECKS = [
    check_masker_clean_kwargs,
    check_masker_compatibility_mask_image,
    check_masker_empty_data_messages,
    check_masker_fit_with_empty_mask,
    check_masker_fit_with_non_finite_in_mask,
    check_masker_generate_report,
    check_masker_generate_report_constant,
    check_masker_generate_report_false,
    check_masker_inverse_transform,
    check_masker_joblib_cache,
    check_masker_mask_img,
    check_masker_mask_img_from_imgs,
    check_masker_no_mask_no_img,
    check_masker_refit,
    check_masker_smooth,
    check_masker_standardization,
    check_masker_transform_resampling,
    check_masker_transformer,
    check_masker_transformer_high_variance_confounds,
    check_masker_verbose,
    check_masker_clean,
    check_masker_detrending,
]

# Checks that apply to all maskers which accept volume image as input
VOLUME_INPUT_MASKER_CHECKS = [
    *COMMON_MASKER_CHECKS,
    check_nifti_masker_dtype,
    check_nifti_masker_fit_transform,
    check_nifti_masker_fit_transform_5d,
    check_nifti_masker_fit_transform_files,
    check_nifti_masker_fit_with_3d_mask,
    check_nifti_masker_generate_report_after_fit_with_only_mask,
    check_masker_shelving,
]

MULTINIFTIMASKER_CHECKS = [check_multi_nifti_masker_shelving]

# Checks that apply to all maskers which accept surface image as input
SURFACE_INPUT_MASKER_CHECKS = [
    *COMMON_MASKER_CHECKS,
    check_surface_masker_fit_transform_errors,
    check_surface_masker_list_surf_images_no_mask,
    check_surface_masker_list_surf_images_with_mask,
]

# Checks specific to multi maskers
MULTI_MASKER_CHECKS = [
    check_multimasker_generate_report,
    check_multimasker_transformer_high_variance_confounds,
    check_multimasker_transformer_sample_mask,
    check_multimasker_with_confounds,
]

# Checks specific to non-multi maskers
NON_MULTI_MASKER_CHECKS = [
    check_masker_transformer_sample_mask,
    check_masker_with_confounds,
]

# Checks that apply to all GLM estimators
GLM_CHECKS = [
    check_glm_empty_data_messages,
    check_verbosity_embedded_masker,
    check_warning_embedded_masker,
]


# Checks that apply to all estimators inheriting from _BaseDecomposition
DECOMPOSITION_CHECKS = [
    check_verbosity_embedded_masker,
    check_warning_embedded_masker,
]


def accepts_image(estimator):
    """Check if estimator accepts volume of surface image."""
    return accepts_volume(estimator) or accepts_surface(estimator)


# List of tuples
# (conditions to test on estimator, list of checks to apply)
CHECK_SELECTOR = [
    (lambda e: True, COMMON_CHECKS),
    # ----------INPUT VOLUME OR SURFACE----------
    (lambda e: accepts_image(e), IMG_INPUT_COMMON_CHECKS),
    (
        lambda e: accepts_image(e) and is_classifier(e),
        IMG_INPUT_CLAS_REG_COMMON_CHECKS,
    ),
    (
        lambda e: accepts_image(e) and is_regressor(e),
        IMG_INPUT_REG_CHECKS,
    ),
    (
        lambda e: accepts_image(e) and _requires_y(e),
        IMG_INPUT_REQUIRES_Y,
    ),
    # ----------MASKERS----------
    (lambda e: is_masker(e) and accepts_volume(e), VOLUME_INPUT_MASKER_CHECKS),
    (
        lambda e: is_masker(e) and accepts_surface(e),
        SURFACE_INPUT_MASKER_CHECKS,
    ),
    (
        lambda e: is_masker(e) and isinstance(e, _MultiMixin),
        MULTI_MASKER_CHECKS,
    ),
    (
        lambda e: is_masker(e) and not isinstance(e, _MultiMixin),
        NON_MULTI_MASKER_CHECKS,
    ),
    # MultiNiftiMasker
    # TODO enforce for other maskers
    (
        lambda e: isinstance(e, NiftiMasker) and isinstance(e, _MultiMixin),
        MULTINIFTIMASKER_CHECKS,
    ),
    # ----------GLM----------
    # checks for glm estimators
    (lambda e: is_glm(e), GLM_CHECKS),
    # ----------DECOMPOSITION----------
    # checks for decomposition
    (lambda e: isinstance(e, _BaseDecomposition), DECOMPOSITION_CHECKS),
]


def nilearn_check_generator(estimator: NilearnBaseEstimator):
    """Yield check that applies to estimator.

    This will yield only the nilearn specific checks
    for a nilearn estimator.

    Each nilearn check can be run on an initialized estimator.
    """
    for condition, checks in CHECK_SELECTOR:
        if condition(estimator):
            yield from checks


def nilearn_check_estimator(estimators: list[NilearnBaseEstimator]):
    """Return a tuple in the form: (estimator, estimator_name, check_function)
    for each estimator in the ``estimators`` list.
    """
    checks_to_run = []
    for est in estimators:
        checks_to_run.extend(
            (_clone_estimator(est), est.__class__.__name__, check)
            for check in nilearn_check_generator(estimator=est)
        )

    return checks_to_run
