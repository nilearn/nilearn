"""Checks for nilearn estimators.

Most of those estimators have pytest dependencies
and importing them will fail if pytest is not installed.
"""

import inspect
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal, assert_raises
from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn import clone
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.exceptions import DimensionError, MeshDimensionError
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.niimg_conversions import check_imgs_equal
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
    _make_surface_img,
    _make_surface_img_and_design,
    _make_surface_mask,
    _rng,
    _shape_3d_default,
    _shape_3d_large,
)
from nilearn.maskers import (
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceMasker,
)
from nilearn.masking import load_mask_img
from nilearn.regions import RegionExtractor
from nilearn.reporting.tests.test_html_report import _check_html
from nilearn.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data
from nilearn.surface.utils import (
    assert_surface_image_equal,
)

SKLEARN_GT_1_5 = parse(sklearn_version).release[1] >= 6

# TODO simplify when dropping sklearn 1.5,
if SKLEARN_GT_1_5:
    from sklearn.utils.estimator_checks import _check_name
    from sklearn.utils.estimator_checks import (
        estimator_checks_generator as sklearn_check_generator,
    )
else:
    from sklearn.utils.estimator_checks import (
        check_estimator as sklearn_check_estimator,
    )

# List of sklearn estimators checks that are 'valid'
# for all nilearn estimators.
# Some may be explicitly skipped : see CHECKS_TO_SKIP_IF_IMG_INPUT below
VALID_CHECKS = [
    "check_complex_data",
    "check_decision_proba_consistency",
    "check_dict_unchangedcheck_clusterer_compute_labels_predict",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_dont_overwrite_parameters",
    "check_dtype_object",
    "check_estimator_cloneable",
    "check_estimators_dtypes",
    "check_estimator_repr",
    "check_estimator_sparse_array",
    "check_estimator_sparse_data",
    "check_estimator_sparse_matrix",
    "check_estimator_sparse_tag",
    "check_estimator_tags_renamed",
    "check_estimators_empty_data_messages",
    "check_estimators_overwrite_params",
    "check_estimators_partial_fit_n_features",
    "check_estimators_unfitted",
    "check_f_contiguous_array_estimator",
    "check_fit1d",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit2d_predict1d",
    "check_fit_check_is_fitted",
    "check_fit_score_takes_y",
    "check_get_params_invariance",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_mixin_order",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_no_attributes_set_in_init",
    "check_non_transformer_estimators_n_iter",
    "check_parameters_default_constructible",
    "check_positive_only_tag_during_fit",
    "check_readonly_memmap_input",
    "check_set_params",
    "check_transformer_data_not_an_array",
    "check_transformer_general",
    "check_transformer_n_iter",
    "check_transformer_preserve_dtypes",
    "check_transformers_unfitted",
]

if SKLEARN_GT_1_5:
    VALID_CHECKS.append("check_valid_tag_types")
else:
    VALID_CHECKS.append("check_estimator_get_tags_default_keys")

# keeping track of some of those in
# https://github.com/nilearn/nilearn/issues/4538


CHECKS_TO_SKIP_IF_IMG_INPUT = {
    # The following do not apply for nilearn maskers
    # as they do not take numpy arrays as input.
    "check_complex_data": "not applicable for image input",
    "check_dtype_object": "not applicable for image input",
    "check_estimator_sparse_array": "not applicable for image input",
    "check_estimator_sparse_data": "not applicable for image input",
    "check_estimator_sparse_matrix": "not applicable for image input",
    "check_estimator_sparse_tag": "not applicable for image input",
    "check_f_contiguous_array_estimator": "not applicable for image input",
    "check_fit1d": "not applicable for image input",
    "check_fit2d_1feature": "not applicable for image input",
    "check_fit2d_1sample": "not applicable for image input",
    "check_fit2d_predict1d": "not applicable for image input",
    "check_n_features_in": "not applicable",
    "check_n_features_in_after_fitting": "not applicable",
    # the following are skipped because there is nilearn specific replacement
    "check_estimators_dtypes": (
        "replaced by check_masker_dtypes andcheck_glm_dtypes"
    ),
    "check_estimators_fit_returns_self": (
        "replaced by check_nifti_masker_fit_returns_self "
        "or check_surface_masker_fit_returns_self or "
        "check_glm_fit_returns_self"
    ),
    "check_fit_check_is_fitted": (
        "replaced by check_masker_fitted or check_glm_is_fitted"
    ),
    "check_transformer_data_not_an_array": (
        "replaced by check_masker_transformer"
    ),
    "check_transformer_general": ("replaced by check_masker_transformer"),
    "check_transformer_preserve_dtypes": (
        "replaced by check_masker_transformer"
    ),
    "check_dict_unchanged": "check_masker_dict_unchanged",
    "check_fit_score_takes_y": {"replaced by check_masker_fit_score_takes_y"},
    # Those are skipped for now they fail
    # for unknown reasons
    #  most often because sklearn inputs expect a numpy array
    #  that errors with maskers,
    # or because a suitable nilearn replacement has not yet been created.
    "check_dont_overwrite_parameters": "TODO",
    "check_estimators_empty_data_messages": "TODO",
    "check_estimators_nan_inf": "TODO",
    "check_estimators_overwrite_params": "TODO",
    "check_estimators_pickle": "TODO",
    "check_fit_idempotent": "TODO",
    "check_methods_sample_order_invariance": "TODO",
    "check_methods_subset_invariance": "TODO",
    "check_positive_only_tag_during_fit": "TODO",
    "check_pipeline_consistency": "TODO",
    "check_readonly_memmap_input": "TODO",
}

# TODO
# remove when bumping to sklearn >= 1.3
try:
    from sklearn.utils.estimator_checks import (
        check_classifiers_one_label_sample_weights,
    )

    VALID_CHECKS.append(check_classifiers_one_label_sample_weights.__name__)
except ImportError:
    ...


def check_estimator(
    estimator=None,
    valid: bool = True,
    expected_failed_checks=None,
):
    """Yield a valid or invalid scikit-learn estimators check.

    As some of Nilearn estimators do not comply
    with sklearn recommendations
    (cannot fit Numpy arrays, do input validation in the constructor...)
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made generator that yields an estimator instance
    along with a
    - valid check from sklearn: those should stay valid
    - or an invalid check that is known to fail.

    If estimator have some nilearn specific tags
    then some checks will skip rather than yield.

    See this section rolling-your-own-estimator in
    the scikit-learn doc for more info:
    https://scikit-learn.org/stable/developers/develop.html

    Parameters
    ----------
    estimator : estimator object or list of estimator object
        Estimator instance to check.

    valid : bool, default=True
        Whether to return only the valid checks or not.

    expected_failed_checks: dict or None, default=None
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.
    """
    valid_checks = VALID_CHECKS

    if not isinstance(estimator, list):
        estimator = [estimator]

    for est in estimator:
        # TODO simplify when dropping sklearn 1.5
        if SKLEARN_GT_1_5:
            tags = est.__sklearn_tags__()

            niimg_input = getattr(tags.input_tags, "niimg_like", False)
            surf_img = getattr(tags.input_tags, "surf_img", False)

            if niimg_input or surf_img:
                if expected_failed_checks is None:
                    expected_failed_checks = CHECKS_TO_SKIP_IF_IMG_INPUT
                else:
                    expected_failed_checks |= CHECKS_TO_SKIP_IF_IMG_INPUT

            for e, check in sklearn_check_generator(
                estimator=est,
                expected_failed_checks=expected_failed_checks,
                # TODO use  mark="xfail"
                # once using only expected_failed_checks and no valid_checks
                mark="skip",
            ):
                # DANGER
                # must rely on sklearn private function _check_name
                # to get name of the check:
                # things may break with no deprecation warning
                name = _check_name(check)

                if valid and name in valid_checks:
                    yield e, check, name
                if not valid and name not in valid_checks:
                    yield e, check, name

        else:
            for e, check in sklearn_check_estimator(
                estimator=est, generate_only=True
            ):
                tags = est._more_tags()

                niimg_input = "niimg_like" in tags["X_types"]
                surf_img = "surf_img" in tags["X_types"]

                if niimg_input or surf_img:
                    if expected_failed_checks is None:
                        expected_failed_checks = CHECKS_TO_SKIP_IF_IMG_INPUT
                    else:
                        expected_failed_checks |= CHECKS_TO_SKIP_IF_IMG_INPUT

                if (
                    isinstance(expected_failed_checks, dict)
                    and check.func.__name__ in expected_failed_checks
                ):
                    continue

                if valid and check.func.__name__ in valid_checks:
                    yield e, check, check.func.__name__
                if not valid and check.func.__name__ not in valid_checks:
                    yield e, check, check.func.__name__

    if valid:
        for est in estimator:
            for e, check in nilearn_check_estimator(estimator=est):
                yield e, check, check.__name__


def nilearn_check_estimator(estimator):
    is_masker = False
    is_glm = False
    surf_img_input = False

    if SKLEARN_GT_1_5:
        tags = estimator.__sklearn_tags__()
    else:  # pragma: no cover
        tags = estimator._more_tags()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        is_masker = "masker" in tags["X_types"]
        is_glm = "glm" in tags["X_types"]
        surf_img_input = "surf_img" in tags["X_types"]
    else:
        is_masker = getattr(tags.input_tags, "masker", False)
        is_glm = getattr(tags.input_tags, "glm", False)
        surf_img_input = getattr(tags.input_tags, "surf_img", False)

    yield (clone(estimator), check_estimator_has_sklearn_is_fitted)

    if is_masker:
        yield (clone(estimator), check_masker_fitted)
        yield (clone(estimator), check_masker_clean_kwargs)
        yield (clone(estimator), check_masker_generate_report)
        yield (clone(estimator), check_masker_generate_report_false)
        yield (clone(estimator), check_masker_refit)

        yield (clone(estimator), check_masker_fit_score_takes_y)

        yield (clone(estimator), check_masker_transformer)
        yield (clone(estimator), check_masker_inverse_transform)

        yield (clone(estimator), check_masker_compatibility_mask_image)

        yield (clone(estimator), check_masker_dict_unchanged)

        yield (clone(estimator), check_masker_fit_with_empty_mask)

        yield (clone(estimator), check_masker_fit_returns_self)

        yield (
            clone(estimator),
            check_masker_fit_with_non_finite_in_mask,
        )
        yield (clone(estimator), check_masker_mask_img)
        yield (clone(estimator), check_masker_no_mask_no_img)
        yield (clone(estimator), check_masker_mask_img_from_imgs)

        yield (clone(estimator), check_masker_smooth)

        if not is_multimasker(estimator):
            yield (clone(estimator), check_masker_detrending)
            yield (clone(estimator), check_masker_clean)
            yield (clone(estimator), check_masker_transformer_sample_mask)
            yield (clone(estimator), check_masker_with_confounds)

            # TODO this should pass for multimasker
            yield (
                clone(estimator),
                check_masker_transformer_high_variance_confounds,
            )

        if accept_niimg_input(estimator):
            yield (clone(estimator), check_nifti_masker_fit_transform)
            yield (clone(estimator), check_nifti_masker_fit_transform_files)
            yield (clone(estimator), check_nifti_masker_fit_with_3d_mask)
            yield (
                clone(estimator),
                check_nifti_masker_generate_report_after_fit_with_only_mask,
            )
            yield (clone(estimator), check_nifti_masker_clean_error)
            yield (clone(estimator), check_nifti_masker_clean_warning)
            yield (clone(estimator), check_nifti_masker_dtype)
            yield (clone(estimator), check_nifti_masker_fit_transform_5d)
            yield (clone(estimator), check_masker_dtypes)

            if is_multimasker(estimator):
                yield (clone(estimator), check_multi_masker_with_confounds)
                yield (
                    clone(estimator),
                    check_multi_masker_transformer_sample_mask,
                )
                yield (
                    clone(estimator),
                    check_multi_nifti_masker_generate_report_4d_fit,
                )

        if surf_img_input:
            yield (clone(estimator), check_surface_masker_fit_with_mask)

    if is_glm:
        yield (clone(estimator), check_glm_fit_returns_self)
        yield (clone(estimator), check_glm_is_fitted)
        yield (clone(estimator), check_glm_dtypes)


def is_multimasker(estimator):
    tags = estimator.__sklearn_tags__()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return "multi_masker" in tags["X_types"]
    else:
        return getattr(tags.input_tags, "multi_masker", False)


def accept_niimg_input(estimator):
    tags = estimator.__sklearn_tags__()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return "niimg_like" in tags["X_types"]
    else:
        return getattr(tags.input_tags, "niimg_like", False)


def _not_fitted_error_message(estimator):
    return (
        f"This {type(estimator).__name__} instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator."
    )


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

    # TODO NiftiLabelsMasker, NiftiMapsMasker are modified at transform time
    # see issue https://github.com/nilearn/nilearn/issues/2720
    if isinstance(estimator, (NiftiLabelsMasker, NiftiMapsMasker)):
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


def check_masker_fit_returns_self(estimator):
    """Check maskers return itself after fit."""
    if accept_niimg_input(estimator):
        imgs = _img_3d_rand()
    else:
        imgs = _make_surface_img(10)

    assert estimator.fit(imgs) is estimator


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
    """
    estimator.high_variance_confounds = False

    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(100)

    signal_1 = estimator.fit_transform(input_img)

    estimator = clone(estimator)
    estimator.high_variance_confounds = True
    signal_2 = estimator.fit_transform(input_img)

    assert_raises(AssertionError, assert_array_equal, signal_1, signal_2)


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

    nilearn_dir = Path(__file__).parents[1]
    confounds_path = nilearn_dir / "tests" / "data" / "spm_confounds.txt"

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

    n_sample = 1
    if accept_niimg_input(estimator):
        imgs = _img_3d_rand()
    else:
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


def check_masker_inverse_transform(estimator):
    """Check output of inverse transform.

    For signal with 1 or more samples.

    For nifti maskers:
        - 1D arrays give 3D images
        - 2D arrays give 4D images

    Check that the proper error is thrown,
    if signal has the wrong shape.
    """
    n_sample = 1
    if accept_niimg_input(estimator):
        imgs = _img_3d_rand()
        mask_img = _img_3d_ones()
        input_shape = imgs.get_fdata().shape
    else:
        imgs = _make_surface_img(n_sample)
        mask_img = _make_surface_mask()
        input_shape = imgs.shape

    if isinstance(estimator, NiftiSpheresMasker):
        estimator.mask_img = mask_img

    estimator.fit(imgs)

    signals_1d = _rng().random((estimator.n_elements_,))
    signals_2d = _rng().random((1, estimator.n_elements_))
    signals_2d_multisample = _rng().random((10, estimator.n_elements_))

    for signal, expected_shape in zip(
        [signals_1d, signals_2d, signals_2d_multisample],
        [input_shape, (*input_shape, 1), (*input_shape, 10)],
    ):
        new_imgs = estimator.inverse_transform(signal)

        if accept_niimg_input(estimator):
            assert new_imgs.get_fdata().shape == expected_shape
        else:
            # TODO for surface maskers
            # assert new_imgs.data.shape == expected_shape
            ...

    signals = _rng().random((1, estimator.n_elements_ + 1))

    with pytest.raises(
        ValueError, match="Input to 'inverse_transform' has wrong shape."
    ):
        estimator.inverse_transform(signals)


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

    Also check 'shape' errors between images to fit and mask.
    """
    mask_img = _make_surface_mask()

    # 1D image
    imgs = _make_surface_img(1)
    estimator.mask_img = mask_img
    estimator.fit(imgs)

    signal = estimator.transform(imgs)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (1, estimator.n_elements_)

    # 2D image
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


# ------------------ NIFTI MASKER CHECKS ------------------


def check_nifti_masker_fit_transform(estimator):
    """Run several checks on maskers.

    - can fit 3D image
    - fitted maskers can transform:
      - 3D image
      - list of 3D images with same affine
    - can fit transform 3D image
    - array from transformed 3D images should have 1D
    - array from transformed 4D images should have 2D
    """
    estimator.fit(_img_3d_rand())

    # 3D images
    signal = estimator.transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (estimator.n_elements_,)

    signal = estimator.fit_transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (estimator.n_elements_,)

    # 4D images
    signal = estimator.transform([_img_3d_rand(), _img_3d_rand()])

    if is_multimasker(estimator):
        signal = signal[0]
    assert isinstance(signal, np.ndarray)
    if is_multimasker(estimator):
        assert signal.ndim == 1
        assert signal.shape == (estimator.n_elements_,)
    else:
        assert signal.ndim == 2
        assert signal.shape[1] == estimator.n_elements_

    signal = estimator.transform(_img_4d_rand_eye())

    if is_multimasker(estimator):
        signal = signal[0]
    assert isinstance(signal, np.ndarray)
    if is_multimasker(estimator):
        assert signal.ndim == 1
        assert signal.shape == (estimator.n_elements_,)
    else:
        assert signal.ndim == 2
        assert signal.shape[1] == estimator.n_elements_


def check_nifti_masker_fit_transform_5d(estimator):
    """Run checks on nifti maskers for transforming 5D images.

    - multi masker should be fine
      and return a list of numpy arrays
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

    # TODO remove this cloning once nifti sphere masker can be refitted
    # See https://github.com/nilearn/nilearn/issues/5091
    estimator = clone(estimator)

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

    Check that error is raised if number of confounds
    does not match number of images
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

    with pytest.raises(
        ValueError, match="number of confounds .* unequal to number of images"
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            confounds=array,
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

    with pytest.raises(
        ValueError,
        match="number of sample_mask .* unequal to number of images",
    ):
        estimator.fit_transform(
            [_img_4d_rand_eye_medium(), _img_4d_rand_eye_medium()],
            sample_mask=sample_mask1,
        )


# ------------------ GLM CHECKS ------------------


def check_glm_fit_returns_self(estimator):
    """Check surface maskers return itself after fit."""
    data, design_matrices = _make_surface_img_and_design()
    # FirstLevel
    if hasattr(estimator, "hrf_model"):
        assert (
            estimator.fit(data, design_matrices=design_matrices) is estimator
        )
    # SecondLevel
    else:
        assert estimator.fit(data, design_matrix=design_matrices) is estimator


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
    from nilearn.maskers import (
        SurfaceMapsMasker,
    )

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
    from nilearn.maskers import (
        MultiNiftiMapsMasker,
        NiftiMapsMasker,
        SurfaceMapsMasker,
    )

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
