"""Small utilities to inspect classes."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal, assert_raises
from sklearn import __version__ as sklearn_version
from sklearn import clone
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils import compare_version
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.helpers import is_matplotlib_installed

# List of sklearn estimators checks that are valid
# for all nilearn estimators.
VALID_CHECKS = [
    "check_decision_proba_consistency",
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_estimator_tags_renamed",
    "check_estimators_partial_fit_n_features",
    "check_estimators_unfitted",
    "check_fit_check_is_fitted",
    "check_get_params_invariance",
    "check_mixin_order",
    "check_non_transformer_estimators_n_iter",
    "check_parameters_default_constructible",
    "check_set_params",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]

if compare_version(sklearn_version, ">", "1.5.2"):
    VALID_CHECKS.append("check_valid_tag_types")
else:
    VALID_CHECKS.append("check_estimator_get_tags_default_keys")


# TODO / TOCHECK: with sklearn >= 1.6
# some of those checks should be skipped 'automatically'
# by sklearn
# and could be removed from this list.
CHECKS_TO_SKIP_IF_IMG_INPUT = {
    "check_complex_data",
    "check_dtype_object",
    "check_dict_unchanged",
    "check_dont_overwrite_parameters",
    "check_estimator_sparse_array",
    "check_estimator_sparse_data",
    "check_estimator_sparse_matrix",
    "check_estimator_sparse_tag",
    "check_estimators_empty_data_messages",
    "check_estimators_dtypes",
    "check_estimators_nan_inf",
    "check_estimators_overwrite_params",
    "check_estimators_pickle",
    "check_estimators_fit_returns_self",
    "check_f_contiguous_array_estimator",
    "check_fit_check_is_fitted",
    "check_fit1d",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit2d_predict1d",
    "check_fit_score_takes_y",
    "check_fit_idempotent",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_positive_only_tag_during_fit",
    "check_pipeline_consistency",
    "check_readonly_memmap_input",
    "check_transformer_data_not_an_array",
    "check_transformer_general",
    "check_transformer_preserve_dtypes",
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
    valid=True,
    extra_valid_checks=None,
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

    If new 'valid' checks are added to scikit-learn,
    then tests marked as xfail will start passing.

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

    extra_valid_checks : list of strings
        Names of checks to be tested as valid for this estimator.

    expected_failed_checks: dict, default=None
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.
    """
    valid_checks = VALID_CHECKS
    if extra_valid_checks is not None:
        valid_checks = VALID_CHECKS + extra_valid_checks

    if not isinstance(estimator, list):
        estimator = [estimator]
    for est in estimator:
        for e, check in sklearn_check_estimator(
            estimator=est, generate_only=True
        ):
            # TODO when dropping sklearn 1.5,
            # pass expected_failed_checks directly to
            # sklearn_check_estimator above
            if (
                isinstance(expected_failed_checks, dict)
                and check.func.__name__ in expected_failed_checks
            ):
                continue

            tags = est._more_tags()

            niimg_input = False
            surf_img = False
            # TODO remove first if when dropping sklearn 1.5
            #  for sklearn >= 1.6 tags are always a dataclass
            if isinstance(tags, dict) and "X_types" in tags:
                niimg_input = "niimg_like" in tags["X_types"]
                surf_img = "surf_img" in tags["X_types"]
            else:
                niimg_input = getattr(tags.input_tags, "niimg_like", False)
                surf_img = getattr(tags.input_tags, "surf_img", False)

            if (
                niimg_input or surf_img
            ) and check.func.__name__ in CHECKS_TO_SKIP_IF_IMG_INPUT:
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
    tags = estimator._more_tags()

    is_masker = False
    surf_img_input = False
    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        is_masker = "masker" in tags["X_types"]
        surf_img_input = "surf_img" in tags["X_types"]
    else:
        is_masker = getattr(tags.input_tags, "masker", False)
        surf_img_input = getattr(tags.input_tags, "surf_img", False)

    yield (clone(estimator), check_estimator_has_sklearn_is_fitted)

    if is_masker:
        yield (clone(estimator), check_masker_fitted)
        yield (clone(estimator), check_masker_clean_kwargs)
        yield (clone(estimator), check_masker_generate_report)
        yield (clone(estimator), check_masker_generate_report_false)

        if not is_multimasker(estimator):
            yield (clone(estimator), check_masker_detrending)
            yield (clone(estimator), check_masker_clean)

        if accept_niimg_input(estimator):
            yield (clone(estimator), check_nifti_masker_fit_transform)
            yield (clone(estimator), check_nifti_masker_fit_transform_files)
            yield (clone(estimator), check_nifti_masker_fit_with_3d_mask)
            yield (clone(estimator), check_nifti_masker_fit_with_4d_mask)
            yield (clone(estimator), check_nifti_masker_fit_with_empty_mask)
            yield (clone(estimator), check_nifti_masker_fit_with_only_mask)
            yield (
                clone(estimator),
                check_nifti_masker_generate_report_after_fit_with_only_mask,
            )
            yield (clone(estimator), check_nifti_masker_clean_error)
            yield (clone(estimator), check_nifti_masker_clean_warning)
            yield (clone(estimator), check_nifti_masker_dtype)
            yield (clone(estimator), check_nifti_masker_smooth)
            yield (clone(estimator), check_nifti_masker_fit_returns_self)
            yield (clone(estimator), check_nifti_masker_fit_transform_5d)

        if surf_img_input:
            yield (clone(estimator), check_surface_masker_fit_returns_self)
            yield (clone(estimator), check_surface_masker_smooth)


def is_multimasker(estimator):
    tags = estimator._more_tags()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return "multi_masker" in tags["X_types"]
    else:
        return getattr(tags.input_tags, "multi_masker", False)


def is_mapsmasker(estimator):
    tags = estimator._more_tags()

    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return "maps_masker" in tags["X_types"]
    else:
        return getattr(tags.input_tags, "maps_masker", False)


def accept_niimg_input(estimator):
    tags = estimator._more_tags()

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


def check_estimator_has_sklearn_is_fitted(estimator):
    """Check appropriate response to check_fitted from sklearn before fitting.

    check that before fitting
    - estimator has a __sklearn_is_fitted__ method
    - running sklearn check_is_fitted on estimator throws an error
    """
    import pytest

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
    """
    import pytest

    from nilearn.conftest import _img_3d_rand, _make_surface_img

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


def check_nifti_masker_fit_returns_self(estimator):
    """Check if self is returned when calling fit."""
    from nilearn.conftest import _img_3d_rand

    assert estimator.fit(_img_3d_rand()) is estimator


def check_surface_masker_fit_returns_self(estimator):
    """Check detrending does something.

    Fit transform on same input should give different results
    if detrend is true or false.
    """
    from nilearn.conftest import _make_surface_img

    assert estimator.fit(_make_surface_img(10)) is estimator


def check_nifti_masker_fit_transform(estimator):
    """Run several checks on maskers.

    - can fit 3D image
    - fitted maskers can transform:
      - 3D image
      - list of 3D images with same affine
    - can fit transform 3D image
    """
    from nilearn.conftest import _img_3d_rand, _img_4d_rand_eye

    estimator.fit(_img_3d_rand())

    signal = estimator.transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1

    estimator.transform([_img_3d_rand(), _img_3d_rand()])

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1

    estimator.transform(_img_4d_rand_eye())

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1

    estimator.fit_transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1


def check_nifti_masker_fit_transform_5d(estimator):
    """Run checks on nifti maskers for transforming 5D images.

    - multi masker should be fine
      and return a list of numpy arrays
    - non multimasker should fail
    """
    import pytest

    from nilearn.conftest import _img_3d_rand, _img_4d_rand_eye

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

        signal = estimator.fit_transform(input_5d_img)

        assert isinstance(signal, list)
        assert all(isinstance(x, np.ndarray) for x in signal)
        assert len(signal) == n_subject


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
    from nilearn.conftest import _img_4d_rand_eye_medium, _make_surface_img

    if accept_niimg_input(estimator):
        input_img = _img_4d_rand_eye_medium()
    else:
        input_img = _make_surface_img(100)

    signal = estimator.fit_transform(input_img)

    estimator.detrend = True
    detrended_signal = estimator.fit_transform(input_img)

    assert_raises(AssertionError, assert_array_equal, detrended_signal, signal)


def check_masker_clean(estimator):
    """Check that cleaning does something on fit transform.

    Fit transform on same input should give different results
    if some cleaning parameters are passed.
    """
    from nilearn.conftest import _img_4d_rand_eye_medium, _make_surface_img

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


def check_nifti_masker_clean_error(estimator):
    """Nifti maskers cannot be given cleaning parameters \
        via both clean_args and kwargs simultaneously.

    TODO remove after nilearn 0.13.0
    """
    import pytest

    from nilearn.conftest import _img_4d_rand_eye_medium

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
    import pytest

    from nilearn.conftest import _img_4d_rand_eye_medium

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


def check_nifti_masker_smooth(estimator):
    """Check that masker can smooth data when extracting.

    Check that masker instance has smoothing_fwhm attribute.
    Check that output is different with and without smoothing.
    """
    from nilearn.conftest import _img_3d_rand

    assert hasattr(estimator, "smoothing_fwhm")

    signal = estimator.fit(_img_3d_rand())
    signal = estimator.transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1

    estimator.smoothing_fwhm = 3
    estimator.fit(_img_3d_rand())
    smoothed_signal = estimator.transform(_img_3d_rand())

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == 1

    assert_raises(AssertionError, assert_array_equal, smoothed_signal, signal)


def check_surface_masker_smooth(estimator):
    """Check smoothing on surface maskers raises warning.

    Check that output is the same with and without smoothing.
    TODO: update once smoothing is implemented.
    """
    import pytest

    from nilearn.conftest import _make_surface_img

    assert hasattr(estimator, "smoothing_fwhm")

    n_sample = 10

    input_img = _make_surface_img(n_sample)

    estimator.fit(input_img)

    signal = estimator.transform(input_img)

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == n_sample

    estimator.smoothing_fwhm = 3
    estimator.fit(input_img)

    with pytest.warns(UserWarning, match="not yet supported"):
        smoothed_signal = estimator.transform(input_img)

    assert estimator.smoothing_fwhm is None

    assert isinstance(signal, np.ndarray)
    assert signal.shape[0] == n_sample

    assert_array_equal(smoothed_signal, signal)


def check_nifti_masker_fit_transform_files(estimator):
    """Check that nifti maskers can work directly on files."""
    from nilearn._utils.testing import write_imgs_to_path
    from nilearn.conftest import _img_3d_rand

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
    from nilearn.conftest import _rng, _shape_3d_default

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
    from nilearn.conftest import _affine_eye, _img_3d_rand

    # TODO
    # (29, 30, 31) is used to match MAP_SHAPE in
    # nilearn/regions/tests/test_region_extractor.py
    # this test would fail for RegionExtractor otherwise
    mask = np.ones((29, 30, 31))
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit([_img_3d_rand()])

    assert hasattr(estimator, "mask_img_")


def check_nifti_masker_fit_with_only_mask(estimator):
    """Check 3D mask is enough to run with nifti maskers."""
    from nilearn.conftest import _affine_eye

    mask = np.ones((29, 30, 31))
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit()

    assert hasattr(estimator, "mask_img_")

    assert estimator.mask_img_ is mask_img


def check_nifti_masker_fit_with_empty_mask(estimator):
    """Check mask that excludes all voxels raise an error."""
    import pytest

    from nilearn.conftest import _img_3d_rand, _img_3d_zeros

    estimator.mask_img = _img_3d_zeros()
    with pytest.raises(
        ValueError,
        match="The mask is invalid as it is empty: it masks all data",
    ):
        estimator.fit([_img_3d_rand()])


def check_nifti_masker_fit_with_4d_mask(estimator):
    """Check 4D mask cannot be used with nifti maskers."""
    import pytest

    from nilearn.conftest import _img_3d_rand, _img_4d_zeros

    with pytest.raises(DimensionError, match="Expected dimension is 3D"):
        estimator.mask_img = _img_4d_zeros()
        estimator.fit([_img_3d_rand()])


def _generate_report_with_no_warning(estimator):
    """Check that report generation throws no warning."""
    from nilearn.conftest import _n_regions
    from nilearn.reporting.tests.test_html_report import _check_html

    with warnings.catch_warnings(record=True) as warning_list:
        if is_mapsmasker(estimator):
            if accept_niimg_input(estimator):
                report = estimator.generate_report(displayed_maps=_n_regions())
            else:
                report = estimator.generate_report(
                    displayed_maps=estimator.maps_img_.shape[1]
                )
        else:
            report = estimator.generate_report()
        assert len(warning_list) == 0, [
            str(warning.message) for warning in warning_list
        ]

    _check_html(report)

    return report


def check_masker_generate_report(estimator):
    """Check that maskers can generate report.

    - check that we get a warning:
      - when matplotlib is not installed
      - when generating reports before fit
    - check content of report before fit and after fit

    """
    from nilearn.conftest import _img_3d_rand, _make_surface_img
    from nilearn.reporting.tests.test_html_report import _check_html

    if not is_matplotlib_installed():
        with warnings.catch_warnings(record=True) as warning_list:
            result = estimator.generate_report()

        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, ImportWarning)
        assert result == [None]

        return

    with warnings.catch_warnings(record=True) as warning_list:
        report = estimator.generate_report()
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
    # SurfaceMapsMasker still throws a warning
    # report = _generate_report_with_no_warning(estimator)
    report = estimator.generate_report()
    _check_html(report)

    with TemporaryDirectory() as tmp_dir:
        report.save_as_html(Path(tmp_dir) / "report.html")
        assert (Path(tmp_dir) / "report.html").is_file()


def check_nifti_masker_generate_report_after_fit_with_only_mask(estimator):
    """Check 3D mask is enough to run with fit and generate report."""
    import pytest

    from nilearn.conftest import _affine_eye, _img_4d_rand_eye_medium
    from nilearn.reporting.tests.test_html_report import _check_html

    mask = np.ones((29, 30, 31))
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit()

    assert estimator._report_content["warning_message"] is None

    with pytest.warns(UserWarning, match="No image provided to fit."):
        report = estimator.generate_report()
    _check_html(report)

    input_img = _img_4d_rand_eye_medium()

    estimator.fit(input_img)

    # TODO
    # NiftiSpheresMasker still throws a warning
    # report = _generate_report_with_no_warning(estimator)
    report = estimator.generate_report()
    _check_html(report)


def check_masker_generate_report_false(estimator):
    """Test with reports set to False."""
    import pytest

    from nilearn.conftest import _img_4d_rand_eye_medium, _make_surface_img
    from nilearn.reporting.tests.test_html_report import _check_html

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
        report = estimator.generate_report()

    _check_html(report, reports_requested=False)

    assert "Empty Report" in str(report)


def get_params(cls, instance, ignore=None):
    """Retrieve the initialization parameters corresponding to a class.

    This helper function retrieves the parameters of function __init__ for
    class 'cls' and returns the value for these parameters in object
    'instance'. When using a composition pattern (e.g. with a NiftiMasker
    class), it is useful to forward parameters from one instance to another.

    Parameters
    ----------
    cls : class
        The class that gives us the list of parameters we are interested in.

    instance : object, instance of BaseEstimator
        The object that gives us the values of the parameters.

    ignore : None or list of strings
        Names of the parameters that are not returned.

    Returns
    -------
    params : dict
        The dict of parameters.

    """
    _ignore = {"memory", "memory_level", "verbose", "copy", "n_jobs"}
    if ignore is not None:
        _ignore.update(ignore)

    param_names = cls._get_param_names()

    params = {}
    for param_name in param_names:
        if param_name in _ignore:
            continue
        if hasattr(instance, param_name):
            params[param_name] = getattr(instance, param_name)

    return params
