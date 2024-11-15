import warnings

import numpy as np
import pytest
from sklearn import __version__ as sklearn_version

from nilearn._utils import compare_version
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers import SurfaceLabelsMasker, SurfaceMasker

extra_valid_checks = [
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
    "check_estimator_repr",
    "check_estimator_cloneable",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_unfitted",
    "check_mixin_order",
    "check_estimator_tags_renamed",
]
# TODO remove when dropping support for sklearn_version < 1.5.0
if compare_version(sklearn_version, "<", "1.5.0"):
    extra_valid_checks.append("check_estimator_sparse_data")


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMasker()], extra_valid_checks=extra_valid_checks
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMasker()],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_surface_label_masker_fit(surf_label_img):
    """Test fit and check estimated attributes.

    0 value in data is considered as background
    and should not be listed in the labels.
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()
    assert masker.n_elements_ == 1
    assert masker._labels_ == [1]
    assert masker._label_names_ == ["1"]
    assert masker._reporting_data is not None


def test_surface_label_masker_fit_with_names(surf_label_img):
    """Check passing labels is reflected in attributes.

    - the value corresponding to 0 (background) is omitted
    - extra value provided (foo) are not listed in attributes
    """
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["background", "bar", "foo"]
    )
    masker = masker.fit()
    assert masker.n_elements_ == 1
    assert masker._labels_ == [1]
    assert masker._label_names_ == ["bar"]


def test_surface_label_masker_fit_no_report(surf_label_img):
    """Check no report data is stored."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, reports=False)
    masker = masker.fit()
    assert masker._reporting_data is None


def test_surface_label_masker_transform(surf_label_img, surf_img):
    """Test transform extract signals."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    # only one 'timepoint'
    signal = masker.transform(surf_img())

    assert isinstance(signal, np.ndarray)
    n_labels = len(masker._labels_)
    assert signal.shape == (1, n_labels)

    # 5 'timepoint'
    n_timepoints = 5
    signal = masker.transform(surf_img((n_timepoints,)))

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (n_timepoints, n_labels)


def test_surface_label_masker_transform_clean(surf_label_img, surf_img):
    """Smoke test for clean args."""
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img,
        t_r=2.0,
        high_pass=1 / 128,
        clean_args={"filter": "cosine"},
    ).fit()
    masker.transform(surf_img((50,)))


def test_surface_label_masker_fit_transform(surf_label_img, surf_img):
    """Smoke test for fit_transform."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker.fit_transform(surf_img())


def test_error_transform_before_fit(surf_label_img, surf_img):
    """Transform requires masker to be fitted."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    with pytest.raises(ValueError, match="has not been fitted"):
        masker.transform(surf_img())


def test_surface_label_masker_inverse_transform(surf_label_img, surf_img):
    """Test transform extract signals."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()
    signal = masker.transform(surf_img())
    masker.inverse_transform(signal)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_masker_reporting_mpl_warning(surf_label_img):
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceLabelsMasker(surf_label_img).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
