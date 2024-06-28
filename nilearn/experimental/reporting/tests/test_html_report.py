import pytest

from nilearn.experimental.conftest import return_mini_binary_mask
from nilearn.experimental.surface import SurfaceLabelsMasker, SurfaceMasker
from nilearn.reporting.tests.test_html_report import _check_html


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("mask_img", [None, return_mini_binary_mask()])
def test_surface_masker_minimal_report_no_fit(mask_img, reports):
    """Test minimal report generation with no fit."""
    masker = SurfaceMasker(mask_img, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("mask_img", [None, return_mini_binary_mask()])
def test_surface_masker_minimal_report_fit(mask_img, mini_img, reports):
    """Test minimal report generation with fit."""
    masker = SurfaceMasker(mask_img, reports=reports)
    masker.fit_transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" not in str(report)
    assert '<div class="image">' in str(report)


def test_surface_masker_report_no_report(mini_img):
    """Check content of no report."""
    masker = SurfaceMasker(reports=False)
    masker.fit_transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_unfitted(
    mini_label_img, label_names, reports
):
    masker = SurfaceLabelsMasker(mini_label_img, label_names, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    # contrary to other maskers information about the label image
    # can be shown before fitting
    assert "Make sure to run `fit`" not in str(report)


def test_surface_label_masker_report_no_report(mini_label_img):
    """Check content of no report."""
    masker = SurfaceLabelsMasker(mini_label_img, reports=False)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)
