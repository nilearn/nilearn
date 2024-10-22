import pytest

from nilearn.experimental.surface import SurfaceLabelsMasker, SurfaceMasker
from nilearn.reporting.tests.test_html_report import _check_html


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("empty_mask", [True, False])
def test_surface_masker_minimal_report_no_fit(surf_mask, empty_mask, reports):
    """Test minimal report generation with no fit."""
    masker = SurfaceMasker(surf_mask(empty=empty_mask), reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("empty_mask", [True, False])
def test_surface_masker_minimal_report_fit(
    surf_mask, empty_mask, surf_img, reports
):
    """Test minimal report generation with fit."""
    masker = SurfaceMasker(surf_mask(empty=empty_mask), reports=reports)
    img = surf_img()
    masker.fit_transform(img)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" not in str(report)
    assert '<div class="image">' in str(report)


def test_surface_masker_report_no_report(surf_img):
    """Check content of no report."""
    masker = SurfaceMasker(reports=False)
    img = surf_img()
    masker.fit_transform(img)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_unfitted(
    surf_label_img, label_names, reports
):
    label_img = surf_label_img()
    masker = SurfaceLabelsMasker(label_img, label_names, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    # contrary to other maskers information about the label image
    # can be shown before fitting
    assert "Make sure to run `fit`" not in str(report)


def test_surface_label_masker_report_no_report(surf_label_img):
    """Check content of no report."""
    label_img = surf_label_img()
    masker = SurfaceLabelsMasker(label_img, reports=False)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)
