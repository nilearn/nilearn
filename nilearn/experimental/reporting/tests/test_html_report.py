import pytest

from nilearn.experimental.surface import SurfaceLabelsMasker, SurfaceMasker
from nilearn.reporting.tests.test_html_report import _check_html


@pytest.mark.parametrize("reports", [True, False])
def test_surface_masker_minimal_report(mini_img, reports):
    """Test minimal report generation."""
    masker = SurfaceMasker(reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "has not been fitted yet" in str(report)

    masker.fit_transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "has not been fitted yet" not in str(report)
    assert '<div class="image">' in str(report)


def test_surface_masker_report_no_report(mini_img):
    """Check content of no report."""
    masker = SurfaceMasker(reports=False)
    masker.fit_transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "Empty Report" in str(report)


def test_surface_masker_report_with_mask(mini_img, mini_binary_mask):
    """Check fitted masker report with mask has image and no warning."""
    masker = SurfaceMasker(mini_binary_mask)
    masker.fit_transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "This object has not been fitted yet !" not in str(report)
    assert '<div class="image">' in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_unfitted(
    mini_label_img, label_names, reports
):
    masker = SurfaceLabelsMasker(mini_label_img, label_names, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "has not been fitted yet" not in str(report)
    assert '<td data-column="Number of vertices">4</td>' in str(report)
    assert '<td data-column="Number of vertices">4</td>' in str(report)
    # TODO this should say 2 regions: related to the fact that some regions
    #  have a 0 label
    assert "The masker has <b>1</b> different non-overlapping" in str(report)


@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_empty(mini_label_img, label_names):
    masker = SurfaceLabelsMasker(mini_label_img, label_names).fit()
    report = masker.generate_report()

    _check_html(report)
    assert "has not been fitted yet" not in str(report)


@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report(mini_img, mini_label_img, label_names):
    masker = SurfaceLabelsMasker(mini_label_img, label_names).fit()
    masker.transform(mini_img)
    report = masker.generate_report()

    _check_html(report)
    assert "has not been fitted yet" not in str(report)
