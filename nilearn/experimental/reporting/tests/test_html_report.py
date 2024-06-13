import pathlib

import pytest

from nilearn.experimental.surface import SurfaceLabelsMasker, SurfaceMasker


def test_surface_masker_report_empty(tmp_path):
    """Check empty report contains a warning and no image."""
    masker = SurfaceMasker()
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    content = (tmp_path / "report.html").read_text()
    assert "has not been fitted yet" in content
    assert 'src="data:image/svg+xml;base64,"\n' in content


def test_surface_masker_report_no_mask(mini_img, tmp_path):
    """Check fitted masker report has image and no warning."""
    masker = SurfaceMasker()
    masker.fit_transform(mini_img)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    content = pathlib.Path(tmp_path / "report.html").read_text()
    assert "This object has not been fitted yet !" not in content
    assert '<div class="image">' in content


def test_surface_masker_report(mini_img, tmp_path, mini_binary_mask):
    """Check fitted masker report with mask has image and no warning."""
    masker = SurfaceMasker(mini_binary_mask)
    masker.fit_transform(mini_img)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    content = (tmp_path / "report.html").read_text()
    assert "This object has not been fitted yet !" not in content
    assert '<div class="image">' in content


def test_surface_masker_report_bug(mini_img, tmp_path):
    masker = SurfaceMasker()
    report = masker.generate_report()

    masker_2 = SurfaceMasker()
    masker_2.fit(mini_img)
    report = masker_2.generate_report()
    report.save_as_html(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()


@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_unfitted(
    mini_label_img, label_names, tmp_path
):
    masker = SurfaceLabelsMasker(mini_label_img, label_names)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()
    content = (tmp_path / "report.html").read_text()
    assert '<div class="image">' in content
    assert '<td data-column="Number of vertices">4</td>' in content
    assert '<td data-column="Number of vertices">4</td>' in content
    # TODO this should say 2 regions: related to the fact that some regions
    #  have a 0 label
    assert "The masker has <b>1</b> different non-overlapping" in content


@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_empty(
    mini_label_img, label_names, tmp_path
):
    masker = SurfaceLabelsMasker(mini_label_img, label_names).fit()
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()


@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report(
    mini_img, mini_label_img, label_names, tmp_path
):
    masker = SurfaceLabelsMasker(mini_label_img, label_names).fit()
    masker.transform(mini_img)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()
