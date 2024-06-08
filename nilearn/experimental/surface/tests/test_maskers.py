import numpy as np
import pytest

from nilearn.experimental.surface import (
    SurfaceImage,
    SurfaceLabelsMasker,
    SurfaceMasker,
)


def test_mask_img_fit_shape_mismatch(
    flip_img, mini_mask, make_mini_img, assert_img_equal
):
    img = make_mini_img()
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="number of vertices"):
        masker.fit(flip_img(img))
    # fitting with the number of vertices is ok
    masker.fit(img)
    # fitting with the same number of vertices and extra dimensions (surface
    # timeseries) is ok
    masker.fit(make_mini_img((2,)))
    assert_img_equal(mini_mask, masker.mask_img_)


def test_mask_img_fit_keys_mismatch(mini_mask, drop_img_part):
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="key"):
        masker.fit(drop_img_part(mini_mask))


def test_none_mask_img(mini_mask):
    masker = SurfaceMasker(None)
    with pytest.raises(ValueError, match="provide either"):
        masker.fit(None)
    # no mask_img but fit argument is ok
    masker.fit(mini_mask)
    # no fit argument but a mask_img is ok
    SurfaceMasker(mini_mask).fit(None)


def test_unfitted_masker(mini_mask):
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="fitted"):
        masker.transform(mini_mask)


def test_mask_img_transform_shape_mismatch(flip_img, mini_img, mini_mask):
    masker = SurfaceMasker(mini_mask).fit()
    with pytest.raises(ValueError, match="number of vertices"):
        masker.transform(flip_img(mini_img))
    # non-flipped is ok
    masker.transform(mini_img)


def test_mask_img_transform_keys_mismatch(mini_mask, mini_img, drop_img_part):
    masker = SurfaceMasker(mini_mask).fit()
    with pytest.raises(ValueError, match="key"):
        masker.transform(drop_img_part(mini_img))
    # full img is ok
    masker.transform(mini_img)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
def test_transform_inverse_transform(shape, make_mini_img, assert_img_equal):
    img = make_mini_img(shape)
    masker = SurfaceMasker().fit(img)
    masked_img = masker.transform(img)
    assert np.array_equal(
        masked_img.ravel()[:9], [1, 2, 3, 4, 10, 20, 30, 40, 50]
    )
    assert masked_img.shape == shape + (img.shape[-1],)
    unmasked_img = masker.inverse_transform(masked_img)
    assert_img_equal(img, unmasked_img)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
def test_transform_inverse_transform_with_mask(
    shape, make_mini_img, assert_img_equal, mini_mask
):
    img = make_mini_img(shape)
    masker = SurfaceMasker(mini_mask).fit(img)
    masked_img = masker.transform(img)
    assert masked_img.shape == shape + (img.shape[-1] - 2,)
    assert np.array_equal(masked_img.ravel()[:7], [2, 3, 4, 20, 30, 40, 50.0])
    unmasked_img = masker.inverse_transform(masked_img)
    expected_data = {k: v.copy() for (k, v) in img.data.parts.items()}
    for v in expected_data.values():
        v[..., 0] = 0.0
    expected_img = SurfaceImage(img.mesh, expected_data)
    assert_img_equal(expected_img, unmasked_img)


def test_surface_masker_report_empty(tmp_path):
    """Check empty report contains a warning and no image."""
    masker = SurfaceMasker()
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    with open(tmp_path / "report.html") as f:
        content = f.read()
    assert "This object has not been fitted yet !" in content
    assert '<div class="image">' not in content


def test_surface_masker_report_no_mask(mini_img, tmp_path):
    """Check fitted masker report has image and no warning."""
    masker = SurfaceMasker()
    masker.fit_transform(mini_img)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    with open(tmp_path / "report.html") as f:
        content = f.read()
    assert "This object has not been fitted yet !" not in content
    assert '<div class="image">' in content


def test_surface_masker_report(mini_img, tmp_path, mini_binary_mask):
    """Check fitted masker report with mask has image and no warning."""
    masker = SurfaceMasker(mini_binary_mask)
    masker.fit_transform(mini_img)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "report.html")

    assert (tmp_path / "report.html").exists()
    with open(tmp_path / "report.html") as f:
        content = f.read()
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
    with open(tmp_path / "report.html") as f:
        content = f.read()
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
