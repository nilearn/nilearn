import pytest

from nilearn.conftest import _img_3d_rand, _make_surface_img
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    MultiSurfaceLabelsMasker,
    MultiSurfaceMapsMasker,
    MultiSurfaceMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)


@pytest.fixture
def masker(request, img_maps, surf_maps_img, img_labels, surf_label_img):
    """Fixture to construct a masker instance with proper input."""
    cls, arg, reports = request.param

    img_generators = {
        "img_maps": img_maps,
        "surf_maps_img": surf_maps_img,
        "img_labels": img_labels,
        "surf_label_img": surf_label_img,
    }
    if arg is None:
        return cls(reports=reports)
    elif isinstance(arg, dict):
        return cls(reports=reports, **arg)
    else:
        img = img_generators[arg]
        return cls(img, reports=reports)


@pytest.mark.slow
@pytest.mark.parametrize(
    "masker, img_func",
    [
        ((NiftiMasker, None, True), _img_3d_rand),
        ((SurfaceMasker, None, True), _make_surface_img),
        ((MultiNiftiMasker, None, True), _img_3d_rand),
        ((MultiSurfaceMasker, None, True), _make_surface_img),
        ((NiftiMapsMasker, "img_maps", True), _img_3d_rand),
        ((MultiNiftiMapsMasker, "img_maps", True), _img_3d_rand),
        ((SurfaceMapsMasker, "surf_maps_img", True), _make_surface_img),
        ((MultiSurfaceMapsMasker, "surf_maps_img", True), _make_surface_img),
        ((NiftiLabelsMasker, "img_labels", True), _img_3d_rand),
        ((MultiNiftiLabelsMasker, "img_labels", True), _img_3d_rand),
        ((SurfaceLabelsMasker, "surf_label_img", True), _make_surface_img),
        (
            (MultiSurfaceLabelsMasker, "surf_label_img", True),
            _make_surface_img,
        ),
        (
            (NiftiSpheresMasker, {"seeds": [(1, 1, 1)], "radius": 1}, True),
            _img_3d_rand,
        ),
    ],
    indirect=["masker"],
)
def test_masker_reporting_true(masker, img_func):
    """Test nilearn.maskers._mixin._ReportingMixin on concrete masker
    instances when ``reports=True``.
    """
    # check masker at initialization
    assert masker._report_content["warning_message"] is None

    # check masker report before fit
    masker.generate_report()
    assert masker._has_report_data() is False

    # check masker after fit
    input_img = img_func()
    masker.fit(input_img)
    assert masker._has_report_data()

    # check masker report without title specified
    masker.generate_report()
    assert masker._report_content["title"] == masker.__class__.__name__

    # check masker report with title specified
    masker.generate_report(title="masker report title")
    assert masker._report_content["title"] == "masker report title"


@pytest.mark.parametrize(
    "masker, img_func",
    [
        ((NiftiMasker, None, False), _img_3d_rand),
        ((SurfaceMasker, None, False), _make_surface_img),
        ((MultiNiftiMasker, None, False), _img_3d_rand),
        ((MultiSurfaceMasker, None, False), _make_surface_img),
        ((NiftiMapsMasker, "img_maps", False), _img_3d_rand),
        ((MultiNiftiMapsMasker, "img_maps", False), _img_3d_rand),
        ((SurfaceMapsMasker, "surf_maps_img", False), _make_surface_img),
        ((MultiSurfaceMapsMasker, "surf_maps_img", False), _make_surface_img),
        ((NiftiLabelsMasker, "img_labels", False), _img_3d_rand),
        ((MultiNiftiLabelsMasker, "img_labels", False), _img_3d_rand),
        ((SurfaceLabelsMasker, "surf_label_img", False), _make_surface_img),
        (
            (MultiSurfaceLabelsMasker, "surf_label_img", False),
            _make_surface_img,
        ),
        (
            (NiftiSpheresMasker, {"seeds": [(1, 1, 1)], "radius": 1}, False),
            _img_3d_rand,
        ),
    ],
    indirect=["masker"],
)
def test_masker_reporting_false(masker, img_func):
    """Test nilearn.maskers._mixin._ReportingMixin on concrete masker
    instances when ``reports=False``.
    """
    # check masker at initialization
    assert masker._report_content is not None
    assert masker._report_content["description"] is not None
    assert masker._report_content["warning_message"] is None
    assert masker._has_report_data() is False

    # check masker report before fit
    masker.generate_report()
    assert masker._has_report_data() is False

    # check masker after fit
    input_img = img_func()
    masker.fit(input_img)
    assert masker._has_report_data() is False

    # check masker report without title specified
    masker.generate_report()

    # check masker report with title specified
    masker.generate_report(title="masker report title")
