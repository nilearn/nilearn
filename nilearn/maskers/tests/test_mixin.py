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
from nilearn.maskers.tests.test_html_report import (
    generate_and_check_masker_report,
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
    "masker, img_func, kwargs",
    [
        ((NiftiMasker, None, True), _img_3d_rand, {}),
        ((SurfaceMasker, None, True), _make_surface_img, {}),
        ((MultiNiftiMasker, None, True), _img_3d_rand, {}),
        ((MultiSurfaceMasker, None, True), _make_surface_img, {}),
        (
            (NiftiMapsMasker, "img_maps", True),
            _img_3d_rand,
            {"displayed_maps": 1},
        ),
        (
            (MultiNiftiMapsMasker, "img_maps", True),
            _img_3d_rand,
            {"displayed_maps": 1},
        ),
        (
            (SurfaceMapsMasker, "surf_maps_img", True),
            _make_surface_img,
            {"displayed_maps": 1},
        ),
        (
            (MultiSurfaceMapsMasker, "surf_maps_img", True),
            _make_surface_img,
            {"displayed_maps": 1},
        ),
        ((NiftiLabelsMasker, "img_labels", True), _img_3d_rand, {}),
        ((MultiNiftiLabelsMasker, "img_labels", True), _img_3d_rand, {}),
        ((SurfaceLabelsMasker, "surf_label_img", True), _make_surface_img, {}),
        (
            (MultiSurfaceLabelsMasker, "surf_label_img", True),
            _make_surface_img,
            {},
        ),
        (
            (NiftiSpheresMasker, {"seeds": [(1, 1, 1)], "radius": 1}, True),
            _img_3d_rand,
            {"displayed_spheres": 1},
        ),
    ],
    indirect=["masker"],
)
def test_masker_reporting_true(masker, img_func, kwargs):
    """Test nilearn.maskers._mixin._ReportingMixin on concrete masker
    instances when ``reports=True``.
    """
    # check masker at initialization
    assert masker._report_content["warning_messages"] == []

    # check masker report before fit
    generate_and_check_masker_report(masker, **kwargs)
    assert masker._has_report_data() is False

    # check masker after fit
    input_img = img_func()
    masker.fit(input_img)
    assert masker._has_report_data()

    # check masker report without title specified
    extra_warnings_allowed = False
    if isinstance(masker, SurfaceMapsMasker):
        extra_warnings_allowed = True

    generate_and_check_masker_report(
        masker, extra_warnings_allowed=extra_warnings_allowed, **kwargs
    )

    # check masker report with title specified
    generate_and_check_masker_report(
        masker,
        title="masker report title",
        extra_warnings_allowed=extra_warnings_allowed,
        **kwargs,
    )

    masker.reports = False
    match = "Report generation not enabled"
    with pytest.warns(UserWarning, match=match):
        report = masker.generate_report(**kwargs)

    assert match in str(report)


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
    assert masker._report_content["warning_messages"] == []
    assert masker._report_content["summary"] == {}
    assert masker._has_report_data() is False

    # check masker report before fit
    generate_and_check_masker_report(masker)

    assert masker._has_report_data() is False

    # check masker after fit
    input_img = img_func()
    masker.fit(input_img)

    assert masker._has_report_data() is False

    # check masker report without title specified
    generate_and_check_masker_report(masker)

    # check masker report with title specified
    generate_and_check_masker_report(masker, title="masker report title")

    # check masker report if the model is fit when reports=False
    # and reports=True is set and report generation is required
    # Regression test for https://github.com/nilearn/nilearn/issues/5831
    masker.reports = True
    generate_and_check_masker_report(
        masker,
        warnings_msg_to_check=["Report generation was disabled when fit"],
    )
