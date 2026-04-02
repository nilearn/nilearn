import pytest
from sklearn import clone

from nilearn._utils.helpers import is_gil_enabled, is_matplotlib_installed
from nilearn._utils.html_document import WIDTH_DEFAULT
from nilearn._utils.versions import SKLEARN_GTE_1_7
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
from nilearn.reporting import HTMLReport
from nilearn.reporting.html_report import MISSING_ENGINE_MSG

COMMON_PARAMS = [
    ((NiftiMasker, None), _img_3d_rand, {}),
    ((SurfaceMasker, None), _make_surface_img, {}),
    ((MultiNiftiMasker, None), _img_3d_rand, {}),
    ((MultiSurfaceMasker, None), _make_surface_img, {}),
    (
        (NiftiMapsMasker, "img_maps"),
        _img_3d_rand,
        {"displayed_maps": 1},
    ),
    (
        (MultiNiftiMapsMasker, "img_maps"),
        _img_3d_rand,
        {"displayed_maps": 1},
    ),
    (
        (SurfaceMapsMasker, "surf_maps_img"),
        _make_surface_img,
        {"displayed_maps": 1},
    ),
    (
        (MultiSurfaceMapsMasker, "surf_maps_img"),
        _make_surface_img,
        {"displayed_maps": 1},
    ),
    ((NiftiLabelsMasker, "img_labels"), _img_3d_rand, {}),
    ((MultiNiftiLabelsMasker, "img_labels"), _img_3d_rand, {}),
    ((SurfaceLabelsMasker, "surf_label_img"), _make_surface_img, {}),
    (
        (MultiSurfaceLabelsMasker, "surf_label_img"),
        _make_surface_img,
        {},
    ),
    (
        (NiftiSpheresMasker, {"seeds": [(1, 1, 1)], "radius": 1}),
        _img_3d_rand,
        {"displayed_spheres": 1},
    ),
]


@pytest.fixture
def masker(
    request, img_maps, surf_maps_img, img_labels, surf_label_img, reports
):
    """Fixture to construct a masker instance with proper input."""
    cls, arg = request.param

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


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_reporting_initial_state(
    reports,
    masker,
    img_func,  # noqa: ARG001
    kwargs,  # noqa: ARG001
):
    """Test nilearn.maskers._mixin._ReportingMixin on concrete masker
    instances for initial state.
    """
    masker = clone(masker)

    # check masker at initialization
    assert masker._report_content is not None
    assert masker._report_content["description"] is not None
    assert masker._report_content["warning_messages"] == []
    assert masker._report_content["summary"] == {}
    assert masker._has_report_data() is False
    assert masker.reports == reports


@pytest.mark.skipif(
    not is_matplotlib_installed(), reason="fails without matplotlib"
)
@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_report_html_before_fit(
    masker,
    img_func,  # noqa: ARG001
    kwargs,
    reports,  # noqa: ARG001
):
    """Test report html generated with
    nilearn.maskers._mixin._ReportingMixin.generate_report before fitting the
    masker.
    """
    masker = clone(masker)
    # generate report without fitting the masker
    report = masker.generate_report(**kwargs)
    # report.open_in_browser()

    assert isinstance(report, HTMLReport)

    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report.get_iframe()

    # resize width and height
    report.resize(1200, 800)
    assert report.width == 1200
    assert report.height == 800

    # invalid values fall back on default dimensions
    with pytest.warns(UserWarning, match="Using default instead"):
        report.width = "foo"
    assert report.width == WIDTH_DEFAULT

    assert report._repr_html_() == report.body

    report_str = str(report)
    assert "No plotting engine found" not in report_str

    if not SKLEARN_GTE_1_7:
        assert "<th>Parameter</th>" in report_str
    else:
        assert 'div id="sk-container-id' in report_str

    assert "data:image/" in report_str
    assert 'id="warnings"' in report_str


@pytest.mark.skipif(is_matplotlib_installed(), reason="fails with matplotlib")
@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_report_html_before_fit_no_matplotlib(
    masker,
    img_func,  # noqa: ARG001
    kwargs,
    reports,  # noqa: ARG001
):
    """Test report html generated with
    nilearn.maskers._mixin._ReportingMixin.generate_report before fitting the
    masker.
    """
    masker = clone(masker)
    # generate report without fitting the masker
    report = masker.generate_report(**kwargs)
    # report.open_in_browser()

    assert isinstance(report, HTMLReport)

    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report.get_iframe()

    # resize width and height
    report.resize(1200, 800)
    assert report.width == 1200
    assert report.height == 800

    # invalid values fall back on default dimensions
    with pytest.warns(UserWarning, match="Using default instead"):
        report.width = "foo"
    assert report.width == WIDTH_DEFAULT

    assert report._repr_html_() == report.body

    report_str = str(report)
    assert "No plotting engine found" in report_str
    assert 'grey">' in report_str

    if not SKLEARN_GTE_1_7:
        assert "<th>Parameter</th>" in report_str
    else:
        assert 'div id="sk-container-id' in report_str

    assert "data:image/" not in report_str
    assert 'id="warnings"' in report_str


@pytest.mark.skipif(
    not is_matplotlib_installed(), reason="fails without matplotlib"
)
@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_report_content_without_fit(
    masker,
    img_func,  # noqa: ARG001
    kwargs,
    reports,
):
    """Test report content generated with
    nilearn.maskers._mixin._ReportingMixin.generate_report before fitting the
    masker.
    """
    masker = clone(masker)
    # generate report without fitting the masker
    masker.generate_report(**kwargs)

    report_content = masker._report_content
    assert report_content["unique_id"]
    assert report_content["title"] == masker.__class__.__name__
    assert report_content["has_plotting_engine"] is True
    assert report_content["engine"] == "matplotlib"
    assert report_content["n_elements"] == 0
    assert "report_at_fit_time" not in report_content
    assert "summary_html" not in report_content
    assert "summary" in report_content

    # check warning messages
    warning_messages = masker._report_content["warning_messages"]
    assert any(
        "This estimator has not been fit yet." in message
        for message in warning_messages
    )

    assert all(
        "\nReport generation was disabled when fit was run." not in message
        for message in warning_messages
    )
    if not reports:
        assert any(
            "\nReport generation not enabled!\nNo visual outputs created."
            in message
            for message in warning_messages
        )


@pytest.mark.skipif(is_matplotlib_installed(), reason="fails with matplotlib")
@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_report_content_before_fit_when_no_matplotlib(
    masker,
    img_func,  # noqa: ARG001
    kwargs,
    reports,
):
    """Test report content generated with
    nilearn.maskers._mixin._ReportingMixin.generate_report before fitting the
    masker when matplotlib is not installed.
    """
    masker = clone(masker)
    # generate report without fitting the masker
    masker.generate_report(**kwargs)

    report_content = masker._report_content
    assert report_content["unique_id"]
    assert report_content["title"] == masker.__class__.__name__
    assert report_content["has_plotting_engine"] is False
    assert report_content["engine"] == "matplotlib"
    assert report_content["n_elements"] == 0
    assert "report_at_fit_time" not in report_content
    assert "summary_html" not in report_content
    assert "summary" in report_content

    # check warning messages
    warning_messages = masker._report_content["warning_messages"]
    assert any(
        "This estimator has not been fit yet." in message
        for message in warning_messages
    )

    assert any(MISSING_ENGINE_MSG in message for message in warning_messages)

    assert all(
        "\nReport generation was disabled when fit was run." not in message
        for message in warning_messages
    )
    if not reports:
        assert any(
            "\nReport generation not enabled!\nNo visual outputs created."
            in message
            for message in warning_messages
        )


@pytest.mark.thread_unsafe
@pytest.mark.skipif(not is_gil_enabled(), reason="fails without GIL")
@pytest.mark.skipif(
    not is_matplotlib_installed(), reason="fails without matplotlib"
)
@pytest.mark.parametrize("reports", [True])
@pytest.mark.parametrize(
    "masker, img_func, kwargs",
    COMMON_PARAMS,
    indirect=["masker"],
)
def test_masker_report_content_after_fit(masker, img_func, kwargs, reports):
    """Test nilearn.maskers._mixin._ReportingMixin.generate_report without
    fitting the masker.
    """
    masker = clone(masker)
    input_imgs = img_func()
    masker.fit(input_imgs)

    masker.generate_report(**kwargs)
    report_content = masker._report_content
    assert report_content["unique_id"]
    assert report_content["title"] == masker.__class__.__name__
    assert report_content["has_plotting_engine"] is True
    assert report_content["engine"] == "matplotlib"
    assert "n_elements" in report_content
    assert "summary" in report_content
    assert report_content["reports_at_fit_time"] == reports
    assert masker._has_report_data() == reports
    warning_messages = masker._report_content["warning_messages"]

    assert all(
        "This estimator has not been fit yet." not in message
        for message in warning_messages
    )

    if not reports:
        assert any(
            "\nReport generation not enabled!\nNo visual outputs created."
            in message
            for message in warning_messages
        )
        assert any(
            "Report generation was disabled when fit was run." in message
            for message in warning_messages
        )
