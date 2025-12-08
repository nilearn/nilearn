"""Centralize tests for masker reports.

More generic tests (those that apply to all maskers)
should go into nilearn/_utils/estimator_checks.
"""

from collections import Counter

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn._utils.html_document import WIDTH_DEFAULT, HTMLDocument
from nilearn.conftest import _img_maps, _surf_maps_img
from nilearn.image import get_data
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)
from nilearn.reporting.html_report import MISSING_ENGINE_MSG
from nilearn.surface import SurfaceImage

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _check_html(html_view, reports_requested=True, is_fit=True):
    """Check the presence of some expected code in the html viewer.

    Also ensure some common behavior to all reports.
    """
    assert isinstance(html_view, HTMLDocument)

    # resize width and height
    html_view.resize(1200, 800)
    assert html_view.width == 1200
    assert html_view.height == 800

    # invalid values fall back on default dimensions
    with pytest.warns(UserWarning, match="Using default instead"):
        html_view.width = "foo"
    assert html_view.width == WIDTH_DEFAULT

    assert html_view._repr_html_() == html_view.body

    # navbar and its css is only for GLM reports
    assert "Adapted from Pure CSS navbar" not in str(html_view)

    if not is_fit:
        assert "This estimator has not been fit yet." in str(html_view)

    if not is_matplotlib_installed():
        assert 'id="warnings"' in str(html_view)
        assert MISSING_ENGINE_MSG in str(html_view)
        assert 'color: grey">No plotting engine found</p>' in str(html_view)

    if not reports_requested:
        assert (
            "\nReport generation not enabled!\nNo visual outputs created."
            in str(html_view)
        )

    if not reports_requested or not is_fit:
        # no image present if reports not requested or masker is not fitted
        assert '<div class="image">' not in str(html_view)
    else:
        if is_fit:
            assert "<th>Parameter</th>" in str(html_view)

        if is_matplotlib_installed():
            if "Surface" in str(html_view):
                assert "data:image/png;base64," in str(html_view)
            else:
                assert "data:image/svg+xml;base64," in str(html_view)

        else:
            assert "data:image/svg+xml;base64," not in str(html_view)
            assert "data:image/png;base64," not in str(html_view)


@pytest.fixture
def niftimapsmasker_inputs():
    """Return inputs for nifti maps masker."""
    return {"maps_img": _img_maps(n_regions=3)}


@pytest.fixture
def labels(n_regions):
    """Return labels for label masker."""
    return ["background"] + [f"region_{i}" for i in range(1, n_regions + 1)]


@pytest.fixture
def input_parameters(masker_class, img_mask_eye, labels, img_labels):
    """Define inputs for each type masker."""
    if masker_class in (NiftiMasker, MultiNiftiMasker):
        return {"mask_img": img_mask_eye}
    if masker_class in (NiftiLabelsMasker, MultiNiftiLabelsMasker):
        return {"labels_img": img_labels, "labels": labels}
    if masker_class in (NiftiMapsMasker, MultiNiftiMapsMasker):
        # using 6 regions to be consistent with n_regions
        # in surface counterpart
        return {"maps_img": _img_maps(n_regions=6)}
    if masker_class is NiftiSpheresMasker:
        # using 6 seeds to be consistent with n_regions
        # in maps maskers counterpart
        return {
            "seeds": [
                (1, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
                (1, 0, 0),
                (0, 1, 1),
                (0, 0, 1),
            ]
        }
    if masker_class is SurfaceMapsMasker:
        return {"maps_img": _surf_maps_img()}


@pytest.mark.slow
@pytest.mark.parametrize(
    "masker_class",
    [NiftiMapsMasker, NiftiSpheresMasker, SurfaceMapsMasker],
)
@pytest.mark.parametrize(
    "displayed_maps, expected_displayed_maps",
    [
        (4, [0, 1, 2, 3]),
        ("all", [0, 1, 2, 3, 4, 5]),
        ([2, 1], [2, 1]),
        (np.asarray([1, 3]), [1, 3]),
    ],
)
def test_displayed_maps_valid_inputs(
    masker_class, input_parameters, displayed_maps, expected_displayed_maps
):
    """Test valid inputs for displayed_maps/spheres."""
    masker = masker_class(**input_parameters)
    masker.fit()

    html = masker.generate_report(displayed_maps)

    # sphere masker display all spheres on index 0
    # so we must offset by 1
    if isinstance(masker, NiftiSpheresMasker):
        tmp = [0]
        tmp.extend([x + 1 for x in expected_displayed_maps])
        expected_displayed_maps = tmp

    assert masker._report_content["displayed_maps"] == expected_displayed_maps

    assert html.body.count("<img") == len(expected_displayed_maps)


@pytest.mark.parametrize(
    "masker_class",
    [NiftiMapsMasker, NiftiSpheresMasker, SurfaceMapsMasker],
)
@pytest.mark.parametrize(
    "displayed_maps", [4.5, [8.4, 3], "invalid", np.asarray([1.2, 3.0])]
)
def test_displayed_maps_error(masker_class, input_parameters, displayed_maps):
    """Test invalid inputs for displayed_maps/spheres."""
    masker = masker_class(**input_parameters)
    masker.fit()
    with pytest.raises(
        TypeError,
        match=(
            "should be either 'all' or a positive 'int', "
            "or a list/array of ints"
        ),
    ):
        masker.generate_report(displayed_maps)


@pytest.mark.slow
@pytest.mark.parametrize(
    "masker_class",
    [NiftiMapsMasker, NiftiSpheresMasker, SurfaceMapsMasker],
)
@pytest.mark.parametrize("displayed_maps", [list(range(7)), [0, 66, 1]])
def test_displayed_maps_warning_too_many(
    masker_class, input_parameters, displayed_maps
):
    """Test invalid inputs for displayed_maps/spheres."""
    masker = masker_class(**input_parameters)
    masker.fit()
    with pytest.warns(
        UserWarning,
        match="Report cannot display the following",
    ):
        masker.generate_report(displayed_maps)


@pytest.mark.slow
@pytest.mark.parametrize(
    "masker_class",
    [NiftiMapsMasker, NiftiSpheresMasker, SurfaceMapsMasker],
)
def test_displayed_maps_warning_int_too_large(masker_class, input_parameters):
    """Test invalid inputs for displayed_maps/spheres."""
    masker = masker_class(**input_parameters)
    masker.fit()
    with pytest.warns(
        UserWarning,
        match="was set to 6",
    ):
        masker.generate_report(7)


def test_nifti_spheres_masker_report_1_sphere(
    matplotlib_pyplot,  # noqa: ARG001
):
    """Check the report for sphere actually works for one sphere.

    See https://github.com/nilearn/nilearn/issues/4268
    """
    report = NiftiSpheresMasker([(1, 1, 1)]).fit().generate_report()

    empty_div = """
                    <img id="map1" class="pure-img" width="100%"
                        src="data:image/svg+xml;base64,D"
                        style="display:none;" alt="image"/>"""

    assert empty_div not in report.body


def test_nifti_labels_masker_report_no_image_for_fit(
    img_3d_rand_eye, n_regions, labels, img_labels
):
    """Check no contour in image when no image was provided to fit."""
    masker = NiftiLabelsMasker(img_labels, labels=labels)
    masker.fit()

    # No image was provided to fit, regions are plotted using
    # plot_roi such that no contour should be in the image
    display = masker._reporting()

    if not is_matplotlib_installed():
        assert display is None
        return

    for d in ["x", "y", "z"]:
        assert len(display.axes[d].ax.collections) == 0

    masker.fit(img_3d_rand_eye)

    display = masker._reporting()
    for d in ["x", "y", "z"]:
        assert len(display.axes[d].ax.collections) > 0
        assert len(display.axes[d].ax.collections) <= n_regions


EXPECTED_COLUMNS = [
    "label value",
    "region name",
    "size (in mm^3)",
    "relative size (in %)",
]


def test_nifti_labels_masker_report(
    img_3d_rand_eye,
    img_mask_eye,
    affine_eye,
    n_regions,
    labels,
    img_labels,
):
    """Check content nifti label masker."""
    masker = NiftiLabelsMasker(
        img_labels,
        labels=labels,
        mask_img=img_mask_eye,
        keep_masked_labels=True,
    )
    masker.fit_transform(img_3d_rand_eye)
    report = masker.generate_report()

    assert masker._reporting_data is not None

    # Check that background label was left as default
    assert masker.background_label == 0
    assert masker._report_content["description"] == (
        "This report shows the regions defined by the labels of the mask."
    )

    # Check that the number of regions is correct
    assert masker._report_content["number_of_regions"] == n_regions

    # Check that all expected columns are present with the right size
    assert (
        masker._report_content["summary"]["region name"].to_list()
        == labels[1:]
    )
    assert len(masker._report_content["summary"]) == n_regions
    for col in EXPECTED_COLUMNS:
        assert col in masker._report_content["summary"].columns

    # Check that labels match

    # Relative sizes of regions should sum to 100%
    assert_almost_equal(
        sum(masker._report_content["summary"]["relative size (in %)"]),
        100,
        decimal=2,
    )

    _check_html(report)

    assert "Regions summary" in str(report)

    # Check region sizes calculations
    expected_region_sizes = Counter(get_data(img_labels).ravel())
    for r in range(1, n_regions + 1):
        assert_almost_equal(
            masker._report_content["summary"]["size (in mm^3)"].to_list()[
                r - 1
            ],
            expected_region_sizes[r]
            * np.abs(np.linalg.det(affine_eye[:3, :3])),
        )


@pytest.mark.slow
@pytest.mark.parametrize("masker_class", [NiftiLabelsMasker])
def test_nifti_labels_masker_report_cut_coords(
    matplotlib_pyplot,  # noqa: ARG001
    masker_class,
    input_parameters,
    img_3d_rand_eye,
):
    """Test cut coordinate are equal with and without passing data to fit."""
    masker = masker_class(**input_parameters, reports=True)
    # Get display without data
    masker.fit()
    display = masker._reporting()
    # Get display with data
    masker.fit(img_3d_rand_eye)
    display_data = masker._reporting()
    assert display.cut_coords == display_data.cut_coords


def test_nifti_masker_4d_reports(img_mask_eye, affine_eye):
    """Test for NiftiMasker reports with 4D data."""
    # Dummy 4D data
    data = np.zeros((10, 10, 10, 3), dtype="int32")
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, affine_eye)

    # test .fit method
    masker = NiftiMasker(mask_strategy="epi")
    masker.fit(data_img_4d)

    assert float(masker._report_content["coverage"]) > 0

    html = masker.generate_report()
    _check_html(html)
    assert "The mask includes" in str(html)

    # test .fit_transform method
    masker = NiftiMasker(mask_img=img_mask_eye, standardize="zscore_sample")
    masker.fit_transform(data_img_4d)

    html = masker.generate_report()
    _check_html(html)


def test_nifti_masker_overlaid_report(
    matplotlib_pyplot,  # noqa: ARG001
    img_fmri,
):
    """Check empty report generated before fit and with image after."""
    masker = NiftiMasker(
        mask_strategy="whole-brain-template",
        mask_args={"threshold": 0.0},
        target_affine=np.eye(3) * 3,
    )
    masker.fit(img_fmri)
    html = masker.generate_report()

    assert '<div class="overlay">' in str(html)


def test_multi_nifti_masker_generate_report_mask(
    img_3d_ones_eye, shape_3d_default, affine_eye
):
    """Smoke test for generate_report method with only mask."""
    masker = MultiNiftiMasker(
        mask_img=img_3d_ones_eye,
        # to test resampling lines without imgs
        target_affine=affine_eye,
        target_shape=shape_3d_default,
    )
    masker.fit().generate_report()


@pytest.mark.slow
def test_multi_nifti_masker_generate_report_imgs_and_mask(
    shape_3d_default, affine_eye, img_fmri
):
    """Smoke test for generate_report method with images and mask."""
    mask = Nifti1Image(np.ones(shape_3d_default), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
        # to test resampling lines with imgs
        target_affine=affine_eye,
        target_shape=shape_3d_default,
    )
    masker.fit([img_fmri, img_fmri]).generate_report()


def test_surface_masker_mask_img_generate_report(surf_img_1d, surf_mask_1d):
    """Smoke test generate report."""
    masker = SurfaceMasker(surf_mask_1d, reports=True).fit()

    assert masker._reporting_data is not None
    assert masker._reporting_data["images"] is None

    masker.transform(surf_img_1d)

    assert isinstance(masker._reporting_data["images"], SurfaceImage)

    masker.generate_report()


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("empty_mask", [True, False])
def test_surface_masker_minimal_report_no_fit(
    surf_mask_1d, empty_mask, reports
):
    """Test minimal report generation with no fit."""
    mask = None if empty_mask else surf_mask_1d
    masker = SurfaceMasker(mask_img=mask, reports=reports)
    report = masker.generate_report()

    _check_html(report, reports_requested=reports, is_fit=False)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("empty_mask", [True, False])
def test_surface_masker_minimal_report_fit(
    surf_mask_1d, empty_mask, surf_img_1d, reports
):
    """Test minimal report generation with fit."""
    mask = None if empty_mask else surf_mask_1d
    masker = SurfaceMasker(mask_img=mask, reports=reports)
    masker.fit_transform(surf_img_1d)
    report = masker.generate_report()

    _check_html(report, reports_requested=reports)

    if reports:
        assert float(masker._report_content["coverage"]) > 0
        assert "The mask includes" in str(report)


def test_surface_maps_masker_generate_report_engine_error(
    matplotlib_pyplot,  # noqa: ARG001
    surf_maps_img,
    surf_img_2d,
):
    """Test error is raised when engine is not 'plotly' or 'matplotlib'."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.raises(
        ValueError,
        match="'engine' must be one of",
    ):
        masker.generate_report(engine="invalid", displayed_maps=2)


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="Test requires plotly not to be installed.",
)
def test_surface_maps_masker_generate_report_engine_no_plotly_warning(
    surf_maps_img, surf_img_2d
):
    """Test warning is raised when engine selected is plotly but it is not
    installed. Only run when plotly is not installed but matplotlib is.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.warns(match="Plotly is not installed"):
        masker.generate_report(engine="plotly", displayed_maps=2)
    # check if the engine is switched to matplotlib
    assert masker._report_content["engine"] == "matplotlib"


def test_surface_maps_masker_generate_report_before_transform_warn(
    matplotlib_pyplot,  # noqa: ARG001
    surf_maps_img,
):
    """Test warning is raised when generate_report is called before
    transform.
    """
    masker = SurfaceMapsMasker(surf_maps_img).fit()

    match = "SurfaceMapsMasker has not been transformed"
    with pytest.warns(match=match):
        masker.generate_report(displayed_maps=1)


def test_surface_maps_masker_generate_report_plotly_out_figure_type(
    plotly,  # noqa: ARG001
    matplotlib_pyplot,  # noqa: ARG001
    surf_maps_img,
    surf_img_2d,
):
    """Test that the report has a iframe tag when engine is plotly
    (default).
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    report = masker.generate_report(engine="plotly", displayed_maps=2)

    # read the html file and see if plotly figure is inserted
    # meaning it should have <iframe tag
    report_str = report.__str__()
    assert "<iframe" in report_str
    # and no <img tag
    assert "<img" not in report_str


def test_surface_maps_masker_generate_report_matplotlib_out_figure_type(
    matplotlib_pyplot,  # noqa: ARG001
    surf_maps_img,
    surf_img_2d,
):
    """Test that the report has a img tag when engine is matplotlib."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    report = masker.generate_report(engine="matplotlib", displayed_maps=2)

    # read the html file and see if matplotlib figure is inserted
    # meaning it should have <img tag
    report_str = report.__str__()
    assert "<img" in report_str
    # and no <iframe tag
    assert "<iframe" not in report_str
