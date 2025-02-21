from collections import Counter

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal

from nilearn._utils.data_gen import generate_random_img
from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn._utils.testing import on_windows_with_old_mpl_and_new_numpy
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
from nilearn.surface import SurfaceImage

# ruff: noqa: ARG001

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _check_html(html_view, reports_requested=True, is_fit=True):
    """Check the presence of some expected code in the html viewer."""
    if reports_requested and is_fit:
        assert "<th>Parameter</th>" in str(html_view)
    if "Surface" in str(html_view):
        assert "data:image/png;base64," in str(html_view)
    else:
        assert "data:image/svg+xml;base64," in str(html_view)
    assert html_view._repr_html_() == html_view.body


@pytest.fixture
def data_img_3d(shape_3d_default, affine_eye):
    """Return Dummy 3D data for testing."""
    data = np.zeros(shape_3d_default)
    data[3:-3, 3:-3, 3:-3] = 10
    return Nifti1Image(data, affine_eye)


@pytest.fixture
def mask(shape_3d_default, affine_eye):
    """Return Dummy mask for testing."""
    data = np.zeros(shape_3d_default, dtype="uint8")
    data[3:7, 3:7, 3:7] = 1
    return Nifti1Image(data, affine_eye)


@pytest.fixture
def niftimapsmasker_inputs(img_maps):
    return {"maps_img": img_maps}


@pytest.fixture
def labels(n_regions):
    return ["background"] + [f"region_{i}" for i in range(1, n_regions + 1)]


@pytest.fixture
def input_parameters(masker_class, mask, labels, img_labels, img_maps):
    if masker_class in (NiftiMasker, MultiNiftiMasker):
        return {"mask_img": mask}
    if masker_class in (NiftiLabelsMasker, MultiNiftiLabelsMasker):
        return {"labels_img": img_labels, "labels": labels}
    if masker_class in (NiftiMapsMasker, MultiNiftiMapsMasker):
        return {"maps_img": img_maps}
    if masker_class is NiftiSpheresMasker:
        return {"seeds": [(1, 1, 1)]}


@pytest.mark.parametrize(
    "masker_class",
    [NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker],
)
def test_warning_in_report_after_empty_fit(masker_class, input_parameters):
    """Tests that a warning is both given and written in the report \
       if no images were provided to fit.
    """
    masker = masker_class(**input_parameters)
    masker.fit()

    warn_message = f"No image provided to fit in {masker_class.__name__}."
    with pytest.warns(UserWarning, match=warn_message):
        html = masker.generate_report()
    assert warn_message in masker._report_content["warning_message"]
    _check_html(html)


@pytest.mark.parametrize("displayed_maps", ["foo", "1", {"foo": "bar"}])
def test_nifti_maps_masker_report_displayed_maps_errors(
    niftimapsmasker_inputs, displayed_maps
):
    """Tests that a TypeError is raised when the argument `displayed_maps` \
       of `generate_report()` is not valid.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    with pytest.raises(TypeError, match=("Parameter ``displayed_maps``")):
        masker.generate_report(displayed_maps)


@pytest.mark.parametrize("displayed_maps", [[2, 5, 10], [0, 66, 1, 260]])
def test_nifti_maps_masker_report_maps_number_errors(
    niftimapsmasker_inputs, displayed_maps
):
    """Tests that a ValueError is raised when the argument `displayed_maps` \
       contains invalid map numbers.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    with pytest.raises(
        ValueError, match="Report cannot display the following maps"
    ):
        masker.generate_report(displayed_maps)


@pytest.mark.parametrize("displayed_maps", [[2, 5, 6], np.array([0, 3, 4, 5])])
def test_nifti_maps_masker_report_list_and_arrays_maps_number(
    niftimapsmasker_inputs, displayed_maps, n_regions
):
    """Tests report generation for NiftiMapsMasker with displayed_maps \
       passed as a list of a Numpy arrays.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    html = masker.generate_report(displayed_maps)

    assert masker._report_content["number_of_maps"] == n_regions
    assert masker._report_content["displayed_maps"] == list(displayed_maps)
    msg = (
        "No image provided to fit in NiftiMapsMasker. "
        "Plotting only spatial maps for reporting."
    )
    assert masker._report_content["warning_message"] == msg
    assert html.body.count("<img") == len(displayed_maps)


@pytest.mark.parametrize("displayed_maps", [1, 6, 9, 12, "all"])
def test_nifti_maps_masker_report_integer_and_all_displayed_maps(
    niftimapsmasker_inputs, displayed_maps, n_regions
):
    """Tests NiftiMapsMasker reporting with no image provided to fit \
       and displayed_maps provided as an integer or as 'all'.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    expected_n_maps = (
        n_regions
        if displayed_maps == "all"
        else min(n_regions, displayed_maps)
    )
    if displayed_maps != "all" and displayed_maps > n_regions:
        with pytest.warns(UserWarning, match="masker only has 9 maps."):
            html = masker.generate_report(displayed_maps)
    else:
        html = masker.generate_report(displayed_maps)

    assert masker._report_content["number_of_maps"] == n_regions
    assert masker._report_content["displayed_maps"] == list(
        range(expected_n_maps)
    )
    msg = (
        "No image provided to fit in NiftiMapsMasker. "
        "Plotting only spatial maps for reporting."
    )
    assert masker._report_content["warning_message"] == msg
    assert html.body.count("<img") == expected_n_maps


def test_nifti_maps_masker_report_image_in_fit(
    niftimapsmasker_inputs, affine_eye, n_regions
):
    """Tests NiftiMapsMasker reporting with image provided to fit."""
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    image, _ = generate_random_img((13, 11, 12, 3), affine=affine_eye)
    masker.fit(image)
    html = masker.generate_report(2)

    assert masker._report_content["number_of_maps"] == n_regions

    assert html.body.count("<img") == 2


@pytest.mark.parametrize("displayed_spheres", ["foo", "1", {"foo": "bar"}])
def test_nifti_spheres_masker_report_displayed_spheres_errors(
    displayed_spheres,
):
    """Tests that a TypeError is raised when the argument `displayed_spheres` \
       of `generate_report()` is not valid.
    """
    masker = NiftiSpheresMasker(seeds=[(1, 1, 1)])
    masker.fit()
    with pytest.raises(TypeError, match=("Parameter ``displayed_spheres``")):
        masker.generate_report(displayed_spheres)


def test_nifti_spheres_masker_report_displayed_spheres_more_than_seeds():
    """Tests that a warning is raised when number of `displayed_spheres` \
       is greater than number of seeds.
    """
    displayed_spheres = 10
    seeds = [(1, 1, 1)]
    masker = NiftiSpheresMasker(seeds=seeds)
    masker.fit()
    with pytest.warns(UserWarning, match="masker only has 1 seeds."):
        masker.generate_report(displayed_spheres=displayed_spheres)


def test_nifti_spheres_masker_report_displayed_spheres_list():
    """Tests that spheres_to_be_displayed is set correctly."""
    displayed_spheres = [0, 1, 2]
    seeds = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    masker = NiftiSpheresMasker(seeds=seeds)
    masker.fit()
    masker.generate_report(displayed_spheres=displayed_spheres)
    assert masker._report_content["displayed_spheres"] == displayed_spheres


def test_nifti_spheres_masker_report_displayed_spheres_list_more_than_seeds():
    """Tests that a ValueError is raised when list of `displayed_spheres` \
       maximum is greater than number of seeds.
    """
    displayed_spheres = [1, 2, 3]
    seeds = [(1, 1, 1)]
    masker = NiftiSpheresMasker(seeds=seeds)
    masker.fit()
    with pytest.raises(ValueError, match="masker only has 1 seeds."):
        masker.generate_report(displayed_spheres=displayed_spheres)


def test_nifti_spheres_masker_report_1_sphere():
    """Check the report for sphere actually works for one sphere.

    See https://github.com/nilearn/nilearn/issues/4268
    """
    report = NiftiSpheresMasker([(1, 1, 1)]).fit().generate_report()

    empty_div = """
                    <img id="map1" class="pure-img" width="100%"
                        src="data:image/svg+xml;base64,D"
                        style="display:none;" alt="image"/>"""

    assert empty_div not in report.body


def test_nifti_labels_masker_report_incorrect_label_error(labels, img_labels):
    """Check that providing incorrect labels raises an error."""
    masker = NiftiLabelsMasker(img_labels, labels=labels[:-1])
    masker.fit()

    with pytest.raises(
        ValueError, match="Mismatch between the number of provided labels"
    ):
        masker.generate_report()


def test_nifti_labels_masker_report_no_image_for_fit(
    data_img_3d, n_regions, labels, img_labels
):
    """Check no contour in image when no image was provided to fit."""
    masker = NiftiLabelsMasker(img_labels, labels=labels)
    masker.fit()

    # No image was provided to fit, regions are plotted using
    # plot_roi such that no contour should be in the image
    display = masker._reporting()
    for d in ["x", "y", "z"]:
        assert len(display[0].axes[d].ax.collections) == 0

    masker.fit(data_img_3d)

    display = masker._reporting()
    for d in ["x", "y", "z"]:
        assert len(display[0].axes[d].ax.collections) > 0
        assert len(display[0].axes[d].ax.collections) <= n_regions


EXPECTED_COLUMNS = [
    "label value",
    "region name",
    "size (in mm^3)",
    "relative size (in %)",
]


def test_nifti_labels_masker_report(
    data_img_3d, mask, affine_eye, n_regions, labels, img_labels
):
    """Check content nifti label masker."""
    masker = NiftiLabelsMasker(img_labels, labels=labels, mask_img=mask)
    masker.fit(data_img_3d)
    report = masker.generate_report()

    assert masker._reporting_data is not None

    # Check that background label was left as default
    assert masker.background_label == 0
    assert masker._report_content["description"] == (
        "This reports shows the regions defined by the labels of the mask."
    )

    # Check that the number of regions is correct
    assert masker._report_content["number_of_regions"] == n_regions

    # Check that all expected columns are present with the right size
    for col in EXPECTED_COLUMNS:
        assert col in masker._report_content["summary"]
        assert len(masker._report_content["summary"][col]) == n_regions

    # Check that labels match
    assert masker._report_content["summary"]["region name"] == labels[1:]

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
            masker._report_content["summary"]["size (in mm^3)"][r - 1],
            expected_region_sizes[r]
            * np.abs(np.linalg.det(affine_eye[:3, :3])),
        )


def test_nifti_labels_masker_report_not_displayed(n_regions, img_labels):
    """Check that region labels are no displayed in the report \
       when they were not provided by the user.
    """
    masker = NiftiLabelsMasker(img_labels)
    masker.fit()
    masker.generate_report()

    for col in EXPECTED_COLUMNS:
        if col == "region name":
            assert col not in masker._report_content["summary"]
        else:
            assert col in masker._report_content["summary"]
            assert len(masker._report_content["summary"][col]) == n_regions


@pytest.mark.parametrize("masker_class", [NiftiLabelsMasker])
def test_nifti_labels_masker_report_cut_coords(
    masker_class, input_parameters, data_img_3d
):
    """Test cut coordinate are equal with and without passing data to fit."""
    masker = masker_class(**input_parameters, reports=True)
    # Get display without data
    masker.fit()
    display = masker._reporting()
    # Get display with data
    masker.fit(data_img_3d)
    display_data = masker._reporting()
    assert display[0].cut_coords == display_data[0].cut_coords


def test_4d_reports(mask, affine_eye):
    # Dummy 4D data
    data = np.zeros((10, 10, 10, 3), dtype="int32")
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, affine_eye)

    # test .fit method
    masker = NiftiMasker(mask_strategy="epi")
    masker.fit(data_img_4d)

    html = masker.generate_report()
    _check_html(html)

    # test .fit_transform method
    masker = NiftiMasker(mask_img=mask, standardize=True)
    masker.fit_transform(data_img_4d)

    html = masker.generate_report()
    _check_html(html)


def test_overlaid_report(img_fmri):
    """Check empty report generated before fit and with image after."""
    masker = NiftiMasker(
        mask_strategy="whole-brain-template",
        mask_args={"threshold": 0.0},
        target_affine=np.eye(3) * 3,
    )
    masker.fit(img_fmri)
    html = masker.generate_report()

    assert '<div class="overlay">' in str(html)


@pytest.mark.parametrize(
    "reports,expected", [(True, dict), (False, type(None))]
)
def test_multi_nifti_masker_generate_report_imgs(reports, expected, img_fmri):
    """Smoke test for generate_report method with image data."""
    masker = MultiNiftiMasker(reports=reports)
    masker.fit([img_fmri, img_fmri])
    assert isinstance(masker._reporting_data, expected)
    masker.generate_report()


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


def test_mask_img_generate_report(surf_img_1d, surf_mask_1d):
    """Smoke test generate report."""
    masker = SurfaceMasker(surf_mask_1d, reports=True).fit()

    assert masker._reporting_data is not None
    assert masker._reporting_data["images"] is None

    masker.transform(surf_img_1d)

    assert isinstance(masker._reporting_data["images"], SurfaceImage)

    masker.generate_report()


def test_mask_img_generate_no_report(surf_img_2d, surf_mask_1d):
    """Smoke test generate report."""
    masker = SurfaceMasker(surf_mask_1d, reports=False).fit()

    assert masker._reporting_data is None

    img = surf_img_2d(5)
    masker.transform(img)

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
    assert '<div class="image">' in str(report)
    if not reports:
        assert 'src="data:image/svg+xml;base64,"' in str(report)


def test_generate_report_engine_error(surf_maps_img, surf_img_2d):
    """Test error is raised when engine is not 'plotly' or 'matplotlib'."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.raises(
        ValueError,
        match="should be either 'matplotlib' or 'plotly'",
    ):
        masker.generate_report(engine="invalid")


@pytest.mark.skipif(
    is_plotly_installed() or not is_matplotlib_installed(),
    reason="Test requires plotly not to be installed.",
)
def test_generate_report_engine_no_plotly_warning(surf_maps_img, surf_img_2d):
    """Test warning is raised when engine selected is plotly but it is not
    installed. Only run when plotly is not installed but matplotlib is.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.warns(match="Plotly is not installed"):
        masker.generate_report(engine="plotly")
    # check if the engine is switched to matplotlib
    assert masker._report_content["engine"] == "matplotlib"


@pytest.mark.parametrize("displayed_maps", [4, [1, 3, 4, 5], "all", [1]])
def test_generate_report_displayed_maps_valid_inputs(
    surf_maps_img, surf_img_2d, displayed_maps
):
    """Test all valid inputs for displayed_maps."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    masker.generate_report(displayed_maps=displayed_maps)


@pytest.mark.parametrize("displayed_maps", [4.5, [8.4, 3], "invalid"])
def test_generate_report_displayed_maps_type_error(
    surf_maps_img, surf_img_2d, displayed_maps
):
    """Test error is raised when displayed_maps is not a list or int or
    np.ndarray or str(all).
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.raises(
        TypeError,
        match="should be either 'all' or an int, or a list/array of ints",
    ):
        masker.generate_report(displayed_maps=displayed_maps)


def test_generate_report_displayed_maps_more_than_regions_warn_int(
    surf_maps_img, surf_img_2d
):
    """Test error is raised when displayed_maps is int and is more than n
    regions.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.warns(
        UserWarning,
        match="But masker only has 6 maps",
    ):
        masker.generate_report(displayed_maps=10)
    # check if displayed_maps is switched to 6
    assert masker.displayed_maps == 6


def test_generate_report_displayed_maps_more_than_regions_warn_list(
    surf_maps_img, surf_img_2d
):
    """Test error is raised when displayed_maps is list has more elements than
    n regions.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    with pytest.raises(
        ValueError,
        match="Report cannot display the following maps",
    ):
        masker.generate_report(displayed_maps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_generate_report_before_transform_warn(surf_maps_img):
    """Test warning is raised when generate_report is called before
    transform.
    """
    masker = SurfaceMapsMasker(surf_maps_img).fit()
    with pytest.warns(match="SurfaceMapsMasker has not been transformed"):
        masker.generate_report()


@pytest.mark.skipif(
    on_windows_with_old_mpl_and_new_numpy(),
    reason="Old matplotlib not compatible with numpy 2.0 on windows.",
)
def test_generate_report_plotly_out_figure_type(
    plotly, surf_maps_img, surf_img_2d
):
    """Test that the report has a iframe tag when engine is plotly
    (default).
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    report = masker.generate_report(engine="plotly")

    # read the html file and see if plotly figure is inserted
    # meaning it should have <iframe tag
    report_str = report.__str__()
    assert "<iframe" in report_str
    # and no <img tag
    assert "<img" not in report_str


def test_generate_report_matplotlib_out_figure_type(
    surf_maps_img,
    surf_img_2d,
):
    """Test that the report has a img tag when engine is matplotlib."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker.fit_transform(surf_img_2d(10))
    report = masker.generate_report(engine="matplotlib")

    # read the html file and see if matplotlib figure is inserted
    # meaning it should have <img tag
    report_str = report.__str__()
    assert "<img" in report_str
    # and no <iframe tag
    assert "<iframe" not in report_str
