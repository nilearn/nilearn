from collections import Counter

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal

from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
    generate_random_img,
)
from nilearn.image import get_data, new_img_like
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMasker,
)

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _check_html(html_view):
    """Check the presence of some expected code in the html viewer."""
    assert "Parameters" in str(html_view)
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
def n_regions():
    return 9


@pytest.fixture
def niftimapsmasker_inputs(n_regions, shape_3d_default, affine_eye):
    label_img, _ = generate_maps(
        shape_3d_default, n_regions=n_regions, affine=affine_eye
    )
    return {"maps_img": label_img}


@pytest.fixture
def labels(n_regions):
    return ["background"] + [f"region_{i}" for i in range(1, n_regions + 1)]


@pytest.fixture
def labels_img(shape_3d_default, affine_eye, n_regions):
    return generate_labeled_regions(
        shape_3d_default, affine=affine_eye, n_regions=n_regions
    )


@pytest.fixture
def input_parameters(
    n_regions,
    shape_3d_default,
    masker_class,
    mask,
    affine_eye,
    labels,
    labels_img,
):
    if masker_class in (NiftiMasker, MultiNiftiMasker):
        return {"mask_img": mask}
    if masker_class in (NiftiLabelsMasker, MultiNiftiLabelsMasker):
        return {"labels_img": labels_img, "labels": labels}
    if masker_class in (NiftiMapsMasker, MultiNiftiMapsMasker):
        label_img, _ = generate_maps(
            shape_3d_default, n_regions=n_regions, affine=affine_eye
        )
        return {"maps_img": label_img}
    if masker_class is NiftiSpheresMasker:
        return {"seeds": [(1, 1, 1)]}


@pytest.mark.parametrize(
    "masker_class",
    [
        NiftiMasker,
        MultiNiftiMasker,
        NiftiLabelsMasker,
        MultiNiftiLabelsMasker,
        NiftiMapsMasker,
        MultiNiftiMapsMasker,
        NiftiSpheresMasker,
    ],
)
def test_report_empty_fit(masker_class, input_parameters):
    """Test minimal report generation."""
    masker = masker_class(**input_parameters)
    masker = masker.fit()
    _check_html(masker.generate_report())


@pytest.mark.parametrize(
    "masker_class",
    [
        NiftiMasker,
        MultiNiftiMasker,
        NiftiLabelsMasker,
        MultiNiftiLabelsMasker,
        NiftiMapsMasker,
        MultiNiftiMapsMasker,
        NiftiSpheresMasker,
    ],
)
def test_empty_report(masker_class, input_parameters):
    """Test with reports set to False."""
    masker = masker_class(**input_parameters, reports=False)
    masker.fit()
    assert masker._reporting_data is None
    assert masker._reporting() == [None]
    with pytest.warns(
        UserWarning,
        match=("No visual outputs created."),
    ):
        masker.generate_report()


@pytest.mark.parametrize("masker_class", [NiftiMasker, NiftiLabelsMasker])
def test_reports_after_fit_3d_data(
    masker_class, input_parameters, data_img_3d
):
    """Tests report generation after fitting on 3D data."""
    masker = masker_class(**input_parameters)
    masker.fit(data_img_3d)
    html = masker.generate_report()
    _check_html(html)


@pytest.mark.parametrize("masker_class", [NiftiMasker, NiftiLabelsMasker])
def test_reports_after_fit_3d_data_with_mask(
    masker_class, input_parameters, img_3d_rand_eye, mask
):
    """Tests report generation after fitting on 3D data with mask_img."""
    input_parameters["mask_img"] = mask
    masker = masker_class(**input_parameters)
    masker.fit(img_3d_rand_eye)
    assert masker._report_content["warning_message"] is None
    html = masker.generate_report()
    _check_html(html)


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
    assert masker._report_content["warning_message"] is None
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
    assert masker._report_content["report_id"] == 0
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
    assert masker._report_content["report_id"] == 0
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
    assert masker._report_content["report_id"] == 0
    assert masker._report_content["number_of_maps"] == n_regions
    assert masker._report_content["warning_message"] is None
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

    assert masker._report_content["report_id"] == 0


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


def test_nifti_labels_masker_report_smoke_test(labels, labels_img):
    """Smoke test."""
    labels_img_floats = new_img_like(
        labels_img, get_data(labels_img).astype(float)
    )
    masker = NiftiLabelsMasker(labels_img_floats, labels=labels)
    masker.fit()
    masker.generate_report()


def test_nifti_labels_masker_report_incorrect_label_error(labels, labels_img):
    """Check that providing incorrect labels raises an error."""
    masker = NiftiLabelsMasker(labels_img, labels=labels[:-1])
    masker.fit()

    with pytest.raises(
        ValueError, match="Mismatch between the number of provided labels"
    ):
        masker.generate_report()


def test_nifti_labels_masker_report_warning_no_img_fit(labels, labels_img):
    """Check warning thrown when no image was provided to fit."""
    masker = NiftiLabelsMasker(labels_img, labels=labels)
    masker.fit()
    with pytest.warns(
        UserWarning, match="No image provided to fit in NiftiLabelsMasker"
    ):
        masker.generate_report()


def test_nifti_labels_masker_report_no_image_for_fit(
    data_img_3d, n_regions, labels, labels_img
):
    """Check no contour in image when no image was provided to fit."""
    masker = NiftiLabelsMasker(labels_img, labels=labels)
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


def test_nifti_labels_masker_report(
    data_img_3d, mask, affine_eye, n_regions, labels, labels_img
):
    """Check warning thrown when no image was provided to fit."""
    masker = NiftiLabelsMasker(labels_img, labels=labels, mask_img=mask)
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
    EXPECTED_COLUMNS = [
        "label value",
        "region name",
        "size (in mm^3)",
        "relative size (in %)",
    ]
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
    expected_region_sizes = Counter(get_data(labels_img).ravel())
    for r in range(1, n_regions + 1):
        assert_almost_equal(
            masker._report_content["summary"]["size (in mm^3)"][r - 1],
            expected_region_sizes[r]
            * np.abs(np.linalg.det(affine_eye[:3, :3])),
        )

    # Check that region labels are no displayed in the report
    # when they were not provided by the user.
    masker = NiftiLabelsMasker(labels_img)
    masker.fit()
    report = masker.generate_report()

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
    assert masker._report_content["warning_message"] is None
    html = masker.generate_report()
    _check_html(html)

    # test .fit_transform method
    masker = NiftiMasker(mask_img=mask, standardize=True)
    masker.fit_transform(data_img_4d)
    assert masker._report_content["warning_message"] is None
    html = masker.generate_report()
    _check_html(html)


def test_overlaid_report(img_3d_mni):
    pytest.importorskip("matplotlib")

    masker = NiftiMasker(target_affine=np.eye(3) * 8)
    html = masker.generate_report()
    assert "Make sure to run `fit`" in str(html)
    masker.fit(img_3d_mni)
    html = masker.generate_report()
    assert '<div class="overlay">' in str(html)


@pytest.mark.parametrize(
    "reports,expected", [(True, dict), (False, type(None))]
)
def test_multi_nifti_masker_generate_report_imgs(
    reports, expected, affine_eye, shape_3d_default
):
    """Smoke test for generate_report method with image data."""
    imgs, _ = generate_fake_fmri(shape_3d_default, affine=affine_eye, length=2)
    masker = MultiNiftiMasker(reports=reports)
    masker.fit([imgs, imgs])
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
    shape_3d_default, affine_eye
):
    """Smoke test for generate_report method with images and mask."""
    imgs, _ = generate_fake_fmri(shape_3d_default, affine=affine_eye, length=2)
    mask = Nifti1Image(np.ones(shape_3d_default), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
        # to test resampling lines with imgs
        target_affine=affine_eye,
        target_shape=shape_3d_default,
    )
    masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_masker_generate_report_warning(
    shape_3d_default, affine_eye
):
    """Test calling generate report on multiple subjects raises warning."""
    imgs, _ = generate_fake_fmri(shape_3d_default, affine=affine_eye, length=5)
    mask = Nifti1Image(np.ones(shape_3d_default), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
    )

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_labels_masker_report_warning(
    shape_3d_default, affine_eye, labels_img
):
    """Test calling generate report on multiple subjects raises warning."""
    length = 3

    imgs, _ = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiLabelsMasker(labels_img)

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_maps_masker_report_warning(
    shape_3d_default, affine_eye, n_regions
):
    """Test calling generate report on multiple subjects raises warning."""
    length = 3

    maps_img, _ = generate_maps(shape_3d_default, n_regions, affine=affine_eye)
    imgs, _ = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(maps_img)

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("empty_mask", [True, False])
def test_surface_masker_minimal_report_no_fit(
    surf_mask_1d, empty_mask, reports
):
    """Test minimal report generation with no fit."""
    mask = None if empty_mask else surf_mask_1d
    masker = SurfaceMasker(mask_img=mask, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" in str(report)


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

    _check_html(report)
    assert '<div class="image">' in str(report)
    if not reports:
        assert "Make sure to run `fit`" in str(report)
        assert 'src="data:image/svg+xml;base64,"' in str(report)


def test_surface_masker_report_no_report(surf_img_1d):
    """Check content of no report."""
    masker = SurfaceMasker(reports=False)
    masker.fit_transform(surf_img_1d)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)


@pytest.mark.parametrize("reports", [True, False])
@pytest.mark.parametrize("label_names", [None, ["region 1", "region 2"]])
def test_surface_label_masker_report_unfitted(
    surf_label_img, label_names, reports
):
    masker = SurfaceLabelsMasker(surf_label_img, label_names, reports=reports)
    report = masker.generate_report()

    _check_html(report)
    assert "Make sure to run `fit`" in str(report)


def test_surface_label_masker_report(surf_label_img, surf_img_1d, tmp_path):
    """Test that a report can be generated and saved as html."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()
    masker.transform(surf_img_1d)
    report = masker.generate_report()
    report.save_as_html(tmp_path / "surface_label_masker.html")


def test_surface_label_masker_report_no_report(surf_label_img):
    """Check content of no report."""
    masker = SurfaceLabelsMasker(surf_label_img, reports=False)
    report = masker.generate_report()

    _check_html(report)
    assert "No visual outputs created." in str(report)
    assert "Empty Report" in str(report)
