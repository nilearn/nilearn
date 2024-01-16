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
def data_img_3d(affine_eye):
    """Dummy 3D data for testing."""
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    return Nifti1Image(data, affine_eye)


@pytest.fixture
def mask(affine_eye):
    """Dummy mask for testing."""
    data = np.zeros((10, 10, 10), dtype="uint8")
    data[3:7, 3:7, 3:7] = 1
    return Nifti1Image(data, affine_eye)


@pytest.fixture
def niftimapsmasker_inputs(affine_eye):
    n_regions = 9
    shape = (8, 8, 8)
    label_img, _ = generate_maps(shape, n_regions=n_regions, affine=affine_eye)
    return {"maps_img": label_img}


@pytest.fixture
def input_parameters(masker_class, data_img_3d, affine_eye):
    n_regions = 9
    shape = (13, 11, 12)
    labels = ["background"]
    labels += [f"region_{i}" for i in range(1, n_regions + 1)]
    if masker_class in (NiftiMasker, MultiNiftiMasker):
        return {"mask_img": data_img_3d}
    if masker_class in (NiftiLabelsMasker, MultiNiftiLabelsMasker):
        labels_img = generate_labeled_regions(
            shape, n_regions=n_regions, affine=affine_eye
        )
        return {"labels_img": labels_img, "labels": labels}
    if masker_class in (NiftiMapsMasker, MultiNiftiMapsMasker):
        label_img, _ = generate_maps(
            shape, n_regions=n_regions, affine=affine_eye
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
    masker.fit()
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
        match=(
            "Report generation not enabled ! "
            "No visual outputs will be created."
        ),
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
    masker_class, input_parameters, data_img_3d, mask
):
    """Tests report generation after fitting on 3D data with mask_img."""
    input_parameters["mask_img"] = mask
    masker = masker_class(**input_parameters)
    masker.fit(data_img_3d)
    assert masker._report_content["warning_message"] is None
    html = masker.generate_report()
    _check_html(html)


@pytest.mark.parametrize(
    "masker_class",
    [NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker],
)
def test_warning_in_report_after_empty_fit(masker_class, input_parameters):
    """Tests that a warning is both given and written in the report if
    no images were provided to fit.
    """
    masker = masker_class(**input_parameters)
    assert masker._report_content["warning_message"] is None
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
    """Tests that a TypeError is raised when the argument `displayed_maps`
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
    """Tests that a ValueError is raised when the argument `displayed_maps`
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
    niftimapsmasker_inputs, displayed_maps
):
    """Tests report generation for NiftiMapsMasker with displayed_maps
    passed as a list of a Numpy arrays.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    html = masker.generate_report(displayed_maps)
    assert masker._report_content["report_id"] == 0
    assert masker._report_content["number_of_maps"] == 9
    assert masker._report_content["displayed_maps"] == list(displayed_maps)
    msg = (
        "No image provided to fit in NiftiMapsMasker. "
        "Plotting only spatial maps for reporting."
    )
    assert masker._report_content["warning_message"] == msg
    assert html.body.count("<img") == len(displayed_maps)


@pytest.mark.parametrize("displayed_maps", [1, 6, 9, 12, "all"])
def test_nifti_maps_masker_report_integer_and_all_displayed_maps(
    niftimapsmasker_inputs, displayed_maps
):
    """Tests NiftiMapsMasker reporting with no image provided to fit
    and displayed_maps provided as an integer or as 'all'.
    """
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    masker.fit()
    expected_n_maps = 9 if displayed_maps == "all" else min(9, displayed_maps)
    if displayed_maps != "all" and displayed_maps > 9:
        with pytest.warns(UserWarning, match="masker only has 9 maps."):
            html = masker.generate_report(displayed_maps)
    else:
        html = masker.generate_report(displayed_maps)
    assert masker._report_content["report_id"] == 0
    assert masker._report_content["number_of_maps"] == 9
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
    niftimapsmasker_inputs, affine_eye
):
    """Tests NiftiMapsMasker reporting with image provided to fit."""
    masker = NiftiMapsMasker(**niftimapsmasker_inputs)
    image, _ = generate_random_img((13, 11, 12, 3), affine=affine_eye)
    masker.fit(image)
    html = masker.generate_report(2)
    assert masker._report_content["report_id"] == 0
    assert masker._report_content["number_of_maps"] == 9
    assert masker._report_content["warning_message"] is None
    assert html.body.count("<img") == 2


@pytest.mark.parametrize("displayed_spheres", ["foo", "1", {"foo": "bar"}])
def test_nifti_spheres_masker_report_displayed_spheres_errors(
    displayed_spheres,
):
    """Tests that a TypeError is raised when the argument `displayed_spheres`
    of `generate_report()` is not valid.
    """
    masker = NiftiSpheresMasker(seeds=[(1, 1, 1)])
    masker.fit()
    with pytest.raises(TypeError, match=("Parameter ``displayed_spheres``")):
        masker.generate_report(displayed_spheres)


def test_nifti_spheres_masker_report_displayed_spheres_more_than_seeds():
    """Tests that a warning is raised when number of `displayed_spheres`
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
    """Tests that a ValueError is raised when list of `displayed_spheres`
    maximum is greater than number of seeds.
    """
    displayed_spheres = [1, 2, 3]
    seeds = [(1, 1, 1)]
    masker = NiftiSpheresMasker(seeds=seeds)
    masker.fit()
    with pytest.raises(ValueError, match="masker only has 1 seeds."):
        masker.generate_report(displayed_spheres=displayed_spheres)


def test_nifti_labels_masker_report(data_img_3d, mask, affine_eye):
    shape = (13, 11, 12)
    n_regions = 9
    labels = ["background"] + [f"region_{i}" for i in range(1, n_regions + 1)]
    EXPECTED_COLUMNS = [
        "label value",
        "region name",
        "size (in mm^3)",
        "relative size (in %)",
    ]
    labels_img = generate_labeled_regions(
        shape, affine=affine_eye, n_regions=n_regions
    )
    labels_img_floats = new_img_like(
        labels_img, get_data(labels_img).astype(float)
    )
    masker = NiftiLabelsMasker(labels_img_floats, labels=labels)
    masker.fit()
    masker.generate_report()

    # Check that providing incorrect labels raises an error
    masker = NiftiLabelsMasker(labels_img, labels=labels[:-1])
    masker.fit()
    with pytest.raises(
        ValueError, match="Mismatch between the number of provided labels"
    ):
        masker.generate_report()
    masker = NiftiLabelsMasker(labels_img, labels=labels)
    masker.fit()
    # Check that a warning is given when generating the report
    # since no image was provided to fit
    with pytest.warns(
        UserWarning, match="No image provided to fit in NiftiLabelsMasker"
    ):
        masker.generate_report()

    # No image was provided to fit, regions are plotted using
    # plot_roi such that no contour should be in the image
    display = masker._reporting()
    for d in ["x", "y", "z"]:
        assert len(display[0].axes[d].ax.collections) == 0

    masker = NiftiLabelsMasker(labels_img, labels=labels)
    masker.fit(data_img_3d)

    display = masker._reporting()
    for d in ["x", "y", "z"]:
        assert len(display[0].axes[d].ax.collections) > 0
        assert len(display[0].axes[d].ax.collections) <= n_regions

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
    for col in EXPECTED_COLUMNS:
        assert col in masker._report_content["summary"]
        assert len(masker._report_content["summary"][col]) == n_regions
    # Check that labels match
    assert masker._report_content["summary"]["region name"] == labels[1:]
    # Relative sizes of regions should sum to 100%
    assert_almost_equal(
        sum(masker._report_content["summary"]["relative size (in %)"]), 100
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


def test_overlaid_report(data_img_3d):
    pytest.importorskip("matplotlib")

    masker = NiftiMasker(target_affine=np.eye(3) * 8)
    html = masker.generate_report()
    assert "Please `fit` the object" in str(html)
    masker.fit(data_img_3d)
    html = masker.generate_report()
    assert '<div class="overlay">' in str(html)


@pytest.mark.parametrize(
    "reports,expected", [(True, dict), (False, type(None))]
)
def test_multi_nifti_masker_generate_report_imgs(
    reports, expected, affine_eye
):
    """Smoke test for generate_report method with image data."""
    shape = (9, 9, 5)
    imgs, _ = generate_fake_fmri(shape, affine=affine_eye, length=2)
    masker = MultiNiftiMasker(reports=reports)
    masker.fit([imgs, imgs])
    assert isinstance(masker._reporting_data, expected)
    masker.generate_report()


def test_multi_nifti_masker_generate_report_mask(affine_eye):
    """Smoke test for generate_report method with only mask."""
    shape = (9, 9, 5)
    mask = Nifti1Image(np.ones(shape), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
        # to test resampling lines without imgs
        target_affine=affine_eye,
        target_shape=shape,
    )
    masker.fit().generate_report()


def test_multi_nifti_masker_generate_report_imgs_and_mask(affine_eye):
    """Smoke test for generate_report method with images and mask."""
    shape = (9, 9, 5)
    imgs, _ = generate_fake_fmri(shape, affine=affine_eye, length=2)
    mask = Nifti1Image(np.ones(shape), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
        # to test resampling lines with imgs
        target_affine=affine_eye,
        target_shape=shape,
    )
    masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_masker_generate_report_warning(affine_eye):
    """Test calling generate report on multiple subjects raises warning."""
    shape = (9, 9, 9)
    imgs, _ = generate_fake_fmri(shape, affine=affine_eye, length=5)
    mask = Nifti1Image(np.ones(shape), affine_eye)
    masker = MultiNiftiMasker(
        mask_img=mask,
    )

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_labels_masker_report_warning(affine_eye):
    """Test calling generate report on multiple subjects raises warning."""
    shape = (13, 11, 12)
    n_regions = 9
    length = 3

    labels_img = generate_labeled_regions(
        shape, affine=affine_eye, n_regions=n_regions
    )
    imgs, _ = generate_fake_fmri(shape, affine=affine_eye, length=length)

    masker = MultiNiftiLabelsMasker(labels_img)

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()


def test_multi_nifti_maps_masker_report_warning(affine_eye):
    """Test calling generate report on multiple subjects raises warning."""
    shape = (13, 11, 12)
    n_regions = 9
    length = 3

    maps_img, _ = generate_maps(shape, n_regions, affine=affine_eye)
    imgs, _ = generate_fake_fmri(shape, affine=affine_eye, length=length)

    masker = MultiNiftiMapsMasker(maps_img)

    with pytest.warns(
        UserWarning, match="A list of 4D subject images were provided to fit. "
    ):
        masker.fit([imgs, imgs]).generate_report()
