import numpy as np
import pandas as pd
import pytest
from nibabel import load

from nilearn._utils.data_gen import (
    basic_paradigm,
    write_fake_fmri_data_and_design,
)
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import glm_reporter as glmr
from nilearn.reporting import make_glm_report


@pytest.fixture()
def flm(tmp_path):
    """Generate first level model."""
    shapes, rk = ((7, 7, 7, 5),), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    return FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )


@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_flm_reporting(flm, height_control):
    """Smoke test for first level model reporting."""
    contrast = np.eye(3)[1]
    report_flm = glmr.make_glm_report(
        flm,
        contrast,
        plot_type="glass",
        height_control=height_control,
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report_flm.get_iframe()


def test_flm_reporting_method(flm):
    """Smoke test for the first level generate method."""
    contrast = np.eye(3)[1]
    flm.generate_report(
        contrast,
        plot_type="glass",
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )


@pytest.fixture()
def slm(tmp_path):
    """Generate a fitted second level model."""
    shapes = ((7, 7, 7, 1),)
    _, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)
    model = SecondLevelModel()
    Y = [func_img] * 2
    X = pd.DataFrame([[1]] * 2, columns=["intercept"])
    return model.fit(Y, design_matrix=X)


@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_slm_reporting_method(slm, height_control):
    """Smoke test for the second level reporting."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    report_slm = glmr.make_glm_report(
        slm, c1, height_control=height_control, threshold=2, alpha=0.01
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    report_slm.get_iframe()


def test_slm_reporting(slm):
    """Smoke test for the second level model generate method."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    slm.generate_report(c1, threshold=2, alpha=0.01)


def test_check_report_dims():
    test_input = (1200, "a")
    expected_output = (1600, 800)
    expected_warning_text = (
        "Report size has invalid values. Using default 1600x800"
    )
    with pytest.warns(UserWarning, match=expected_warning_text):
        actual_output = glmr._check_report_dims(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_string():
    test_input = "StopSuccess - Go"
    expected_output = {"StopSuccess - Go": "StopSuccess - Go"}
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_strings():
    test_input = ["contrast_name_0", "contrast_name_1"]
    expected_output = {
        "contrast_name_0": "contrast_name_0",
        "contrast_name_1": "contrast_name_1",
    }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_dict():
    test_input = {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]}
    expected_output = {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]}
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_lists():
    test_input = [[0, 0, 1], [0, 1, 0]]
    expected_output = {"[0, 0, 1]": [0, 0, 1], "[0, 1, 0]": [0, 1, 0]}
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_arrays():
    test_input = [np.array([0, 0, 1]), np.array([0, 1, 0])]
    expected_output = {
        "[0 0 1]": np.array([0, 0, 1]),
        "[0 1 0]": np.array([0, 1, 0]),
    }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output.keys() == expected_output.keys()
    for key in actual_output:
        assert np.array_equal(actual_output[key], expected_output[key])


def test_coerce_to_dict_with_list_of_ints():
    test_input = [1, 0, 1]
    expected_output = {"[1, 0, 1]": [1, 0, 1]}
    actual_output = glmr._coerce_to_dict(test_input)
    assert np.array_equal(
        actual_output["[1, 0, 1]"], expected_output["[1, 0, 1]"]
    )


def test_coerce_to_dict_with_array_of_ints():
    test_input = np.array([1, 0, 1])
    expected_output = {"[1 0 1]": np.array([1, 0, 1])}
    actual_output = glmr._coerce_to_dict(test_input)
    assert expected_output.keys() == actual_output.keys()
    assert np.array_equal(actual_output["[1 0 1]"], expected_output["[1 0 1]"])


def test_make_headings_with_contrasts_title_none():
    model = SecondLevelModel()
    test_input = (
        {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
        None,
        model,
    )
    expected_output = (
        "Report: Second Level Model for contrast_0, contrast_1",
        "Statistical Report for contrast_0, contrast_1",
        "Second Level Model",
    )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_headings_with_contrasts_title_custom():
    model = SecondLevelModel()
    test_input = (
        {"contrast_0": [0, 0, 1], "contrast_1": [0, 1, 1]},
        "Custom Title for report",
        model,
    )
    expected_output = (
        "Custom Title for report",
        "Custom Title for report",
        "Second Level Model",
    )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_headings_with_contrasts_none_title_custom():
    model = FirstLevelModel()
    test_input = (None, "Custom Title for report", model)
    expected_output = (
        "Custom Title for report",
        "Custom Title for report",
        "First Level Model",
    )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_slice_z(img_3d_mni, cut_coords):
    table_details = pd.DataFrame.from_dict({"junk": 0}, orient="index")
    glmr._stat_map_to_svg(
        stat_img=img_3d_mni,
        bg_img=None,
        cut_coords=cut_coords,
        display_mode="ortho",
        plot_type="slice",
        table_details=table_details,
        threshold=2.76,
    )


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_glass_z(img_3d_mni, cut_coords):
    table_details = pd.DataFrame.from_dict({"junk": 0}, orient="index")
    glmr._stat_map_to_svg(
        stat_img=img_3d_mni,
        bg_img=None,
        cut_coords=cut_coords,
        display_mode="z",
        plot_type="glass",
        table_details=table_details,
        threshold=2.76,
    )


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_invalid_plot_type(img_3d_mni, cut_coords):
    expected_error = (
        "Invalid plot type provided. "
        "Acceptable options are 'slice' or 'glass'."
    )
    with pytest.raises(ValueError, match=expected_error):
        glmr._stat_map_to_svg(
            stat_img=img_3d_mni,
            bg_img=None,
            cut_coords=cut_coords,
            display_mode="z",
            plot_type="junk",
            table_details={"junk": 0},
            threshold=2.76,
        )


def _make_dummy_contrasts_dmtx():
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times,
        drift_model="polynomial",
        drift_order=3,
    )
    contrast = {"test": np.ones(4)}
    return contrast, dmtx


def test_plot_contrasts():
    contrast, dmtx = _make_dummy_contrasts_dmtx()
    glmr._plot_contrasts(
        contrast,
        [dmtx],
    )


def test_masking_first_level_model(tmp_path):
    """Check that using NiftiMasker when instantiating FirstLevelModel \
       doesn't raise Error when calling generate_report().
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    masker = NiftiMasker(mask_img=mask)
    masker.fit(fmri_data)
    flm = FirstLevelModel(mask_img=masker).fit(
        fmri_data, design_matrices=design_matrices
    )
    contrast = np.eye(3)[1]

    report_flm = flm.generate_report(
        contrast,
        plot_type="glass",
        height_control=None,
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )

    report_flm.get_iframe()


# -----------------------surface tests--------------------------------------- #


def test_flm_generate_report_error_with_surface_data(
    surf_mask_1d, surf_img_2d
):
    """Raise NotImplementedError when generate report is called on surface."""
    model = FirstLevelModel(mask_img=surf_mask_1d, t_r=2.0)
    events = basic_paradigm()
    model.fit(surf_img_2d(9), events=events)

    with pytest.raises(NotImplementedError):
        model.generate_report("c0")

    with pytest.raises(NotImplementedError):
        make_glm_report(model, "c0")
