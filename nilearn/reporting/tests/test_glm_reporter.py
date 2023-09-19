import warnings

import numpy as np
import pandas as pd
import pytest
from nibabel import load

from nilearn._utils.data_gen import write_fake_fmri_data_and_design
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import glm_reporter as glmr

try:
    import matplotlib as mpl  # noqa: F401
except ImportError:
    not_have_mpl = True
else:
    not_have_mpl = False


@pytest.mark.skipif(
    not_have_mpl, reason="Matplotlib not installed; required for this test"
)
@pytest.mark.parametrize("use_method", [True, False])
def test_flm_reporting(tmp_path, use_method):
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    flm = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )
    contrast = np.eye(3)[1]
    if use_method:
        report_flm = flm.generate_report(
            contrast,
            plot_type="glass",
            height_control=None,
            min_distance=15,
            alpha=0.001,
            threshold=2.78,
        )
    else:
        report_flm = glmr.make_glm_report(
            flm,
            contrast,
            plot_type="glass",
            height_control=None,
            min_distance=15,
            alpha=0.001,
            threshold=2.78,
        )
    """
    catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    in case certain unicode characters are mishandled,
    like the greek alpha symbol.
    """
    report_flm.get_iframe()


@pytest.mark.skipif(
    not_have_mpl, reason="Matplotlib not installed; required for this test"
)
@pytest.mark.parametrize("use_method", [True, False])
def test_slm_reporting(tmp_path, use_method):
    shapes = ((7, 8, 9, 1),)
    _, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)
    model = SecondLevelModel()
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    c1 = np.eye(len(model.design_matrix_.columns))[0]
    if use_method:
        report_slm = glmr.make_glm_report(model, c1)
    else:
        report_slm = model.generate_report(c1)
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    report_slm.get_iframe()


def test_check_report_dims():
    test_input = (1200, "a")
    expected_output = (1600, 800)
    expected_warning_text = (
        "Report size has invalid values. Using default 1600x800"
    )
    with warnings.catch_warnings(record=True) as raised_warnings:
        actual_output = glmr._check_report_dims(test_input)
    raised_warnings_texts = [
        str(warning_.message) for warning_ in raised_warnings
    ]
    assert actual_output == expected_output
    assert expected_warning_text in raised_warnings_texts


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
    """
    Checks that using NiftiMasker when instantiating
    FirstLevelModel doesn't raise Error when calling
    generate_report().
    """
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
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
        alpha=0.001,
        threshold=2.78,
    )

    report_flm.get_iframe()
