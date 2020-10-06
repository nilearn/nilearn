# -*- coding: utf-8 -*-

import warnings

import nibabel as nib
import numpy as np
import pytest
from nibabel import load
from nibabel.tmpdirs import InTemporaryDirectory

import pandas as pd

from nilearn._utils.data_gen import write_fake_fmri_data_and_design
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix)
from nilearn.glm.first_level import FirstLevelModel
from nilearn.reporting import glm_reporter as glmr
from nilearn.glm.second_level import SecondLevelModel

try:
    import matplotlib as mpl  # noqa: F841
except ImportError:
    not_have_mpl = True
else:
    not_have_mpl = False


@pytest.mark.skipif(not_have_mpl,
                    reason='Matplotlib not installed; required for this test')
@pytest.mark.parametrize("use_method", [True, False])
def test_flm_reporting(use_method):
    with InTemporaryDirectory():
        shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
        mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(shapes, rk)
        flm = FirstLevelModel(mask_img=mask).fit(
            fmri_data, design_matrices=design_matrices)
        contrast = np.eye(3)[1]
        if use_method:
            report_flm = flm.generate_report(
                contrast, plot_type='glass', height_control=None,
                min_distance=15, alpha=0.001, threshold=2.78)
        else:
            report_flm = glmr.make_glm_report(flm, contrast, plot_type='glass',
                                              height_control=None,
                                              min_distance=15,
                                              alpha=0.001, threshold=2.78,
            )
        '''
        catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
        Python2's limited unicode support causes  HTMLDocument.get_iframe() to
        mishandle certain unicode characters, like the greek alpha symbol
        and raises this error.
        Calling HTMLDocument.get_iframe() here causes the tests
        to fail on Python2, alerting us if such a situation arises
        due to future modifications.
        '''
        report_iframe = report_flm.get_iframe()
        # So flake8 doesn't complain about not using variable (F841)
        report_iframe
        del mask, flm, fmri_data


@pytest.mark.skipif(not_have_mpl,
                    reason='Matplotlib not installed; required for this test')
@pytest.mark.parametrize("use_method", [True, False])
def test_slm_reporting(use_method):
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        model = SecondLevelModel()
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        c1 = np.eye(len(model.design_matrix_.columns))[0]
        if use_method:
            report_slm = glmr.make_glm_report(model, c1)
        else:
            report_slm = model.generate_report(c1)
        # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
        report_iframe = report_slm.get_iframe()
        # So flake8 doesn't complain about not using variable (F841)
        report_iframe
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del Y, FUNCFILE, func_img, model


def test_check_report_dims():
    test_input = (1200, 'a')
    expected_output = (1600, 800)
    expected_warning_text = ('Report size has invalid values. '
                             'Using default 1600x800')
    with warnings.catch_warnings(record=True) as raised_warnings:
        actual_output = glmr._check_report_dims(test_input)
    raised_warnings_texts = [str(warning_.message) for warning_ in
                             raised_warnings]
    assert actual_output == expected_output
    assert expected_warning_text in raised_warnings_texts


def test_coerce_to_dict_with_string():
    test_input = 'StopSuccess - Go'
    expected_output = {'StopSuccess - Go': 'StopSuccess - Go'}
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_strings():
    test_input = ['contrast_name_0', 'contrast_name_1']
    expected_output = {'contrast_name_0': 'contrast_name_0',
                       'contrast_name_1': 'contrast_name_1',
                       }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_dict():
    test_input = {'contrast_0': [0, 0, 1],
                  'contrast_1': [0, 1, 1],
                  }
    expected_output = {'contrast_0': [0, 0, 1],
                       'contrast_1': [0, 1, 1],
                       }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_lists():
    test_input = [[0, 0, 1], [0, 1, 0]]
    expected_output = {'[0, 0, 1]': [0, 0, 1],
                       '[0, 1, 0]': [0, 1, 0],
                       }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output == expected_output


def test_coerce_to_dict_with_list_of_arrays():
    test_input = [np.array([0, 0, 1]), np.array([0, 1, 0])]
    expected_output = {'[0 0 1]': np.array([0, 0, 1]),
                       '[0 1 0]': np.array([0, 1, 0]),
                       }
    actual_output = glmr._coerce_to_dict(test_input)
    assert actual_output.keys() == expected_output.keys()
    for key in actual_output:
        assert np.array_equal(actual_output[key],
                              expected_output[key],
                              )


def test_coerce_to_dict_with_list_of_ints():
    test_input = [1, 0, 1]
    expected_output = {'[1, 0, 1]': [1, 0, 1]}
    actual_output = glmr._coerce_to_dict(test_input)
    assert np.array_equal(actual_output['[1, 0, 1]'],
                          expected_output['[1, 0, 1]'],
                          )


def test_coerce_to_dict_with_array_of_ints():
    test_input = np.array([1, 0, 1])
    expected_output = {'[1 0 1]': np.array([1, 0, 1])}
    actual_output = glmr._coerce_to_dict(test_input)
    assert expected_output.keys() == actual_output.keys()
    assert np.array_equal(actual_output['[1 0 1]'],
                          expected_output['[1 0 1]'],
                          )


def test_make_headings_with_contrasts_title_none():
    model = SecondLevelModel()
    test_input = ({'contrast_0': [0, 0, 1],
                   'contrast_1': [0, 1, 1],
                   },
                  None,
                  model
                  )
    expected_output = (
        'Report: Second Level Model for contrast_0, contrast_1',
        'Statistical Report for contrast_0, contrast_1',
        'Second Level Model',
    )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_headings_with_contrasts_title_custom():
    model = SecondLevelModel()
    test_input = ({'contrast_0': [0, 0, 1],
                   'contrast_1': [0, 1, 1],
                   },
                  'Custom Title for report',
                  model,
                  )
    expected_output = ('Custom Title for report',
                       'Custom Title for report',
                       'Second Level Model',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_headings_with_contrasts_none_title_custom():
    model = FirstLevelModel()
    test_input = (None,
                  'Custom Title for report',
                  model,
                  )
    expected_output = ('Custom Title for report',
                       'Custom Title for report',
                       'First Level Model',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def _generate_img():
    mni_affine = np.array([[-2., 0., 0., 90.],
                           [0., 2., 0., -126.],
                           [0., 0., 2., -72.],
                           [0., 0., 0., 1.]])

    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.rand(7, 7, 3)
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]

    return nib.Nifti1Image(data_positive, mni_affine)


def test_stat_map_to_svg_slice_z():
    with InTemporaryDirectory():
        img = _generate_img()
        table_details = pd.DataFrame.from_dict({'junk': 0}, orient='index')
        stat_map_html_code = glmr._stat_map_to_svg(  # noqa: F841
            stat_img=img,
            bg_img=None,
            display_mode='ortho',
            plot_type='slice',
            table_details=table_details,
        )


def test_stat_map_to_svg_glass_z():
    with InTemporaryDirectory():
        img = _generate_img()
        table_details = pd.DataFrame.from_dict({'junk': 0}, orient='index')
        stat_map_html_code = glmr._stat_map_to_svg(  # noqa: F841
            stat_img=img,
            bg_img=None,
            display_mode='z',
            plot_type='glass',
            table_details=table_details,
        )


def test_stat_map_to_svg_invalid_plot_type():
    with InTemporaryDirectory():
        img = _generate_img()
        expected_error = ValueError(
            'Invalid plot type provided. Acceptable options are'
            "'slice' or 'glass'.")
        try:
            stat_map_html_code = glmr._stat_map_to_svg(  # noqa: F841
                stat_img=img,
                bg_img=None,
                display_mode='z',
                plot_type='junk',
                table_details={'junk': 0
                               },
            )
        except ValueError as raised_exception:
            assert str(raised_exception) == str(expected_error)


def _make_dummy_contrasts_dmtx():
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(frame_times,
                                          drift_model='polynomial',
                                          drift_order=3,
                                          )
    contrast = {'test': np.ones(4)}
    return contrast, dmtx


def test_plot_contrasts():
    contrast, dmtx = _make_dummy_contrasts_dmtx()
    contrast_plots = glmr._plot_contrasts(contrast,  # noqa: F841
                                          [dmtx],
                                          )
