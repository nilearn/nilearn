"""
Test the second level model.
"""
from __future__ import with_statement

import os

import numpy as np
import pandas as pd
import warnings

from nibabel import (load,
                     Nifti1Image,
                     )
from nibabel.tmpdirs import InTemporaryDirectory
from nilearn.image import concat_imgs
from nose.tools import (assert_true,
                        assert_equal,
                        assert_raises,
                        )
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           )

from nistats.first_level_model import (FirstLevelModel,
                                       run_glm,
                                       )
from nistats.second_level_model import (SecondLevelModel,
                                        non_parametric_inference,
                                        )
from nistats._utils.testing import _write_fake_fmri_data

# This directory path
BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, 'functional.nii.gz')


def test_high_level_glm_with_paths():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        # asking for contrast before model fit gives error
        assert_raises(ValueError, model.compute_contrast, [])
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        c1 = np.eye(len(model.design_matrix_.columns))[0]
        z_image = model.compute_contrast(c1, output_type='z_score')
        assert_true(isinstance(z_image, Nifti1Image))
        assert_array_equal(z_image.affine, load(mask).affine)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del Y, FUNCFILE, func_img, model


def test_high_level_non_parametric_inference_with_paths():
    with InTemporaryDirectory():
        n_perm = 100
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        c1 = np.eye(len(X.columns))[0]
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     second_level_contrast=c1,
                                                     mask=mask, n_perm=n_perm)
        neg_log_pvals = neg_log_pvals_img.get_data()

        assert_true(isinstance(neg_log_pvals_img, Nifti1Image))
        assert_array_equal(neg_log_pvals_img.affine, load(mask).affine)

        assert_true(np.all(neg_log_pvals <= - np.log10(1.0 / (n_perm + 1))))
        assert_true(np.all(0 <= neg_log_pvals))
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory
        del X, Y, FUNCFILE, func_img, neg_log_pvals_img


def test_fmri_inputs():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        # prepare fake data
        p, q = 80, 10
        X = np.random.randn(p, q)
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        des = pd.DataFrame(np.ones((T, 1)), columns=['a'])
        des_fname = 'design.csv'
        des.to_csv(des_fname)

        # prepare correct input first level models
        flm = FirstLevelModel(subject_label='01').fit(FUNCFILE,
                                                      design_matrices=des)
        flms = [flm, flm, flm]
        # prepare correct input dataframe and lists
        shapes = ((7, 8, 9, 1),)
        _, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]

        dfcols = ['subject_label', 'map_name', 'effects_map_path']
        dfrows = [['01', 'a', FUNCFILE], ['02', 'a', FUNCFILE],
                  ['03', 'a', FUNCFILE]]
        niidf = pd.DataFrame(dfrows, columns=dfcols)
        niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
        niimg_4d = concat_imgs(niimgs)
        confounds = pd.DataFrame([['01', 1], ['02', 2], ['03', 3]],
                                 columns=['subject_label', 'conf1'])
        sdes = pd.DataFrame(X[:3, :3], columns=['intercept', 'b', 'c'])

        # smoke tests with correct input
        # First level models as input
        SecondLevelModel(mask_img=mask).fit(flms)
        SecondLevelModel().fit(flms)
        # Note : the following one creates a singular design matrix
        SecondLevelModel().fit(flms, confounds)
        SecondLevelModel().fit(flms, None, sdes)
        # dataframes as input
        SecondLevelModel().fit(niidf)
        SecondLevelModel().fit(niidf, confounds)
        SecondLevelModel().fit(niidf, confounds, sdes)
        SecondLevelModel().fit(niidf, None, sdes)
        # niimgs as input
        SecondLevelModel().fit(niimgs, None, sdes)
        # 4d niimg as input
        SecondLevelModel().fit(niimg_4d, None, sdes)

        # test wrong input errors
        # test first level model requirements
        assert_raises(ValueError, SecondLevelModel().fit, flm)
        assert_raises(ValueError, SecondLevelModel().fit, [flm])
        # test dataframe requirements
        assert_raises(ValueError, SecondLevelModel().fit,
                      niidf['subject_label'])
        # test niimgs requirements
        assert_raises(ValueError, SecondLevelModel().fit, niimgs)
        assert_raises(ValueError, SecondLevelModel().fit, niimgs + [[]],
                      confounds)
        # test first_level_conditions, confounds, and design
        assert_raises(ValueError, SecondLevelModel().fit, flms, ['', []])
        assert_raises(ValueError, SecondLevelModel().fit, flms, [])
        assert_raises(ValueError, SecondLevelModel().fit, flms,
                      confounds['conf1'])
        assert_raises(ValueError, SecondLevelModel().fit, flms, None, [])


def test_fmri_inputs_for_non_parametric_inference():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        # prepare fake data
        p, q = 80, 10
        X = np.random.randn(p, q)
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        des = pd.DataFrame(np.ones((T, 1)), columns=['a'])
        des_fname = 'design.csv'
        des.to_csv(des_fname)

        # prepare correct input first level models
        flm = FirstLevelModel(subject_label='01').fit(FUNCFILE,
                                                      design_matrices=des)
        # prepare correct input dataframe and lists
        shapes = ((7, 8, 9, 1),)
        _, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]

        dfcols = ['subject_label', 'map_name', 'effects_map_path']
        dfrows = [['01', 'a', FUNCFILE], ['02', 'a', FUNCFILE],
                  ['03', 'a', FUNCFILE]]
        niidf = pd.DataFrame(dfrows, columns=dfcols)
        niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
        niimg_4d = concat_imgs(niimgs)
        confounds = pd.DataFrame([['01', 1], ['02', 2], ['03', 3]],
                                 columns=['subject_label', 'conf1'])
        sdes = pd.DataFrame(X[:3, :3], columns=['intercept', 'b', 'c'])

        # test missing second-level contrast
        # niimgs as input
        assert_raises(ValueError, non_parametric_inference, niimgs, None, sdes)
        assert_raises(ValueError, non_parametric_inference, niimgs, confounds,
                      sdes)
        # 4d niimg as input
        assert_raises(ValueError, non_parametric_inference, niimg_4d, None,
                      sdes)

        # test wrong input errors
        # test first level model
        assert_raises(ValueError, non_parametric_inference, flm)
        # test list of less than two niimgs
        assert_raises(ValueError, non_parametric_inference, [FUNCFILE])
        # test dataframe
        assert_raises(ValueError, non_parametric_inference, niidf)
        # test niimgs requirements
        assert_raises(ValueError, non_parametric_inference, niimgs)
        assert_raises(ValueError, non_parametric_inference, niimgs + [[]],
                      confounds)
        assert_raises(ValueError, non_parametric_inference, [FUNCFILE])
        # test other objects
        assert_raises(ValueError, non_parametric_inference,
                      'random string object')
        del X, FUNCFILE, func_img


def _first_level_dataframe():
    names = ['con_01', 'con_01', 'con_01']
    subjects = ['01', '02', '03']
    maps = ['', '', '']
    dataframe = pd.DataFrame({'map_name': names,
                              'subject_label': subjects,
                              'effects_map_path': maps})
    return dataframe


def test_second_level_model_glm_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])

        model = model.fit(Y, design_matrix=X)
        model.compute_contrast()
        labels1 = model.labels_
        results1 = model.results_

        labels2, results2 = run_glm(
            model.masker_.transform(Y), X.values, 'ols')
        assert_almost_equal(labels1, labels2, decimal=1)
        assert_equal(len(results1), len(results2))
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_non_parametric_inference_permutation_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)

        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])

        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     mask=mask, n_perm=100)

        assert_equal(neg_log_pvals_img.get_data().shape, shapes[0][:3])
        del func_img, FUNCFILE, neg_log_pvals_img, X, Y


def test_second_level_model_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        # asking for contrast before model fit gives error
        assert_raises(ValueError, model.compute_contrast, 'intercept')
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        ncol = len(model.design_matrix_.columns)
        c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)
        # smoke test for different contrasts in fixed effects
        model.compute_contrast(c1)
        z_image = model.compute_contrast(c1, output_type='z_score')
        stat_image = model.compute_contrast(c1, output_type='stat')
        p_image = model.compute_contrast(c1, output_type='p_value')
        effect_image = model.compute_contrast(c1, output_type='effect_size')
        variance_image = \
            model.compute_contrast(c1, output_type='effect_variance')

        # Test output_type='all', and verify images are equivalent
        all_images = model.compute_contrast(c1, output_type='all')
        assert_array_equal(all_images['z_score'].get_data(),
                           z_image.get_data())
        assert_array_equal(all_images['stat'].get_data(),
                           stat_image.get_data())
        assert_array_equal(all_images['p_value'].get_data(),
                           p_image.get_data())
        assert_array_equal(all_images['effect_size'].get_data(),
                           effect_image.get_data())
        assert_array_equal(all_images['effect_variance'].get_data(),
                           variance_image.get_data())

        # formula should work (passing variable name directly)
        model.compute_contrast('intercept')
        # or simply pass nothing
        model.compute_contrast()
        # passing null contrast should give back a value error
        assert_raises(ValueError, model.compute_contrast, cnull)
        # passing wrong parameters
        assert_raises(ValueError, model.compute_contrast, [])
        assert_raises(ValueError, model.compute_contrast, c1, None, '')
        assert_raises(ValueError, model.compute_contrast, c1, None, [])
        assert_raises(ValueError, model.compute_contrast, c1, None, None, '')
        # check that passing no explicit contrast when the design
        # matrix has more than one columns raises an error
        X = pd.DataFrame(np.random.rand(4, 2), columns=['r1', 'r2'])
        model = model.fit(Y, design_matrix=X)
        assert_raises(ValueError, model.compute_contrast, None)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_non_parametric_inference_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # asking for contrast before model fit gives error
        assert_raises(ValueError, non_parametric_inference, None, None, None,
                      'intercept', mask)
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        # formula should work without second-level contrast
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     mask=mask, n_perm=100)

        ncol = len(X.columns)
        c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)
        # formula should work with second-level contrast
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     second_level_contrast=c1,
                                                     mask=mask, n_perm=100)
        # formula should work passing variable name directly
        neg_log_pvals_img = \
            non_parametric_inference(Y, design_matrix=X,
                                     second_level_contrast='intercept',
                                     mask=mask, n_perm=100)

        # passing null contrast should give back a value error
        assert_raises(ValueError, non_parametric_inference, Y, X, cnull,
                      'intercept', mask)
        # passing wrong parameters
        assert_raises(ValueError, non_parametric_inference, Y, X, [],
                      'intercept', mask)
        # check that passing no explicit contrast when the design
        # matrix has more than one columns raises an error
        X = pd.DataFrame(np.random.rand(4, 2), columns=['r1', 'r2'])
        assert_raises(ValueError, non_parametric_inference, Y, X, None)
        del func_img, FUNCFILE, neg_log_pvals_img, X, Y


def test_second_level_model_contrast_computation_with_memory_caching():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask, memory='nilearn_cache')
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        ncol = len(model.design_matrix_.columns)
        c1 = np.eye(ncol)[0, :]
        # test memory caching for compute_contrast
        model.compute_contrast(c1, output_type='z_score')
        # or simply pass nothing
        model.compute_contrast()
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_param_mask_deprecation_SecondLevelModel():
    """ Tests whether use of deprecated keyword parameter `mask`
    raises the correct warning & transfers its value to
    replacement parameter `mask_img` correctly.
    """
    deprecation_msg = (
        'The parameter "mask" will be removed in next release of Nistats. '
        'Please use the parameter "mask_img" instead.'
    )
    mask_filepath = '~/masks/mask_01.nii.gz'
    with warnings.catch_warnings(record=True) as raised_warnings:
        slm1 = SecondLevelModel(mask=mask_filepath)
        slm2 = SecondLevelModel(mask_img=mask_filepath)
        slm3 = SecondLevelModel(mask_filepath)
    assert slm1.mask_img == mask_filepath
    assert slm2.mask_img == mask_filepath
    assert slm3.mask_img == mask_filepath

    with assert_raises(AttributeError):
        slm1.mask == mask_filepath
    with assert_raises(AttributeError):
        slm2.mask == mask_filepath
    with assert_raises(AttributeError):
        slm3.mask == mask_filepath

    raised_param_deprecation_warnings = [
        raised_warning_ for raised_warning_
        in raised_warnings if
        str(raised_warning_.message).startswith('The parameter')
        ]

    assert len(raised_param_deprecation_warnings) == 1
    for param_warning_ in raised_param_deprecation_warnings:
        assert str(param_warning_.message) == deprecation_msg
        assert param_warning_.category is DeprecationWarning
