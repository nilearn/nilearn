# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the second level model.
"""
from __future__ import with_statement

import os

import numpy as np

from nibabel import load, Nifti1Image, save

from nistats.first_level_model import FirstLevelModel, run_glm
from nistats.second_level_model import SecondLevelModel
from nistats.design_matrix import create_second_level_design

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import (assert_almost_equal, assert_array_equal)
from nibabel.tmpdirs import InTemporaryDirectory
import pandas as pd


# This directory path
BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, 'functional.nii.gz')


def write_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    mask_file, fmri_files, design_files = 'mask.nii', [], []
    for i, shape in enumerate(shapes):
        fmri_files.append('fmri_run%d.nii' % i)
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        save(Nifti1Image(data, affine), fmri_files[-1])
        design_files.append('dmtx_%d.csv' % i)
        pd.DataFrame(np.random.randn(shape[3], rk),
                     columns=['', '', '']).to_csv(design_files[-1])
    save(Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                     affine), mask_file)
    return mask_file, fmri_files, design_files


def test_high_level_glm_with_paths():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # ols case
        model = SecondLevelModel(mask=mask)
        # asking for contrast before model fit gives error
        assert_raises(ValueError, model.compute_contrast, [])
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4)
        model = model.fit(Y, design_matrix=X)
        c1 = np.eye(len(model.design_matrix_.columns))[0]
        z_image = model.compute_contrast(c1, None, 'z_score')
        assert_true(isinstance(z_image, Nifti1Image))
        assert_array_equal(z_image.get_affine(), load(mask).get_affine())
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory
        del z_image, FUNCFILE, func_img, model


def test_fmri_inputs():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        # prepare fake data
        p, q = 80, 10
        X = np.random.randn(p, q)
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        des = pd.DataFrame(np.ones((T, 1)), columns=['a'])
        des_fname = 'design.csv'
        des.to_csv(des_fname)

        # prepare correct input first level models
        flm = FirstLevelModel(subject_id='1').fit(FUNCFILE, design_matrices=des)
        flms = [flm, flm, flm]
        # prepare correct input dataframe and lists
        shapes = ((7, 8, 9, 1),)
        _, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]

        dfcols = ['subject_id', 'map_name', 'effects_map_path']
        dfrows = [['1', 'a', FUNCFILE], ['2', 'a', FUNCFILE],
                  ['3', 'a', FUNCFILE]]
        niidf = pd.DataFrame(dfrows, columns=dfcols)
        niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
        flcondstr = [('a', 'a')]
        flcondval = [('a', np.array([1]))]
        confounds = pd.DataFrame([['1', 1], ['2', 2], ['3', 3]],
                                 columns=['subject_id', 'conf1'])
        sdes = pd.DataFrame(X[:3, :3], columns=['a', 'b', 'c'])

        # smoke tests with correct input
        # First level models as input
        SecondLevelModel(mask=mask).fit(flms, flcondstr)
        SecondLevelModel().fit(flms, flcondval)
        SecondLevelModel().fit(flms, flcondval, confounds)
        SecondLevelModel().fit(flms, flcondval, None, sdes)
        # dataframes as input
        SecondLevelModel().fit(niidf, None)
        SecondLevelModel().fit(niidf, None, confounds)
        SecondLevelModel().fit(niidf, None, confounds, sdes)
        SecondLevelModel().fit(niidf, None, None, sdes)
        SecondLevelModel().fit(niidf, flcondstr)
        SecondLevelModel().fit(niidf, flcondstr, confounds)
        SecondLevelModel().fit(niidf, flcondstr, confounds, sdes)
        SecondLevelModel().fit(niidf, flcondstr, None, sdes)
        # niimgs as input
        SecondLevelModel().fit(niimgs, None, None, sdes)
        # test wrong input errors
        # test first level model requirements
        assert_raises(ValueError, SecondLevelModel().fit, flm, flcondval)
        assert_raises(ValueError, SecondLevelModel().fit, [flm], flcondval)
        assert_raises(ValueError, SecondLevelModel().fit, flms)
        assert_raises(ValueError, SecondLevelModel().fit, flms + [''],
                      flcondval)
        # test dataframe requirements
        assert_raises(ValueError, SecondLevelModel().fit, niidf['subject_id'])
        # test niimgs requirements
        assert_raises(ValueError, SecondLevelModel().fit, niimgs)
        assert_raises(ValueError, SecondLevelModel().fit, niimgs + [[]], sdes)
        # test first_level_conditions, confounds, and design
        assert_raises(ValueError, SecondLevelModel().fit, flms, ['', []])
        assert_raises(ValueError, SecondLevelModel().fit, flms, flcondval, [])
        assert_raises(ValueError, SecondLevelModel().fit, flms, flcondval,
                      confounds['conf1'])
        assert_raises(ValueError, SecondLevelModel().fit, flms, flcondval,
                      None, [])


def _first_level_dataframe():
    conditions = ['map_name', 'subject_id', 'map_path']
    names = ['con_01', 'con_02', 'con_01', 'con_02']
    subjects = ['01', '01', '02', '02']
    maps = ['', '', '', '']
    dataframe = pd.DataFrame({'map_name': names,
                              'subject_id': subjects,
                              'effects_map_path': maps})
    return dataframe


def test_create_second_level_design():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        first_level_input = _first_level_dataframe()
        first_level_input['effects_map_path'] = [FUNCFILE] * 4
        confounds = [['01', 0.1], ['02', 0.75]]
        confounds = pd.DataFrame(confounds, columns=['subject_id', 'f1'])
        design = create_second_level_design(first_level_input, confounds)
        expected_design = np.array([[1, 0, 1, 0, 0.1], [0, 1, 1, 0, 0.1],
                                    [1, 0, 0, 1, 0.75], [0, 1, 0, 1, 0.75]])
        assert_array_equal(design, expected_design)
        assert_true(len(design.columns) == 2 + 2 + 1)
        assert_true(len(design) == 2 + 2)
        model = SecondLevelModel(mask=mask).fit(first_level_input,
                                                confounds=confounds)
        design = model.design_matrix_
        assert_array_equal(design, expected_design)
        assert_true(len(design.columns) == 2 + 2 + 1)
        assert_true(len(design) == 2 + 2)


def test_second_level_model_glm_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # ols case
        model = SecondLevelModel(mask=mask)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4)
        model = model.fit(Y, design_matrix=X)
        labels1 = model.labels_
        results1 = model.results_
        labels2, results2 = run_glm(model.masker_.transform(Y), X, 'ols')
        assert_almost_equal(labels1, labels2, decimal=1)
        assert_equal(len(results1), len(results2))


def test_second_level_model_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # ols case
        model = SecondLevelModel(mask=mask)
        # asking for contrast before model fit gives error
        assert_raises(ValueError, model.compute_contrast, [])
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['c1'])
        model = model.fit(Y, design_matrix=X)
        ncol = len(model.design_matrix_.columns)
        c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)
        # smoke test for different contrasts in fixed effects
        model.compute_contrast(c1)
        model.compute_contrast(c1, 'F')
        model.compute_contrast(c1, 'F', 'z_score')
        model.compute_contrast(c1, 'F', 'stat')
        model.compute_contrast(c1, 'F', 'p_value')
        model.compute_contrast(c1, 'F', 'effect_size')
        model.compute_contrast(c1, 'F', 'effect_variance')
        # formula should work (passing variable name directly)
        model.compute_contrast('c1')
        # passing null contrast should give back a value error
        assert_raises(ValueError, model.compute_contrast, cnull)
        # passing wrong parameters
        assert_raises(ValueError, model.compute_contrast, [])
        assert_raises(ValueError, model.compute_contrast, c1, '', '')
        assert_raises(ValueError, model.compute_contrast, c1, '', [])
