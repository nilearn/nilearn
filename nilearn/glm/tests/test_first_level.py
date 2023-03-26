"""
Test the first level model.
"""
import os
import shutil
import unittest.mock

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image, load
from nibabel.tmpdirs import InTemporaryDirectory
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_array_less)
from sklearn.cluster import KMeans

from nilearn._utils.data_gen import (create_fake_bids_dataset,
                                     generate_fake_fmri_data_and_design,
                                     write_fake_fmri_data_and_design,
                                     basic_paradigm)
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.glm.first_level import (FirstLevelModel, first_level_from_bids,
                                     mean_scaling, run_glm)
from nilearn.glm.first_level.design_matrix import (
    check_design_matrix, make_first_level_design_matrix)
from nilearn.glm.first_level.first_level import _yule_walker
from nilearn.image import get_data
from nilearn.interfaces.bids import get_bids_files
from nilearn.glm.regression import ARModel, OLSModel
from nilearn.maskers import NiftiMasker


BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, 'functional.nii.gz')


def test_high_level_glm_one_session():
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    # Give an unfitted NiftiMasker as mask_img and check that we get an error
    masker = NiftiMasker(mask)
    with pytest.raises(ValueError,
                       match="It seems that NiftiMasker has not been fitted."):
        FirstLevelModel(mask_img=masker).fit(
            fmri_data[0], design_matrices=design_matrices[0])

    # Give a fitted NiftiMasker with a None mask_img_ attribute
    # and check that the masker parameters are overridden by the
    # FirstLevelModel parameters
    masker.fit()
    masker.mask_img_ = None
    with pytest.warns(UserWarning,
                      match="Parameter memory of the masker overridden"):
        FirstLevelModel(mask_img=masker).fit(
            fmri_data[0], design_matrices=design_matrices[0])

    # Give a fitted NiftiMasker
    masker = NiftiMasker(mask)
    masker.fit()
    single_session_model = FirstLevelModel(mask_img=masker).fit(
        fmri_data[0], design_matrices=design_matrices[0])
    assert single_session_model.masker_ == masker

    # Call with verbose (improve coverage)
    single_session_model = FirstLevelModel(mask_img=None,
                                           verbose=1).fit(
        fmri_data[0], design_matrices=design_matrices[0])

    single_session_model = FirstLevelModel(mask_img=None).fit(
        fmri_data[0], design_matrices=design_matrices[0])
    assert isinstance(single_session_model.masker_.mask_img_,
                      Nifti1Image)

    single_session_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data[0], design_matrices=design_matrices[0])
    z1 = single_session_model.compute_contrast(np.eye(rk)[:1])
    assert isinstance(z1, Nifti1Image)


def test_explicit_fixed_effects():
    """ tests the fixed effects performed manually/explicitly"""
    with InTemporaryDirectory():
        shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
        mask, fmri_data, design_matrices =\
            write_fake_fmri_data_and_design(shapes, rk)
        contrast = np.eye(rk)[1]
        # session 1
        multi_session_model = FirstLevelModel(mask_img=mask).fit(
            fmri_data[0], design_matrices=design_matrices[:1])
        dic1 = multi_session_model.compute_contrast(
            contrast, output_type='all')

        # session 2
        multi_session_model.fit(
            fmri_data[1], design_matrices=design_matrices[1:])
        dic2 = multi_session_model.compute_contrast(
            contrast, output_type='all')

        # fixed effects model
        multi_session_model.fit(
            fmri_data, design_matrices=design_matrices)
        fixed_fx_dic = multi_session_model.compute_contrast(
            contrast, output_type='all')

        # manual version
        contrasts = [dic1['effect_size'], dic2['effect_size']]
        variance = [dic1['effect_variance'], dic2['effect_variance']]
        (
            fixed_fx_contrast,
            fixed_fx_variance,
            fixed_fx_stat,
        ) = compute_fixed_effects(contrasts, variance, mask)

        assert_almost_equal(
            get_data(fixed_fx_contrast),
            get_data(fixed_fx_dic['effect_size']))
        assert_almost_equal(
            get_data(fixed_fx_variance),
            get_data(fixed_fx_dic['effect_variance']))
        assert_almost_equal(
            get_data(fixed_fx_stat), get_data(fixed_fx_dic['stat']))

        # test without mask variable
        (
            fixed_fx_contrast,
            fixed_fx_variance,
            fixed_fx_stat,
        ) = compute_fixed_effects(contrasts, variance)
        assert_almost_equal(
            get_data(fixed_fx_contrast),
            get_data(fixed_fx_dic['effect_size']))
        assert_almost_equal(
            get_data(fixed_fx_variance),
            get_data(fixed_fx_dic['effect_variance']))
        assert_almost_equal(
            get_data(fixed_fx_stat), get_data(fixed_fx_dic['stat']))

        # ensure that using unbalanced effects size and variance images
        # raises an error
        with pytest.raises(ValueError):
            compute_fixed_effects(contrasts * 2, variance, mask)
        del mask, multi_session_model


def test_high_level_glm_with_data():
    with InTemporaryDirectory():
        shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
        mask, fmri_data, design_matrices =\
            write_fake_fmri_data_and_design(shapes, rk)
        multi_session_model = FirstLevelModel(mask_img=None).fit(
            fmri_data, design_matrices=design_matrices)
        n_voxels = get_data(multi_session_model.masker_.mask_img_).sum()
        z_image = multi_session_model.compute_contrast(np.eye(rk)[1])
        assert np.sum(get_data(z_image) != 0) == n_voxels
        assert get_data(z_image).std() < 3.
        # with mask
        multi_session_model = FirstLevelModel(mask_img=mask).fit(
            fmri_data, design_matrices=design_matrices)
        z_image = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='z_score')
        p_value = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='p_value')
        stat_image = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='stat')
        effect_image = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='effect_size')
        variance_image = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='effect_variance')
        assert_array_equal(get_data(z_image) == 0., get_data(load(mask)) == 0.)
        assert (get_data(variance_image)[get_data(load(mask)) > 0] > .001
                ).all()
        all_images = multi_session_model.compute_contrast(
            np.eye(rk)[:2], output_type='all')
        assert_array_equal(get_data(all_images['z_score']), get_data(z_image))
        assert_array_equal(get_data(all_images['p_value']), get_data(p_value))
        assert_array_equal(get_data(all_images['stat']), get_data(stat_image))
        assert_array_equal(get_data(all_images['effect_size']),
                           get_data(effect_image))
        assert_array_equal(get_data(all_images['effect_variance']),
                           get_data(variance_image))
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del (all_images,
             design_matrices,
             effect_image,
             fmri_data,
             mask,
             multi_session_model,
             n_voxels,
             p_value,
             rk,
             shapes,
             stat_image,
             variance_image,
             z_image)


def test_high_level_glm_with_paths():
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 14)), 3
    with InTemporaryDirectory():
        mask_file, fmri_files, design_files =\
            write_fake_fmri_data_and_design(shapes, rk)
        multi_session_model = FirstLevelModel(mask_img=None).fit(
            fmri_files, design_matrices=design_files)
        z_image = multi_session_model.compute_contrast(np.eye(rk)[1])
        assert_array_equal(z_image.affine, load(mask_file).affine)
        assert get_data(z_image).std() < 3.
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del z_image, fmri_files, multi_session_model


def test_high_level_glm_null_contrasts():
    # test that contrast computation is resilient to 0 values.
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    mask, fmri_data, design_matrices = \
        generate_fake_fmri_data_and_design(shapes, rk)

    multi_session_model = FirstLevelModel(mask_img=None).fit(
        fmri_data, design_matrices=design_matrices)
    single_session_model = FirstLevelModel(mask_img=None).fit(
        fmri_data[0], design_matrices=design_matrices[0])
    z1 = multi_session_model.compute_contrast([np.eye(rk)[:1],
                                               np.zeros((1, rk))],
                                              output_type='stat')
    z2 = single_session_model.compute_contrast(np.eye(rk)[:1],
                                               output_type='stat')
    np.testing.assert_almost_equal(get_data(z1), get_data(z2))


def test_high_level_glm_different_design_matrices():
    # test that one can estimate a contrast when design matrices are different
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    # add a column to the second design matrix
    design_matrices[1]['new'] = np.ones((19, 1))

    # Fit a glm with two sessions and design matrices
    multi_session_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices)
    z_joint = multi_session_model.compute_contrast(
        [np.eye(rk)[:1], np.eye(rk + 1)[:1]], output_type='effect_size')
    assert z_joint.shape == (7, 8, 7)

    # compare the estimated effects to seprarately-fitted models
    model1 = FirstLevelModel(mask_img=mask).fit(
        fmri_data[0], design_matrices=design_matrices[0])
    z1 = model1.compute_contrast(np.eye(rk)[:1], output_type='effect_size')
    model2 = FirstLevelModel(mask_img=mask).fit(
        fmri_data[1], design_matrices=design_matrices[1])
    z2 = model2.compute_contrast(np.eye(rk + 1)[:1],
                                 output_type='effect_size')
    assert_almost_equal(get_data(z1) + get_data(z2),
                        2 * get_data(z_joint))


def test_high_level_glm_different_design_matrices_formulas():
    # test that one can estimate a contrast when design matrices are different
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    # make column names identical
    design_matrices[1].columns = design_matrices[0].columns
    # add a column to the second design matrix
    design_matrices[1]['new'] = np.ones((19, 1))

    # Fit a glm with two sessions and design matrices
    multi_session_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices)

    # Compute contrast with formulas
    cols_formula = tuple(design_matrices[0].columns[:2])
    formula = "%s-%s" % cols_formula
    with pytest.warns(UserWarning, match='One contrast given, '
                                         'assuming it for all 2 runs'):
        multi_session_model.compute_contrast(formula,
                                             output_type='effect_size')


def test_compute_contrast_num_contrasts():
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19), (7, 8, 7, 13)), 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    # Fit a glm with 3 sessions and design matrices
    multi_session_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices)

    # raise when n_contrast != n_runs | 1
    with pytest.raises(ValueError):
        multi_session_model.compute_contrast([np.eye(rk)[1]] * 2)

    multi_session_model.compute_contrast([np.eye(rk)[1]] * 3)
    with pytest.warns(UserWarning, match='One contrast given, '
                                         'assuming it for all 3 runs'):
        multi_session_model.compute_contrast([np.eye(rk)[1]])


def test_run_glm():
    rng = np.random.RandomState(42)
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    # Ordinary Least Squares case
    labels, results = run_glm(Y, X, 'ols')
    assert_array_equal(labels, np.zeros(n))
    assert list(results.keys()) == [0.0]
    assert results[0.0].theta.shape == (q, n)
    assert_almost_equal(results[0.0].theta.mean(), 0, 1)
    assert_almost_equal(results[0.0].theta.var(), 1. / p, 1)
    assert type(results[labels[0]].model) == OLSModel

    # ar(1) case
    labels, results = run_glm(Y, X, 'ar1')
    assert len(labels) == n
    assert len(results.keys()) > 1
    tmp = sum([val.theta.shape[1] for val in results.values()])
    assert tmp == n
    assert results[labels[0]].model.order == 1
    assert type(results[labels[0]].model) == ARModel

    # ar(3) case
    labels_ar3, results_ar3 = run_glm(Y, X, 'ar3', bins=10)
    assert len(labels_ar3) == n
    assert len(results_ar3.keys()) > 1
    tmp = sum([val.theta.shape[1] for val in results_ar3.values()])
    assert tmp == n
    assert type(results_ar3[labels_ar3[0]].model) == ARModel
    assert results_ar3[labels_ar3[0]].model.order == 3
    assert len(results_ar3[labels_ar3[0]].model.rho) == 3

    # Check correct errors are thrown for nonsense noise model requests
    with pytest.raises(ValueError):
        run_glm(Y, X, 'ar0')
    with pytest.raises(ValueError):
        run_glm(Y, X, 'arfoo')
    with pytest.raises(ValueError):
        run_glm(Y, X, 'arr3')
    with pytest.raises(ValueError):
        run_glm(Y, X, 'ar1.2')
    with pytest.raises(ValueError):
        run_glm(Y, X, 'ar')
    with pytest.raises(ValueError):
        run_glm(Y, X, '3ar')


def test_glm_AR_estimates():
    """Test that Yule-Walker AR fits are correct."""

    n, p, q = 1, 500, 2
    X_orig = np.random.RandomState(2).randn(p, q)
    Y_orig = np.random.RandomState(2).randn(p, n)

    for ar_vals in [[-0.2], [-0.2, -0.5], [-0.2, -0.5, -0.7, -0.3]]:
        ar_order = len(ar_vals)
        ar_arg = 'ar' + str(ar_order)

        X = X_orig.copy()
        Y = Y_orig.copy()

        for idx in range(1, len(Y)):
            for lag in range(ar_order):
                Y[idx] += ar_vals[lag] * Y[idx - 1 - lag]

        # Test using run_glm
        labels, results = run_glm(Y, X, ar_arg, bins=100)
        assert len(labels) == n
        for lab in results.keys():
            ar_estimate = lab.split("_")
            for lag in range(ar_order):
                assert_almost_equal(float(ar_estimate[lag]),
                                    ar_vals[lag], decimal=1)

        # Test using _yule_walker
        yw = _yule_walker(Y.T, ar_order)
        assert_almost_equal(yw[0], ar_vals, decimal=1)

    with pytest.raises(TypeError):
        _yule_walker(Y_orig, 1.2)
    with pytest.raises(ValueError):
        _yule_walker(Y_orig, 0)
    with pytest.raises(ValueError):
        _yule_walker(Y_orig, -2)
    with pytest.raises(TypeError, match='at least 1 dim'):
        _yule_walker(np.array(0.), 2)


@pytest.mark.parametrize("random_state", [3, np.random.RandomState(42)])
def test_glm_random_state(random_state):
    rng = np.random.RandomState(42)
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    with unittest.mock.patch.object(
        KMeans,
        "__init__",
        autospec=True,
        side_effect=KMeans.__init__,
    ) as spy_kmeans:
        run_glm(Y, X, 'ar3', random_state=random_state)
        spy_kmeans.assert_called_once_with(
            unittest.mock.ANY,
            n_clusters=unittest.mock.ANY,
            random_state=random_state)


def test_scaling():
    """Test the scaling function"""
    rng = np.random.RandomState(42)
    shape = (400, 10)
    u = rng.standard_normal(size=shape)
    mean = 100 * rng.uniform(size=shape[1]) + 1
    Y = u + mean
    Y_, mean_ = mean_scaling(Y)
    assert_almost_equal(Y_.mean(0), 0, 5)
    assert_almost_equal(mean_, mean, 0)
    assert Y.std() > 1


def test_fmri_inputs():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        conf = pd.DataFrame([0, 0])
        des = pd.DataFrame(np.ones((T, 1)), columns=[''])
        des_fname = 'design.csv'
        des.to_csv(des_fname)
        events = basic_paradigm()
        for fi in func_img, FUNCFILE:
            for d in des, des_fname:
                FirstLevelModel().fit(fi, design_matrices=d)
                FirstLevelModel(mask_img=None).fit([fi], design_matrices=d)
                FirstLevelModel(mask_img=mask).fit(fi, design_matrices=[d])
                FirstLevelModel(mask_img=mask).fit([fi], design_matrices=[d])
                with pytest.warns(UserWarning, match="If design matrices "
                                                     "are supplied"):
                    # test with confounds
                    FirstLevelModel(mask_img=mask).fit([fi],
                                                       design_matrices=[d],
                                                       confounds=conf)

                # Provide t_r, confounds, and events but no design matrix
                FirstLevelModel(mask_img=mask, t_r=2.0).fit(
                    fi,
                    confounds=pd.DataFrame([0] * 10, columns=['conf']),
                    events=events)

                # Same, but check that an error is raised if there is a
                # mismatch in the dimensions of the inputs
                with pytest.raises(ValueError,
                                   match="Rows in confounds does not match"):
                    FirstLevelModel(mask_img=mask, t_r=2.0).fit(
                        fi, confounds=conf, events=events)

                # test with confounds as numpy array
                FirstLevelModel(mask_img=mask).fit([fi], design_matrices=[d],
                                                   confounds=conf.values)

                FirstLevelModel(mask_img=mask).fit([fi, fi],
                                                   design_matrices=[d, d])
                FirstLevelModel(mask_img=None).fit((fi, fi),
                                                   design_matrices=(d, d))
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None).fit([fi, fi], d)
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None).fit(fi, [d, d])
                # At least paradigms or design have to be given
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None).fit(fi)
                # If paradigms are given then both tr and slice time ref were
                # required
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None).fit(fi, d)
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None, t_r=1.0).fit(fi, d)
                with pytest.raises(ValueError):
                    FirstLevelModel(mask_img=None,
                                    slice_time_ref=0.).fit(fi, d)
            # confounds rows do not match n_scans
            with pytest.raises(ValueError):
                FirstLevelModel(mask_img=None).fit(fi, d, conf)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del fi, func_img, mask, d, des, FUNCFILE, _


def test_first_level_design_creation():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # basic test based on basic_paradigm and glover hrf
        t_r = 10.0
        slice_time_ref = 0.
        events = basic_paradigm()
        model = FirstLevelModel(t_r, slice_time_ref, mask_img=mask,
                                drift_model='polynomial', drift_order=3)
        model = model.fit(func_img, events)
        frame1, X1, names1 = check_design_matrix(model.design_matrices_[0])
        # check design computation is identical
        n_scans = get_data(func_img).shape[3]
        start_time = slice_time_ref * t_r
        end_time = (n_scans - 1 + slice_time_ref) * t_r
        frame_times = np.linspace(start_time, end_time, n_scans)
        design = make_first_level_design_matrix(frame_times, events,
                                                drift_model='polynomial',
                                                drift_order=3)
        frame2, X2, names2 = check_design_matrix(design)
        assert_array_equal(frame1, frame2)
        assert_array_equal(X1, X2)
        assert_array_equal(names1, names2)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del FUNCFILE, mask, model, func_img


def test_first_level_glm_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # basic test based on basic_paradigm and glover hrf
        t_r = 10.0
        slice_time_ref = 0.
        events = basic_paradigm()
        # Ordinary Least Squares case
        model = FirstLevelModel(t_r, slice_time_ref, mask_img=mask,
                                drift_model='polynomial', drift_order=3,
                                minimize_memory=False)
        model = model.fit(func_img, events)

        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del mask, FUNCFILE, func_img, model


def test_first_level_glm_computation_with_memory_caching():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # initialize FirstLevelModel with memory option enabled
        t_r = 10.0
        slice_time_ref = 0.
        events = basic_paradigm()
        # Ordinary Least Squares case
        model = FirstLevelModel(t_r, slice_time_ref, mask_img=mask,
                                drift_model='polynomial', drift_order=3,
                                memory='nilearn_cache', memory_level=1,
                                minimize_memory=False)
        model.fit(func_img, events)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del mask, func_img, FUNCFILE, model


def test_first_level_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # basic test based on basic_paradigm and glover hrf
        t_r = 10.0
        slice_time_ref = 0.
        events = basic_paradigm()
        # Ordinary Least Squares case
        model = FirstLevelModel(t_r, slice_time_ref, mask_img=mask,
                                drift_model='polynomial', drift_order=3,
                                minimize_memory=False)
        c1, c2, cnull = np.eye(7)[0], np.eye(7)[1], np.zeros(7)
        # asking for contrast before model fit gives error
        with pytest.raises(ValueError):
            model.compute_contrast(c1)
        # fit model
        model = model.fit([func_img, func_img], [events, events])
        # Check that an error is raised for invalid contrast_def
        with pytest.raises(ValueError,
                           match="contrast_def must be an "
                                 "array or str or list"):
            model.compute_contrast(37)
        # smoke test for different contrasts in fixed effects
        model.compute_contrast([c1, c2])
        # smoke test for same contrast in fixed effects
        model.compute_contrast([c2, c2])
        # smoke test for contrast that will be repeated
        model.compute_contrast(c2)
        model.compute_contrast(c2, 'F')
        model.compute_contrast(c2, 't', 'z_score')
        model.compute_contrast(c2, 't', 'stat')
        model.compute_contrast(c2, 't', 'p_value')
        model.compute_contrast(c2, None, 'effect_size')
        model.compute_contrast(c2, None, 'effect_variance')
        # formula should work (passing variable name directly)
        model.compute_contrast('c0')
        model.compute_contrast('c1')
        model.compute_contrast('c2')
        # smoke test for one null contrast in group
        model.compute_contrast([c2, cnull])
        # only passing null contrasts should give back a value error
        with pytest.raises(ValueError):
            model.compute_contrast(cnull)
        with pytest.raises(ValueError):
            model.compute_contrast([cnull, cnull])
        # passing wrong parameters
        with pytest.raises(ValueError):
            model.compute_contrast([])
        with pytest.raises(ValueError):
            model.compute_contrast([c1, []])
        with pytest.raises(ValueError):
            model.compute_contrast(c1, '', '')
        with pytest.raises(ValueError):
            model.compute_contrast(c1, '', [])
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model


def test_first_level_from_bids():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # test arguments are provided correctly
        with pytest.raises(TypeError):
            first_level_from_bids(2, 'main', 'MNI')
        with pytest.raises(ValueError):
            first_level_from_bids('lolo', 'main', 'MNI')
        with pytest.raises(TypeError):
            first_level_from_bids(bids_path, 2, 'MNI')
        with pytest.raises(TypeError):
            first_level_from_bids(bids_path, 'main', 'MNI',
                                  model_init=[])
        with pytest.raises(TypeError,
                           match="space_label must be a string"):
            first_level_from_bids(bids_path, 'main',
                                  space_label=42)

        with pytest.raises(TypeError,
                           match="sub_labels must be a list"):
            first_level_from_bids(bids_path, 'main',
                                  sub_labels="foo")

        with pytest.raises(TypeError,
                           match="img_filters must be a list"):
            first_level_from_bids(bids_path, 'main',
                                  img_filters="foo")

        with pytest.raises(TypeError,
                           match="filters in img"):
            first_level_from_bids(bids_path, 'main',
                                  img_filters=[(1, 2)])

        with pytest.raises(ValueError,
                           match="field foo is not a possible filter."):
            first_level_from_bids(bids_path, 'main',
                                  img_filters=[("foo", "bar")])

        # test output is as expected
        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            bids_path, 'main', 'MNI', img_filters=[('desc', 'preproc')])
        assert len(models) == len(m_imgs)
        assert len(models) == len(m_events)
        assert len(models) == len(m_confounds)
        # test repeated run tag error when run tag is in filenames
        # can arise when desc or space is present and not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path, 'main', 'T1w')  # desc not specified
        # test more than one ses file error when run tag is not in filenames
        # can arise when desc or space is present and not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path, 'localizer', 'T1w')  # desc not specified
        # test issues with confound files. There should be only one confound
        # file per img. An one per image or None. Case when one is missing
        confound_files = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                        file_tag='desc-confounds_timeseries')
        os.remove(confound_files[-1])
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path, 'main', 'MNI')
        # test issues with event files
        events_files = get_bids_files(bids_path, file_tag='events')
        os.remove(events_files[0])
        # one file missing
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path, 'main', 'MNI')
        for f in events_files[1:]:
            os.remove(f)
        # all files missing
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path, 'main', 'MNI')

        # In case different desc and spaces exist and are not selected we
        # fail and ask for more specific information
        shutil.rmtree(os.path.join(bids_path, 'derivatives'))
        # issue if no derivatives folder is present
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path, 'main', 'MNI')

        # check runs are not repeated when ses field is not used
        shutil.rmtree(bids_path)
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=1,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3], no_session=True)
        # test repeated run tag error when run tag is in filenames and not ses
        # can arise when desc or space is present and not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path, 'main', 'T1w')  # desc not specified


def test_first_level_from_bids_with_subject_labels():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        warning_message = ('Subject label foo is not present in'
                           ' the dataset and cannot be processed')
        # check that the incorrect label `foo` raises a warning
        with pytest.warns(UserWarning, match=warning_message):
            models, m_imgs, m_events, m_confounds = first_level_from_bids(
                                  bids_path, 'main',
                                  sub_labels=["foo", "01"],
                                  space_label='MNI',
                                  img_filters=[('desc', 'preproc')])
            # check that the correct label `01` gets a model
            assert models[0].subject_label == '01'

def test_first_level_with_scaling():
    shapes, rk = [(3, 1, 1, 2)], 1
    fmri_data = list()
    fmri_data.append(Nifti1Image(np.zeros((1, 1, 1, 2)) + 6, np.eye(4)))
    design_matrices = list()
    design_matrices.append(
        pd.DataFrame(
            np.ones((shapes[0][-1], rk)),
            columns=list('abcdefghijklmnopqrstuvwxyz')[:rk])
    )
    fmri_glm = FirstLevelModel(
        mask_img=False, noise_model='ols', signal_scaling=0,
        minimize_memory=True
    )
    assert fmri_glm.signal_scaling == 0
    assert not fmri_glm.standardize
    with pytest.warns(DeprecationWarning,
                      match="Deprecated. `scaling_axis` will be removed"):
        assert fmri_glm.scaling_axis == 0
    glm_parameters = fmri_glm.get_params()
    test_glm = FirstLevelModel(**glm_parameters)
    fmri_glm = fmri_glm.fit(fmri_data, design_matrices=design_matrices)
    test_glm = test_glm.fit(fmri_data, design_matrices=design_matrices)
    assert glm_parameters['signal_scaling'] == 0


def test_first_level_with_no_signal_scaling():
    """
    test to ensure that the FirstLevelModel works correctly with a
    signal_scaling==False. In particular, that derived theta are correct for a
    constant design matrix with a single valued fmri image
    """
    shapes, rk = [(3, 1, 1, 2)], 1
    fmri_data = list()
    design_matrices = list()
    design_matrices.append(pd.DataFrame(np.ones((shapes[0][-1], rk)),
                                        columns=list(
                                            'abcdefghijklmnopqrstuvwxyz')[:rk])
                           )
    # Check error with invalid signal_scaling values
    with pytest.raises(ValueError,
                       match="signal_scaling must be"):
        FirstLevelModel(mask_img=False, noise_model='ols',
                        signal_scaling="foo")

    first_level = FirstLevelModel(mask_img=False, noise_model='ols',
                                  signal_scaling=False)
    fmri_data.append(Nifti1Image(np.zeros((1, 1, 1, 2)) + 6, np.eye(4)))

    first_level.fit(fmri_data, design_matrices=design_matrices)
    # trivial test of signal_scaling value
    assert first_level.signal_scaling is False
    # assert that our design matrix has one constant
    assert first_level.design_matrices_[0].equals(
        pd.DataFrame([1.0, 1.0], columns=['a']))
    # assert that we only have one theta as there is only on voxel in our image
    assert first_level.results_[0][0].theta.shape == (1, 1)
    # assert that the theta is equal to the one voxel value
    assert_almost_equal(first_level.results_[0][0].theta[0, 0], 6.0, 2)


def test_first_level_residuals():
    shapes, rk = [(10, 10, 10, 100)], 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    # Check that voxelwise model attributes cannot be
    # accessed if minimize_memory is set to True
    model = FirstLevelModel(mask_img=mask, minimize_memory=True,
                            noise_model='ols')
    model.fit(fmri_data, design_matrices=design_matrices)

    with pytest.raises(ValueError,
                       match="To access voxelwise attributes"):
        residuals = model.residuals[0]

    model = FirstLevelModel(mask_img=mask, minimize_memory=False,
                            noise_model='ols')

    # Check that trying to access residuals without fitting
    # raises an error
    with pytest.raises(ValueError,
                       match="The model has not been fit yet"):
        residuals = model.residuals[0]

    model.fit(fmri_data, design_matrices=design_matrices)

    # For coverage
    with pytest.raises(ValueError,
                       match="attribute must be one of"):
        model._get_voxelwise_model_attribute("foo", True)
    residuals = model.residuals[0]
    mean_residuals = model.masker_.transform(residuals).mean(0)
    assert_array_almost_equal(mean_residuals, 0)


@pytest.mark.parametrize("shapes", [
    [(10, 10, 10, 25)],
    [(10, 10, 10, 25), (10, 10, 10, 100)],
])
def test_get_voxelwise_attributes_should_return_as_many_as_design_matrices(shapes):
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes)

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    model = FirstLevelModel(mask_img=mask, minimize_memory=False,
                            noise_model='ols')
    model.fit(fmri_data, design_matrices=design_matrices)

    # Check that length of outputs is the same as the number of design matrices
    assert len(model._get_voxelwise_model_attribute("residuals", True)) == \
           len(shapes)


def test_first_level_predictions_r_square():
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, design_matrices =\
        generate_fake_fmri_data_and_design(shapes, rk)

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    model = FirstLevelModel(mask_img=mask,
                            signal_scaling=False,
                            minimize_memory=False,
                            noise_model='ols')
    model.fit(fmri_data, design_matrices=design_matrices)

    pred = model.predicted[0]
    data = fmri_data[0]
    r_square_3d = model.r_square[0]

    y_predicted = model.masker_.transform(pred)
    y_measured = model.masker_.transform(data)

    assert_almost_equal(np.mean(y_predicted - y_measured), 0)

    r_square_2d = model.masker_.transform(r_square_3d)
    assert_array_less(0., r_square_2d)


@pytest.mark.parametrize("hrf_model", [
    "spm",
    "spm + derivative",
    "glover",
    lambda tr, ov: np.ones(int(tr * ov))
])
@pytest.mark.parametrize("spaces", [
    False,
    True
])
def test_first_level_hrf_model(hrf_model, spaces):
    """
    Ensure that FirstLevelModel runs without raising errors
    for different values of hrf_model. In particular, one checks that it runs
    without raising errors when given a custom response function.
    When :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`
    is used errors should be raised when event (ie condition) names are not
    valid identifiers.
    """
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, _ =\
        generate_fake_fmri_data_and_design(shapes, rk)

    events = basic_paradigm(condition_names_have_spaces=spaces)

    model = FirstLevelModel(t_r=2.0,
                            mask_img=mask,
                            hrf_model=hrf_model)

    model.fit(fmri_data, events)

    columns = model.design_matrices_[0].columns
    exp = f"{columns[0]}-{columns[1]}"
    try:
        model.compute_contrast(exp)
    except Exception:
        with pytest.raises(
                ValueError,
                match='invalid python identifiers'
        ):
            model.compute_contrast(exp)


def test_glm_sample_mask():
    """Ensure the sample mask is performing correctly in GLM."""
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, design_matrix =\
        generate_fake_fmri_data_and_design(shapes, rk)
    model = FirstLevelModel(t_r=2.0,
                            mask_img=mask,
                            minimize_memory=False)
    sample_mask = np.arange(25)[3:]  # censor the first three volumes
    model.fit(fmri_data,
              design_matrices=design_matrix,
              sample_masks=sample_mask)
    assert model.design_matrices_[0].shape[0] == 22
    assert model.predicted[0].shape[-1] == 22
