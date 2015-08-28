# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the glm utilities.
"""
from __future__ import with_statement

import os

import numpy as np

from nibabel import load, Nifti1Image, save

from ..glm import (
    data_scaling, session_glm, FirstLevelGLM, compute_contrast)

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal)
from nibabel.tmpdirs import InTemporaryDirectory


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
        design_files.append('dmtx_%d.npz' % i)
        np.savez(design_files[-1], np.random.randn(shape[3], rk))
    save(Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                     affine), mask_file)
    return mask_file, fmri_files, design_files


def generate_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    fmri_data = []
    design_matrices = []
    for i, shape in enumerate(shapes):
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        fmri_data.append(Nifti1Image(data, affine))
        design_matrices.append(np.random.randn(shape[3], rk))
    mask = Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                       affine)
    return mask, fmri_data, design_matrices


def test_high_level_glm_one_session():
    # New API
    shapes, rk = [(5, 6, 7, 20)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data(shapes, rk)

    single_session_model = FirstLevelGLM(mask=None).fit(
        fmri_data[0], design_matrices[0])
    assert_true(isinstance(single_session_model.masker_.mask_img_,
                           Nifti1Image))

    single_session_model = FirstLevelGLM(mask=mask).fit(
        fmri_data[0], design_matrices[0])
    z1, = single_session_model.transform(np.eye(rk)[:1])
    assert_true(isinstance(z1, Nifti1Image))


def test_high_level_glm_with_data():
    # New API
    shapes, rk = ((7, 6, 5, 20), (7, 6, 5, 19)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data(shapes, rk)

    multi_session_model = FirstLevelGLM(mask=None).fit(
        fmri_data, design_matrices)
    n_voxels = multi_session_model.masker_.mask_img_.get_data().sum()
    z_image, = multi_session_model.transform([np.eye(rk)[1]] * 2)
    assert_equal(np.sum(z_image.get_data() != 0), n_voxels)
    assert_true(z_image.get_data().std() < 3. )

    # with mask
    multi_session_model = FirstLevelGLM(mask=mask).fit(
        fmri_data, design_matrices)
    z_image, effect_image, variance_image = multi_session_model.transform(
        [np.eye(rk)[:2]] * 2, output_effects=True, output_variance=True)
    assert_array_equal(z_image.get_data() == 0., load(mask).get_data() == 0.)
    assert_true(
        (variance_image.get_data()[load(mask).get_data() > 0, 0] > .001).all())


def test_high_level_glm_with_paths():
    # New API
    shapes, rk = ((5, 6, 4, 20), (5, 6, 4, 19)), 3
    with InTemporaryDirectory():
        mask_file, fmri_files, design_files = write_fake_fmri_data(shapes, rk)
        multi_session_model = FirstLevelGLM(mask=None).fit(
            fmri_files, design_files)
        z_image, = multi_session_model.transform([np.eye(rk)[1]] * 2)
        assert_array_equal(z_image.get_affine(), load(mask_file).get_affine())
        assert_true(z_image.get_data().std() < 3.)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory
        del z_image, fmri_files, multi_session_model


def test_high_level_glm_null_contrasts():
    # test that contrast computation is resilient to 0 values.
    # new API
    shapes, rk = ((5, 6, 7, 20), (5, 6, 7, 19)), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data(shapes, rk)

    multi_session_model = FirstLevelGLM(mask=None).fit(
        fmri_data, design_matrices)
    single_session_model = FirstLevelGLM(mask=None).fit(
        fmri_data[0], design_matrices[0])
    z1, = multi_session_model.transform([np.eye(rk)[:1], np.zeros((1, rk))],
                                        output_z=False, output_stat=True)
    z2, = single_session_model.transform([np.eye(rk)[:1]],
                                         output_z=False, output_stat=True)
    np.testing.assert_almost_equal(z1.get_data(), z2.get_data())


def test_session_glm():
    # New API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)

    # ols case
    labels, results = session_glm(Y, X, 'ols')
    assert_array_equal(labels, np.zeros(n))
    assert_equal(list(results.keys()), [0.0])
    assert_equal(results[0.0].theta.shape, (q, n))
    assert_almost_equal(results[0.0].theta.mean(), 0, 1)
    assert_almost_equal(results[0.0].theta.var(), 1. / p, 1)

    # ar(1) case
    labels, results = session_glm(Y, X, 'ar1')
    assert_equal(len(labels), n)
    assert_true(len(results.keys()) > 1)
    tmp = sum([val.theta.shape[1] for val in results.values()])
    assert_equal(tmp, n)

    # non-existant case
    assert_raises(ValueError, session_glm, Y, X, 'ar2')
    assert_raises(ValueError, session_glm, Y, X.T)


def test_Tcontrast():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    labels, results = session_glm(Y, X, 'ar1')
    con_val = np.eye(q)[0]
    z_vals = compute_contrast(labels, results, con_val).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    for model in ['ols', 'ar1']:
        labels, results = session_glm(Y, X, model)
        for con_val in [np.eye(q)[0], np.eye(q)[:3]]:
            z_vals = compute_contrast(
                labels, results, con_val, contrast_type='F').z_score()
            assert_almost_equal(z_vals.mean(), 0, 0)
            assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = session_glm(Y, X, 'ols')
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_F_contrast_add():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = session_glm(Y, X, 'ar1')
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]
    con = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

    # first test with dependent contrast
    con1 = compute_contrast(lab, res, c1)
    con2 = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c1)
    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())


def test_contrast_mul():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = session_glm(Y, X, 'ar1')
    for c1 in [np.eye(q)[0], np.eye(q)[:3]]:
        con1 = compute_contrast(lab, res, c1)
        con2 = con1 * 2
        assert_almost_equal(con1.effect * 2, con2.effect)
        # assert_almost_equal(con1.variance * 2, con2.variance) FIXME
        # assert_almost_equal(con1.stat() * 2, con2.stat()) FIXME
        assert_almost_equal(con1.z_score(), con2.z_score())


def test_contrast_values():
    # new API
    # but this test is circular and should be removed
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = session_glm(Y, X, 'ar1', bins=1)
    # t test
    cval = np.eye(q)[0]
    con = compute_contrast(lab, res, cval)
    t_ref = list(res.values())[0].Tcontrast(cval).t
    assert_almost_equal(np.ravel(con.stat()), t_ref)
    # F test
    cval = np.eye(q)[:3]
    con = compute_contrast(lab, res, cval)
    F_ref = list(res.values())[0].Fcontrast(cval).F
    # Note that the values are not strictly equal,
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)


def test_scaling():
    """Test the scaling function"""
    shape = (400, 10)
    u = np.random.randn(*shape)
    mean = 100 * np.random.rand(shape[1]) + 1
    Y = u + mean
    Y_, mean_ = data_scaling(Y)
    assert_almost_equal(Y_.mean(0), 0, 5)
    assert_almost_equal(mean_, mean, 0)
    assert_true(Y.std() > 1)


def test_fmri_inputs():
    # Test processing of FMRI inputs
    func_img = load(FUNCFILE)
    T = func_img.shape[-1]
    des = np.ones((T, 1))
    des_fname = 'design.npz'
    with InTemporaryDirectory():
        np.savez(des_fname, des)
        for fi in func_img, FUNCFILE:
            for d in des, des_fname:
                FirstLevelGLM().fit(fi, d)
                FirstLevelGLM(mask=None).fit([fi], d)
                FirstLevelGLM(mask=None).fit(fi, [d])
                FirstLevelGLM(mask=None).fit([fi], [d])
                FirstLevelGLM(mask=None).fit([fi, fi], [d, d])
                FirstLevelGLM(mask=None).fit((fi, fi), (d, d))
                assert_raises(
                    ValueError, FirstLevelGLM(mask=None).fit, [fi, fi], d)
                assert_raises(
                    ValueError, FirstLevelGLM(mask=None).fit, fi, [d, d])
