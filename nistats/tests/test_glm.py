# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the glm utilities.
"""
from __future__ import with_statement

import numpy as np

from nibabel import load, Nifti1Image, save

from ..glm import GeneralLinearModel, data_scaling, FMRILinearModel

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal)
from nibabel.tmpdirs import InTemporaryDirectory

from nipy.testing import funcfile


def write_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    mask_file, fmri_files, design_files = 'mask.nii', [], []
    for i, shape in enumerate(shapes):
        fmri_files.append('fmri_run%d.nii' %i)
        data = 100 + np.random.randn(*shape)
        data[0] -= 10
        save(Nifti1Image(data, affine), fmri_files[-1])
        design_files.append('dmtx_%d.npz' %i)
        np.savez(design_files[-1], np.random.randn(shape[3], rk))
    save(Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8), 
                     affine), mask_file)
    return mask_file, fmri_files, design_files 


def generate_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    fmri_data = []
    design_matrices = []
    for i, shape in enumerate(shapes):
        data = 100 + np.random.randn(*shape)
        data[0] -= 10
        fmri_data.append(Nifti1Image(data, affine))
        design_matrices.append(np.random.randn(shape[3], rk))
    mask = Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                       affine)
    return mask, fmri_data, design_matrices


def test_high_level_glm_with_paths():
    shapes, rk = ((5, 6, 4, 20), (5, 6, 4, 19)), 3
    with InTemporaryDirectory():
        mask_file, fmri_files, design_files = write_fake_fmri_data(shapes, rk)
        multi_session_model = FMRILinearModel(fmri_files, design_files,
                                              mask_file)
        multi_session_model.fit()
        z_image, = multi_session_model.contrast([np.eye(rk)[1]] * 2)
        assert_array_equal(z_image.get_affine(), load(mask_file).get_affine())
        assert_true(z_image.get_data().std() < 3.)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory
        del z_image, fmri_files, multi_session_model


def test_high_level_glm_with_data():
    shapes, rk = ((7, 6, 5, 20), (7, 6, 5, 19)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data(shapes, rk)

    # without mask
    multi_session_model = FMRILinearModel(fmri_data, design_matrices, mask=None)
    multi_session_model.fit()
    z_image, = multi_session_model.contrast([np.eye(rk)[1]] * 2)
    assert_equal(np.sum(z_image.get_data() == 0), 0)

    # compute the mask
    multi_session_model = FMRILinearModel(fmri_data, design_matrices,
                                          m=0, M=.01, threshold=0.)
    multi_session_model.fit()
    z_image, = multi_session_model.contrast([np.eye(rk)[1]] * 2)
    assert_true(z_image.get_data().std() < 3. )

    # with mask
    multi_session_model = FMRILinearModel(fmri_data, design_matrices, mask)
    multi_session_model.fit()
    z_image, effect_image, variance_image= multi_session_model.contrast(
        [np.eye(rk)[:2]] * 2, output_effects=True, output_variance=True)
    assert_array_equal(z_image.get_data() == 0., load(mask).get_data() == 0.)
    assert_true(
        (variance_image.get_data()[load(mask).get_data() > 0, 0] > .001).all())

    # without scaling
    multi_session_model.fit(do_scaling=False)
    z_image, = multi_session_model.contrast([np.eye(rk)[1]] * 2)
    assert_true(z_image.get_data().std() < 3. )


def test_high_level_glm_contrasts():
    shapes, rk = ((5, 6, 7, 20), (5, 6, 7, 19)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data(shapes, rk)
    multi_session_model = FMRILinearModel(fmri_data, design_matrices, mask=None)
    multi_session_model.fit()
    z_image, = multi_session_model.contrast([np.eye(rk)[:2]] * 2,
                                            contrast_type='tmin-conjunction')
    z1, = multi_session_model.contrast([np.eye(rk)[:1]] * 2)
    z2, = multi_session_model.contrast([np.eye(rk)[1:2]] * 2)
    assert_true((z_image.get_data() < np.maximum(
        z1.get_data(), z2.get_data())).all())


def test_high_level_glm_null_contrasts():
    shapes, rk = ((5, 6, 7, 20), (5, 6, 7, 19)), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data(shapes, rk)

    multi_session_model = FMRILinearModel(
        fmri_data, design_matrices, mask=None)
    multi_session_model.fit()
    single_session_model = FMRILinearModel(
        fmri_data[:1], design_matrices[:1], mask=None)
    single_session_model.fit()
    z1, = multi_session_model.contrast([np.eye(rk)[:1], np.zeros((1, rk))])
    z2, = single_session_model.contrast([np.eye(rk)[:1]])
    np.testing.assert_almost_equal(z1.get_data(), z2.get_data())


def ols_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    glm = GeneralLinearModel(X)
    glm.fit(Y, 'ols')
    return glm, n, p, q


def ar1_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    glm = GeneralLinearModel(X)
    glm.fit(Y, 'ar1')
    return glm, n, p, q


def test_glm_ols():
    mulm, n, p, q = ols_glm()
    assert_array_equal(mulm.labels_, np.zeros(n))
    assert_equal(mulm.results_.keys(), [0.0])
    assert_equal(mulm.results_[0.0].theta.shape, (q, n))
    assert_almost_equal(mulm.results_[0.0].theta.mean(), 0, 1)
    assert_almost_equal(mulm.results_[0.0].theta.var(), 1. / p, 1)


def test_glm_beta():
    mulm, n, p, q = ols_glm()
    assert_equal(mulm.get_beta().shape, (q, n)) 
    assert_equal(mulm.get_beta([0, -1]).shape, (2, n))
    assert_equal(mulm.get_beta(6).shape, (1, n))


def test_glm_mse():
    mulm, n, p, q = ols_glm()
    mse = mulm.get_mse()
    assert_array_almost_equal(mse, np.ones(n), 0) 


def test_glm_logL():
    mulm, n, p, q = ols_glm()
    logL = mulm.get_logL()
    assert_array_almost_equal(logL / n, - p * 1.41 * np.ones(n) / n, 0) 


def test_glm_ar():
    mulm, n, p, q = ar1_glm()
    assert_equal(len(mulm.labels_), n)
    assert_true(len(mulm.results_.keys()) > 1)
    tmp = sum([mulm.results_[key].theta.shape[1]
               for key in mulm.results_.keys()])
    assert_equal(tmp, n)


def test_Tcontrast():
    mulm, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    z_vals = mulm.contrast(cval).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_1d():
    mulm, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    con = mulm.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_nd():
    mulm, n, p, q = ar1_glm()
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    assert_equal(con.contrast_type, 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_1d_old():
    mulm, n, p, q = ols_glm()
    cval = np.hstack((1, np.ones(9)))
    con = mulm.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_nd_ols():
    mulm, n, p, q = ols_glm()
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    assert_equal(con.contrast_type, 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add():
    mulm, n, p, q = ols_glm()
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = mulm.contrast(c1) + mulm.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_F_contrast_add():
    mulm, n, p, q = ar1_glm()
    # first test with independent contrast
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]
    con = mulm.contrast(c1) + mulm.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)
    # first test with dependent contrast
    con1 = mulm.contrast(c1)
    con2 = mulm.contrast(c1) + mulm.contrast(c1)
    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())


def test_t_contrast_mul():
    mulm, n, p, q = ar1_glm()
    con1 = mulm.contrast(np.eye(q)[0])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)


def test_F_contrast_mul():
    mulm, n, p, q = ar1_glm()
    con1 = mulm.contrast(np.eye(q)[:4])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)


def test_t_contrast_values():
    mulm, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[0]
    con = mulm.contrast(cval)
    t_ref = mulm.results_.values()[0].Tcontrast(cval).t
    assert_almost_equal(np.ravel(con.stat()), t_ref)


def test_F_contrast_calues():
    mulm, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    F_ref = mulm.results_.values()[0].Fcontrast(cval).F
    # Note that the values are not strictly equal,
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)


def test_tmin():
    mulm, n, p, q = ar1_glm(n=1)
    c1, c2, c3 = np.eye(q)[0], np.eye(q)[1], np.eye(q)[2]
    t1, t2, t3 = mulm.contrast(c1).stat(), mulm.contrast(c2).stat(), \
        mulm.contrast(c3).stat()
    tmin = min(t1, t2, t3)
    con = mulm.contrast(np.eye(q)[:3], 'tmin-conjunction')
    assert_equal(con.stat(), tmin)


def test_scaling():
    """Test the scaling function"""
    shape = (400, 10)
    u = np.random.randn(*shape)
    mean = 100 * np.random.rand(shape[1])
    Y = u + mean
    Y, mean_ = data_scaling(Y)
    assert_almost_equal(Y.mean(0), 0)
    assert_almost_equal(mean_, mean, 0)
    assert_true(Y.std() > 1)


def test_fmri_inputs():
    # Test processing of FMRI inputs
    func_img = load(funcfile)
    T = func_img.shape[-1]
    des = np.ones((T, 1))
    des_fname = 'design.npz'
    with InTemporaryDirectory():
        np.savez(des_fname, des)
        for fi in func_img, funcfile:
            for d in des, des_fname:
                fmodel = FMRILinearModel(fi, d, mask='compute')
                fmodel = FMRILinearModel([fi], d, mask=None)
                fmodel = FMRILinearModel(fi, [d], mask=None)
                fmodel = FMRILinearModel([fi], [d], mask=None)
                fmodel = FMRILinearModel([fi, fi], [d, d], mask=None)
                fmodel = FMRILinearModel((fi, fi), (d, d), mask=None)
                assert_raises(ValueError, FMRILinearModel, [fi, fi], d,
                              mask=None)
                assert_raises(ValueError, FMRILinearModel, fi, [d, d],
                              mask=None)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
