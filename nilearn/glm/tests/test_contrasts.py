import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from nilearn.glm.contrasts import (Contrast,
                                   _compute_fixed_effect_contrast,
                                   _compute_fixed_effects_params,
                                   compute_contrast,
                                   expression_to_contrast_vector)
from nilearn.glm.first_level import run_glm


def test_expression_to_contrast_vector():
    cols = "a face xy_z house window".split()
    contrast = expression_to_contrast_vector(
        "face / 10 + (window - face) * 2 - house", cols)
    assert np.allclose(contrast, [0., -1.9, 0., -1., 2.])
    contrast = expression_to_contrast_vector("xy_z", cols)
    assert np.allclose(contrast, [0., 0., 1., 0., 0.])
    cols = ["a", "b", "a - b"]
    contrast = expression_to_contrast_vector("a - b", cols)
    assert np.allclose(contrast, [0., 0., 1.])
    cols = ["column_1"]
    contrast = expression_to_contrast_vector("column_1", cols)
    assert np.allclose(contrast, [1.])


def test_Tcontrast():
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    labels, results = run_glm(Y, X, 'ar1')
    con_val = np.eye(q)[0]
    z_vals = compute_contrast(labels, results, con_val).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast():
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    for model in ['ols', 'ar1']:
        labels, results = run_glm(Y, X, model)
        for con_val in [np.eye(q)[0], np.eye(q)[:3]]:
            z_vals = compute_contrast(
                labels, results, con_val, contrast_type='F').z_score()
            assert_almost_equal(z_vals.mean(), 0, 0)
            assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add():
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    lab, res = run_glm(Y, X, 'ols')
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_fixed_effect_contrast():
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    lab, res = run_glm(Y, X, 'ols')
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = _compute_fixed_effect_contrast([lab, lab], [res, res], [c1, c2])
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_fixed_effect_contrast_nonzero_effect():
    X, y = make_regression(n_features=5, n_samples=20, random_state=0)
    y = y[:, None]
    labels, results = run_glm(y, X, 'ols')
    coef = LinearRegression(fit_intercept=False).fit(X, y).coef_
    for i in range(X.shape[1]):
        contrast = np.zeros(X.shape[1])
        contrast[i] = 1.
        fixed_effect = _compute_fixed_effect_contrast([labels],
                                                      [results],
                                                      [contrast],
                                                      )
        assert_almost_equal(fixed_effect.effect_size(), coef.ravel()[i])
        fixed_effect = _compute_fixed_effect_contrast(
            [labels] * 3, [results] * 3, [contrast] * 3)
        assert_almost_equal(fixed_effect.effect_size(), coef.ravel()[i])


def test_F_contrast_add():
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    lab, res = run_glm(Y, X, 'ar1')
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
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    lab, res = run_glm(Y, X, 'ar1')
    for c1 in [np.eye(q)[0], np.eye(q)[:3]]:
        con1 = compute_contrast(lab, res, c1)
        con2 = con1 * 2
        assert_almost_equal(con1.effect * 2, con2.effect)
        assert_almost_equal(con1.z_score(), con2.z_score())


def test_contrast_values():
    # but this test is circular and should be removed
    rng = np.random.RandomState(42)
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    lab, res = run_glm(Y, X, 'ar1', bins=1)
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


def test_low_level_fixed_effects():
    rng = np.random.RandomState(42)
    p = 100
    # X1 is some effects estimate, V1 their variance for "session 1"
    X1, V1 = rng.standard_normal(size=p), np.ones(p)
    # same thing for a "session 2"
    X2, V2 = 2 * X1, 4 * V1
    # compute the fixed effects estimate, Xf, their variance Vf,
    # and the corresponding t statistic tf
    Xf, Vf, tf = _compute_fixed_effects_params([X1, X2], [V1, V2],
                                               precision_weighted=False)
    # check that the values are correct
    assert_almost_equal(Xf, 1.5 * X1)
    assert_almost_equal(Vf, 1.25 * V1)
    assert_almost_equal(tf, Xf / np.sqrt(Vf))

    # Same thing, but now there is no precision weighting
    Xw, Vw, tw = _compute_fixed_effects_params([X1, X2], [V1, V2],
                                               precision_weighted=True)
    assert_almost_equal(Xw, 1.2 * X1)
    assert_almost_equal(Vw, .8 * V1)


def test_one_minus_pvalue():
    effect = np.ones((1, 3))
    variance = effect[0]
    contrast = Contrast(effect, variance, contrast_type="t")
    assert np.allclose(contrast.one_minus_pvalue(), 0.84, 1)
    assert np.allclose(contrast.stat_, 1., 1)
