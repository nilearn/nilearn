import numpy as np
import pytest
import scipy.stats as st
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from nilearn.glm.contrasts import (
    Contrast,
    _compute_fixed_effects_params,
    compute_contrast,
    compute_fixed_effect_contrast,
    expression_to_contrast_vector,
)
from nilearn.glm.first_level import run_glm


@pytest.mark.parametrize(
    "expression, design_columns, expected",
    [
        (
            "face / 10 + (window - face) * 2 - house",
            ["a", "face", "xy_z", "house", "window"],
            [0.0, -1.9, 0.0, -1.0, 2.0],
        ),
        (
            "xy_z",
            ["a", "face", "xy_z", "house", "window"],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        ("a - b", ["a", "b", "a - b"], [0.0, 0.0, 1.0]),
        ("column_1", ["column_1"], [1.0]),
    ],
)
def test_expression_to_contrast_vector(expression, design_columns, expected):
    contrast = expression_to_contrast_vector(
        expression=expression, design_columns=design_columns
    )
    assert np.allclose(contrast, expected)


def test_expression_to_contrast_vector_error():
    with pytest.raises(ValueError, match="invalid python identifiers"):
        expression_to_contrast_vector(
            expression="0-1", design_columns=["0", "1"]
        )


@pytest.fixture
def set_up_glm():
    def _set_up_glm(rng, noise_model, bins=100):
        n, p, q = 100, 80, 10
        X, Y = (
            rng.standard_normal(size=(p, q)),
            rng.standard_normal(size=(p, n)),
        )
        labels, results = run_glm(Y, X, noise_model, bins=bins)
        return labels, results, q

    return _set_up_glm


def test_deprecation_contrast_type(rng, set_up_glm):
    """Throw deprecation warning when using contrast_type as parameter."""
    labels, results, q = set_up_glm(rng, "ar1")
    con_val = np.eye(q)[0]

    with pytest.deprecated_call(match="0.13.0"):
        compute_contrast(
            labels=labels,
            regression_result=results,
            con_val=con_val,
            contrast_type="t",
        )


def test_t_contrast(rng, set_up_glm):
    labels, results, q = set_up_glm(rng, "ar1")
    con_val = np.eye(q)[0]

    z_vals = compute_contrast(labels, results, con_val).z_score()

    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


@pytest.mark.parametrize("model", ["ols", "ar1"])
def test_f_contrast(rng, set_up_glm, model):
    labels, results, q = set_up_glm(rng, model)
    for con_val in [np.eye(q)[0], np.eye(q)[:3]]:
        z_vals = compute_contrast(
            labels, results, con_val, stat_type="F"
        ).z_score()

        assert_almost_equal(z_vals.mean(), 0, 0)
        assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add(set_up_glm, rng):
    labels, results, q = set_up_glm(rng, "ols")
    c1, c2 = np.eye(q)[0], np.eye(q)[1]

    con = compute_contrast(labels, results, c1) + compute_contrast(
        labels, results, c2
    )

    z_vals = con.z_score()

    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_fixed_effect_contrast(set_up_glm, rng):
    labels, results, q = set_up_glm(rng, "ols")
    c1, c2 = np.eye(q)[0], np.eye(q)[1]

    con = compute_fixed_effect_contrast(
        [labels, labels], [results, results], [c1, c2]
    )

    z_vals = con.z_score()

    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_fixed_effect_contrast_nonzero_effect():
    X, y = make_regression(n_features=5, n_samples=20, random_state=0)
    y = y[:, None]
    labels, results = run_glm(y, X, "ols")
    coef = LinearRegression(fit_intercept=False).fit(X, y).coef_
    for i in range(X.shape[1]):
        contrast = np.zeros(X.shape[1])
        contrast[i] = 1.0
        fixed_effect = compute_fixed_effect_contrast(
            [labels],
            [results],
            [contrast],
        )

        assert_almost_equal(fixed_effect.effect_size(), coef.ravel()[i])

        fixed_effect = compute_fixed_effect_contrast(
            [labels] * 3, [results] * 3, [contrast] * 3
        )

        assert_almost_equal(fixed_effect.effect_size(), coef.ravel()[i])


def test_f_contrast_add(set_up_glm, rng):
    labels, results, q = set_up_glm(rng, "ar1")
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]

    con = compute_contrast(labels, results, c1) + compute_contrast(
        labels, results, c2
    )

    z_vals = con.z_score()

    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

    # first test with dependent contrast
    con1 = compute_contrast(labels, results, c1)
    con2 = compute_contrast(labels, results, c1) + compute_contrast(
        labels, results, c1
    )

    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())


def test_contrast_mul(set_up_glm, rng):
    labels, results, q = set_up_glm(rng, "ar1")
    for c1 in [np.eye(q)[0], np.eye(q)[:3]]:
        con1 = compute_contrast(labels, results, c1)
        con2 = con1 * 2
        assert_almost_equal(con1.effect * 2, con2.effect)
        assert_almost_equal(con1.z_score(), con2.z_score())


def test_contrast_values(set_up_glm, rng):
    # but this test is circular and should be removed
    labels, results, q = set_up_glm(rng, "ar1", bins=1)

    # t test
    cval = np.eye(q)[0]
    con = compute_contrast(labels, results, cval)
    t_ref = next(iter(results.values())).Tcontrast(cval).t

    assert_almost_equal(np.ravel(con.stat()), t_ref)

    # F test
    cval = np.eye(q)[:3]
    con = compute_contrast(labels, results, cval)
    F_ref = next(iter(results.values())).Fcontrast(cval).F

    # Note that the values are not strictly equal,
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)


def test_low_level_fixed_effects(rng):
    p = 100
    # X1 is some effects estimate, V1 their variance for "run 1"
    X1, V1 = rng.standard_normal(p), np.ones(p)
    # same thing for a "run 2"
    X2, V2 = 2 * X1, 4 * V1
    # compute the fixed effects estimate, Xf, their variance Vf,
    # and the corresponding t statistic tf
    Xf, Vf, tf, zf = _compute_fixed_effects_params(
        [X1, X2], [V1, V2], dofs=[100, 100], precision_weighted=False
    )
    # check that the values are correct
    assert_almost_equal(Xf, 1.5 * X1)
    assert_almost_equal(Vf, 1.25 * V1)
    assert_almost_equal(tf, (Xf / np.sqrt(Vf)).ravel())
    assert_almost_equal(zf, st.norm.isf(st.t.sf(tf, 200)))

    # Same thing, but now there is precision weighting
    Xw, Vw, _, _ = _compute_fixed_effects_params(
        [X1, X2], [V1, V2], dofs=[200, 200], precision_weighted=True
    )
    assert_almost_equal(Xw, 1.2 * X1)
    assert_almost_equal(Vw, 0.8 * V1)

    # F test
    XX1 = np.vstack((X1, X1))
    XX2 = np.vstack((X2, X2))

    Xw, Vw, *_ = _compute_fixed_effects_params(
        [XX1, XX2], [V1, V2], dofs=[200, 200], precision_weighted=False
    )
    assert_almost_equal(Xw, 1.5 * XX1)
    assert_almost_equal(Vw, 1.25 * V1)

    # check with 2D image
    Xw, Vw, *_ = _compute_fixed_effects_params(
        [X1[:, np.newaxis], X2[:, np.newaxis]],
        [V1, V2],
        dofs=[200, 200],
        precision_weighted=False,
    )
    assert_almost_equal(Xw, 1.5 * X1[:, np.newaxis])
    assert_almost_equal(Vw, 1.25 * V1)


def test_one_minus_pvalue():
    effect = np.ones((1, 3))
    variance = effect[0]

    contrast = Contrast(effect, variance, stat_type="t")

    assert np.allclose(contrast.one_minus_pvalue(), 0.84, 1)
    assert np.allclose(contrast.stat_, 1.0, 1)


def test_deprecation_contrast_type_attribute():
    effect = np.ones((1, 3))
    variance = effect[0]

    with pytest.deprecated_call(match="0.13.0"):
        contrast = Contrast(effect, variance, contrast_type="t")

    with pytest.deprecated_call(match="0.13.0"):
        contrast.contrast_type  # noqa: B018


@pytest.mark.parametrize(
    "effect, variance, match",
    [
        (
            np.ones((3, 1, 1)),
            np.ones(1),
            "Effect array should have 1 or 2 dimensions",
        ),
        (
            np.ones((1, 3)),
            np.ones((1, 1)),
            "Variance array should have 1 dimension",
        ),
    ],
)
def test_improper_contrast_inputs(effect, variance, match):
    with pytest.raises(ValueError, match=match):
        Contrast(effect, variance, stat_type="t")


def test_automatic_t2f_conversion():
    effect = np.ones((5, 3))
    variance = np.ones(5)
    contrast = Contrast(effect, variance, stat_type="t")
    assert contrast.stat_type == "F"


def test_invalid_contrast_type():
    effect = np.ones((1, 3))
    variance = np.ones(1)
    with pytest.raises(ValueError, match="is not a valid stat_type."):
        Contrast(effect, variance, stat_type="foo")


def test_contrast_padding(rng):
    n, p, q = 100, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))
    labels, results = run_glm(Y, X, "ar1")

    con_val = [1, 1]

    with pytest.warns(
        UserWarning, match="The rest of the contrast was padded with zeros."
    ):
        compute_contrast(labels, results, con_val).z_score()

    con_val = np.eye(q)[:3, :3]
    compute_contrast(labels, results, con_val, stat_type="F").z_score()
