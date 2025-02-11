"""Testing models module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nilearn.glm import OLSModel

N = 10
X = np.c_[np.linspace(-1, 1, N), np.ones((N,))]
Y = np.r_[range(5), range(1, 6)]
MODEL = OLSModel(X)
RESULTS = MODEL.fit(Y)

""" R script

::

    X = cbind(0:9 * 2/9 -1, 1)
    Y = as.matrix(c(0:4, 1:5))
    results = lm(Y ~ X-1)
    print(results)
    print(summary(results))

gives::

    Call:
    lm(formula = Y ~ X - 1)

    Coefficients:
    X1     X2
    1.773  2.500

    Residuals:
        Min      1Q  Median      3Q     Max
    -1.6970 -0.6667  0.0000  0.6667  1.6970

    Coefficients:
    Estimate Std. Error t value Pr(>|t|)
    X1   1.7727     0.5455   3.250   0.0117 *
    X2   2.5000     0.3482   7.181 9.42e-05 ***
    ---

    Residual standard error: 1.101 on 8 degrees of freedom
    Multiple R-squared: 0.8859, Adjusted R-squared: 0.8574
    F-statistic: 31.06 on 2 and 8 DF,  p-value: 0.0001694
"""


def test_model():
    # Check basics about the model fit
    # Check we fit the mean
    assert_array_almost_equal(RESULTS.theta[1], np.mean(Y))
    # Check we get the same as R
    assert_array_almost_equal(RESULTS.theta, [1.773, 2.5], 3)
    percentile = np.percentile
    pcts = percentile(RESULTS.residuals, [0, 25, 50, 75, 100])
    assert_array_almost_equal(pcts, [-1.6970, -0.6667, 0, 0.6667, 1.6970], 4)


def test_t_contrast():
    # Test individual t against R
    assert_array_almost_equal(RESULTS.t(0), 3.25)
    assert_array_almost_equal(RESULTS.t(1), 7.181, 3)
    # And contrast
    assert_array_almost_equal(RESULTS.Tcontrast([1, 0]).t, 3.25)
    assert_array_almost_equal(RESULTS.Tcontrast([0, 1]).t, 7.181, 3)


def test_t_contrast_errors():
    match = "t contrasts should be of length P=.*, but it has length .*"
    with pytest.warns(UserWarning, match=match):
        RESULTS.Tcontrast([1])
    with pytest.raises(ValueError, match=match):
        RESULTS.Tcontrast([1, 0, 0])
    # And shape
    with pytest.raises(
        ValueError, match="t contrasts should have only one row"
    ):
        RESULTS.Tcontrast(np.array([1, 0])[:, None])


def test_t_output():
    # Check we get required outputs
    exp_t = RESULTS.t(0)
    exp_effect = RESULTS.theta[0]
    exp_sd = exp_effect / exp_t

    res = RESULTS.Tcontrast([1, 0])

    assert_array_almost_equal(res.t, exp_t)
    assert_array_almost_equal(res.effect, exp_effect)
    assert_array_almost_equal(res.sd, exp_sd)

    res = RESULTS.Tcontrast([1, 0], store=("effect",))

    assert res.t is None
    assert_array_almost_equal(res.effect, exp_effect)
    assert res.sd is None

    res = RESULTS.Tcontrast([1, 0], store=("t",))

    assert_array_almost_equal(res.t, exp_t)
    assert res.effect is None
    assert res.sd is None

    res = RESULTS.Tcontrast([1, 0], store=("sd",))

    assert res.t is None
    assert res.effect is None
    assert_array_almost_equal(res.sd, exp_sd)

    res = RESULTS.Tcontrast([1, 0], store=("effect", "sd"))

    assert res.t is None
    assert_array_almost_equal(res.effect, exp_effect)
    assert_array_almost_equal(res.sd, exp_sd)


def test_f_output():
    # Test f_output
    res = RESULTS.Fcontrast([1, 0])
    exp_f = RESULTS.t(0) ** 2

    assert_array_almost_equal(exp_f, res.F)

    # Test arrays work as well as lists
    res = RESULTS.Fcontrast(np.array([1, 0]))

    assert_array_almost_equal(exp_f, res.F)

    # Test with matrix against R
    res = RESULTS.Fcontrast(np.eye(2))

    assert_array_almost_equal(31.06, res.F, 2)


def test_f_output_errors():
    # Input matrix checked for size
    match = (
        r"F contrasts should have shape\[1\]=.*, but this has shape\[1\]=.*"
    )
    with pytest.raises(ValueError, match=match):
        RESULTS.Fcontrast([1])
    with pytest.raises(ValueError, match=match):
        RESULTS.Fcontrast([1, 0, 0])
    # And shape
    with pytest.raises(ValueError, match=match):
        RESULTS.Fcontrast(np.array([1, 0])[:, None])


def test_f_output_new_api():
    res = RESULTS.Fcontrast([1, 0])

    assert_array_almost_equal(res.effect, RESULTS.theta[0])
    assert_array_almost_equal(res.covariance, RESULTS.vcov()[0][0])


def test_conf_int():
    lower_, upper_ = RESULTS.conf_int()

    assert (lower_ < upper_).all()
    assert (lower_ > upper_ - 10).all()

    lower_, upper_ = RESULTS.conf_int(cols=[1]).T

    assert lower_ < upper_
    assert lower_ > upper_ - 10
