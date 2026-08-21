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

# Unlike ``X`` above, the columns of this design matrix are correlated,
# so the covariance matrix of the estimates is not diagonal.
X_CORRELATED = np.c_[np.arange(N, dtype=float), np.ones((N,))]
# Several columns of data, each with a different variance: one with more
# columns of data than regressors, one with as many as there are regressors.
Y_3_COLUMNS = np.c_[Y, 2.0 * Y, 3.0 * Y]
Y_2_COLUMNS = np.c_[Y, 2.0 * Y]
RESULTS_3_COLUMNS = OLSModel(X_CORRELATED).fit(Y_3_COLUMNS)
RESULTS_2_COLUMNS = OLSModel(X_CORRELATED).fit(Y_2_COLUMNS)
# Columns of data that are not multiples of one another, so that no
# per-column quantity is forced to agree with any other by construction.
Y_3_COLUMNS_UNRELATED = np.c_[
    Y,
    np.r_[3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
    np.r_[2, 7, 1, 8, 2, 8, 1, 8, 2, 8],
]
RESULTS_3_UNRELATED = OLSModel(X_CORRELATED).fit(Y_3_COLUMNS_UNRELATED)


def test_model():
    """Test basics about the model fit, checking against R results."""
    # Check we fit the mean
    assert_array_almost_equal(RESULTS.theta[1], np.mean(Y))
    # Check we get the same as R
    assert_array_almost_equal(RESULTS.theta, [1.773, 2.5], 3)
    percentile = np.percentile
    pcts = percentile(RESULTS.residuals, [0, 25, 50, 75, 100])
    assert_array_almost_equal(pcts, [-1.6970, -0.6667, 0, 0.6667, 1.6970], 4)


def test_t_contrast():
    """Test individual t-values and t-contrasts against R."""
    assert_array_almost_equal(RESULTS.t(0), 3.25)
    assert_array_almost_equal(RESULTS.t(1), 7.181, 3)
    # And contrast
    assert_array_almost_equal(RESULTS.Tcontrast([1, 0]).t, 3.25)
    assert_array_almost_equal(RESULTS.Tcontrast([0, 1]).t, 7.181, 3)


def test_t_contrast_errors():
    """Test that malformed t-contrasts warn or raise as expected."""
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
    """Test that Tcontrast only returns the requested outputs."""
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
    """Test Fcontrast with a list, an array, and a matrix, against R."""
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
    """Test that malformed F-contrasts raise a ValueError."""
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
    """Test that Fcontrast exposes effect and covariance attributes."""
    res = RESULTS.Fcontrast([1, 0])

    assert_array_almost_equal(res.effect, RESULTS.theta[0])
    assert_array_almost_equal(res.covariance, RESULTS.vcov()[0][0])


def test_conf_int():
    """Test that conf_int returns consistent lower and upper bounds."""
    lower_, upper_ = RESULTS.conf_int()

    assert (lower_ < upper_).all()
    assert (lower_ > upper_ - 10).all()

    lower_, upper_ = RESULTS.conf_int(cols=[1]).T

    assert lower_ < upper_
    assert lower_ > upper_ - 10


@pytest.mark.parametrize("results", [RESULTS_3_COLUMNS, RESULTS_2_COLUMNS])
@pytest.mark.parametrize(
    "call",
    [
        lambda results: results.vcov(),
        lambda results: results.vcov(column=[0, 1]),
        # vcov runs its column branch whenever column is given, whether
        # or not matrix is given with it, so the guard must catch this
        # spelling too instead of taking the matrix escape.
        lambda results: results.vcov(matrix=np.eye(2), column=[0, 1]),
        # A row or a column vector overlaps the covariance matrix just
        # as a 1-D array does; only axes of the dispersion's own, as in
        # dispersion[:, None, None], stack instead of smearing.
        lambda results: results.vcov(
            dispersion=np.asarray(results.dispersion)[None, :]
        ),
        lambda results: results.vcov(
            dispersion=np.asarray(results.dispersion)[:, None]
        ),
    ],
    ids=[
        "vcov",
        "vcov_column_sequence",
        "vcov_column_wins_over_matrix",
        "vcov_dispersion_row",
        "vcov_dispersion_column",
    ],
)
@pytest.mark.ai_generated
def test_vcov_several_dispersions_error(call, results):
    """Test that vcov rejects calls that would need a single matrix."""
    with pytest.raises(ValueError, match="one covariance matrix per disp"):
        call(results)


@pytest.mark.ai_generated
def test_vcov_several_dispersions_error_when_passed_explicitly():
    """Test that a dispersion per value is rejected on 1-D data too."""
    with pytest.raises(ValueError, match="one covariance matrix per disp"):
        RESULTS.vcov(dispersion=np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_3_COLUMNS, 3), (RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_dispersion_shaped_to_stack_is_kept(results, n_columns):
    """Test that a dispersion reshaped to add its own axes still works."""
    # A dispersion shaped (n_columns, 1, 1) broadcasts onto axes of its
    # own rather than along the columns of the covariance matrix, so it
    # gives one matrix per column of data and must not be rejected.
    dispersion = np.asarray(results.dispersion)[:, None, None]

    stacked = results.vcov(dispersion=dispersion)

    assert stacked.shape == (n_columns, 2, 2)
    assert_array_almost_equal(
        stacked, [results.cov * d for d in results.dispersion]
    )


@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_3_COLUMNS, 3), (RESULTS_2_COLUMNS, 2)],
)
@pytest.mark.ai_generated
def test_vcov_several_columns_of_data_still_supported(results, n_columns):
    """Test that vcov calls with one matrix per column of data still work."""
    assert results.vcov(column=0).shape == (n_columns,)
    assert results.t(0).shape == (n_columns,)
    assert results.conf_int(cols=[0, 1]).shape == (2, 2, n_columns)
    assert results.vcov(matrix=np.eye(2)).shape == (2, 2, n_columns)
    # Any way of asking for a single regressor is a one by one
    # covariance, so it is still one number per column of data, and t
    # has to agree whichever way that regressor is named.
    for one_regressor in ([0], (0,), [True, False]):
        assert_array_almost_equal(
            np.ravel(results.vcov(column=one_regressor)),
            results.vcov(column=0),
        )
        assert_array_almost_equal(
            np.ravel(results.t(one_regressor)), results.t(0)
        )


@pytest.mark.parametrize(
    ("results", "n_columns"),
    [
        (RESULTS_3_COLUMNS, 3),
        (RESULTS_2_COLUMNS, 2),
        # A 2-D Y with exactly one column: on main this is a silent
        # wrong-number case for t and conf_int, not a shape error.
        (OLSModel(X_CORRELATED).fit(Y[:, None]), 1),
    ],
)
@pytest.mark.ai_generated
def test_t_and_conf_int_keep_the_column_of_data_axis(results, n_columns):
    """Test that t and conf_int stack their per regressor values."""
    n_regressors = results.theta.shape[0]

    assert results.t().shape == (n_regressors, n_columns)
    assert results.conf_int().shape == (n_regressors, 2, n_columns)

    # Asking for every regressor at once has to give back what asking
    # for them one at a time gives, which is what already worked.
    assert_array_almost_equal(
        results.t(), [results.t(i) for i in range(n_regressors)]
    )
    assert_array_almost_equal(
        results.conf_int(),
        np.concatenate(
            [results.conf_int(cols=[i]) for i in range(n_regressors)]
        ),
    )


@pytest.mark.ai_generated
def test_conf_int_boolean_mask_selects_like_t():
    """Test that conf_int reads a boolean mask the way t does.

    Iterating a mask raw hands 0-d booleans to vcov, which index a
    block of the covariance matrix instead of one element of it.
    """
    mask = np.array([True, False])

    assert_array_almost_equal(
        RESULTS_2_COLUMNS.conf_int(cols=mask),
        RESULTS_2_COLUMNS.conf_int(cols=[0]),
    )
    assert_array_almost_equal(
        RESULTS_2_COLUMNS.t(mask), RESULTS_2_COLUMNS.t(0)[np.newaxis]
    )


@pytest.mark.ai_generated
def test_two_dimensional_selectors_raise():
    """Test that selectors with more than one dimension are refused.

    Iterating one would hand whole rows to vcov, which selects blocks
    of the covariance matrix, not single regressors.
    """
    with pytest.raises(ValueError, match="1-D"):
        RESULTS_2_COLUMNS.t(np.array([[0, 1]]))
    with pytest.raises(ValueError, match="1-D"):
        RESULTS_2_COLUMNS.conf_int(cols=np.array([[0, 1]]))


@pytest.mark.ai_generated
def test_conf_int_cols_iterables_and_empty_keep_working():
    """Test that generators, sets and empty cols work as they did."""
    expected = RESULTS_3_COLUMNS.conf_int(cols=[0, 1])

    assert_array_almost_equal(
        RESULTS_3_COLUMNS.conf_int(cols=(i for i in (0, 1))), expected
    )
    assert_array_almost_equal(
        RESULTS_3_COLUMNS.conf_int(cols={0, 1}), expected
    )
    assert_array_almost_equal(
        RESULTS_3_COLUMNS.conf_int(cols=(0, 1)), expected
    )

    assert RESULTS_3_COLUMNS.conf_int(cols=[]).shape == (0,)


@pytest.mark.ai_generated
def test_conf_int_dispersion_as_list_matches_array():
    """Test that a dispersion passed as a python list acts as an array."""
    dispersion = [1.0, 2.0, 3.0]

    assert_array_almost_equal(
        RESULTS_3_COLUMNS.conf_int(dispersion=dispersion),
        RESULTS_3_COLUMNS.conf_int(dispersion=np.asarray(dispersion)),
    )


@pytest.mark.parametrize(
    "results",
    [RESULTS_3_COLUMNS, OLSModel(X_CORRELATED[:, :1]).fit(Y_3_COLUMNS)],
    ids=["two_regressors", "one_regressor"],
)
@pytest.mark.ai_generated
def test_t_identical_across_proportional_columns_of_data(results):
    """Test t on columns of data that are multiples of one another."""
    # Y_3_COLUMNS holds Y, 2 * Y and 3 * Y. A common factor on a column
    # of data cancels out of its t statistics, so all three columns have
    # to give the same value for every regressor.
    for row in results.t():
        assert_array_almost_equal(row, np.repeat(row[0], 3))


@pytest.mark.parametrize(
    "results",
    [RESULTS, RESULTS_3_COLUMNS, RESULTS_2_COLUMNS, RESULTS_3_UNRELATED],
    ids=["one_dimensional", "three_columns", "two_columns", "unrelated"],
)
@pytest.mark.ai_generated
def test_t_agrees_with_tcontrast(results):
    """Test t against Tcontrast, which reaches the same statistic."""
    # Tcontrast does not go through the code this fix touches, so it is
    # an independent check, and one that does not rely on the columns of
    # data being related to one another.
    n_regressors = results.theta.shape[0]

    assert_array_almost_equal(
        results.t(),
        [
            results.Tcontrast(np.eye(n_regressors)[i]).t
            for i in range(n_regressors)
        ],
    )


@pytest.mark.ai_generated
def test_t_and_conf_int_unchanged_on_one_dimensional_data_no_dispersion():
    """Test 1-D data calls with no dispersion argument stay identical.

    An explicit dispersion argument keeps its own axis even on 1-D
    data, so the pin is scoped to the calls that must not move.
    """
    assert RESULTS.t().shape == (2,)
    assert RESULTS.conf_int().shape == (2, 2)
    assert_array_almost_equal(RESULTS.t(), [RESULTS.t(0), RESULTS.t(1)])


@pytest.mark.ai_generated
def test_vcov_single_regressor_several_columns_of_data():
    """Test that a one regressor model still scales by every dispersion."""
    results = OLSModel(X_CORRELATED[:, :1]).fit(Y_3_COLUMNS)

    assert_array_almost_equal(
        np.ravel(results.vcov()),
        results.cov[0, 0] * results.dispersion,
    )
    # vcov is allowed through on this model, so every method that reads
    # it has to be right on it as well.
    assert results.t().shape == (1, 3)
    assert_array_almost_equal(np.ravel(results.t()), results.t(0))
    assert_array_almost_equal(results.conf_int(), results.conf_int(cols=[0]))
