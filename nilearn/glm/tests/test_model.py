"""Testing models module."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nilearn.glm import OLSModel
from nilearn.glm.contrasts import compute_contrast

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


@pytest.mark.ai_generated
def test_f_output_new_api():
    """Test that Fcontrast exposes effect and covariance attributes."""
    res = RESULTS.Fcontrast([1, 0])

    assert_array_almost_equal(res.effect, RESULTS.theta[0])
    assert_array_almost_equal(
        res.covariance, RESULTS.vcov(uniform=False)[0][0]
    )


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
        lambda results: results.vcov(uniform=False),
        lambda results: results.vcov(column=[0, 1], uniform=False),
        # vcov runs its column branch whenever column is given, whether
        # or not matrix is given with it, so the guard must catch this
        # spelling too instead of taking the matrix escape.
        lambda results: results.vcov(
            matrix=np.eye(2), column=[0, 1], uniform=False
        ),
        # A row or a column vector overlaps the covariance matrix just
        # as a 1-D array does; only axes of the dispersion's own, as in
        # dispersion[:, None, None], stack instead of smearing.
        lambda results: results.vcov(
            dispersion=np.asarray(results.dispersion)[None, :], uniform=False
        ),
        lambda results: results.vcov(
            dispersion=np.asarray(results.dispersion)[:, None], uniform=False
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
        RESULTS.vcov(dispersion=np.array([1.0, 2.0]), uniform=False)


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

    stacked = results.vcov(dispersion=dispersion, uniform=False)

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
    assert results.vcov(column=0, uniform=False).shape == (n_columns,)
    assert results.t(0).shape == (n_columns,)
    assert results.conf_int(cols=[0, 1]).shape == (2, 2, n_columns)
    assert results.vcov(matrix=np.eye(2), uniform=False).shape == (
        2,
        2,
        n_columns,
    )
    # Any way of asking for a single regressor is a one by one
    # covariance, so it is still one number per column of data, and t
    # has to agree whichever way that regressor is named.
    for one_regressor in ([0], (0,), [True, False]):
        assert_array_almost_equal(
            np.ravel(results.vcov(column=one_regressor, uniform=False)),
            results.vcov(column=0, uniform=False),
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
        np.ravel(results.vcov(uniform=False)),
        results.cov[0, 0] * results.dispersion,
    )
    # vcov is allowed through on this model, so every method that reads
    # it has to be right on it as well.
    assert results.t().shape == (1, 3)
    assert_array_almost_equal(np.ravel(results.t()), results.t(0))
    assert_array_almost_equal(results.conf_int(), results.conf_int(cols=[0]))


# One entry per way of naming a selection, with the number of
# regressors it selects. ``dim`` is that number, so the uniform shape
# is (n_dispersion, dim, dim) for every row.
UNIFORM_SELECTIONS = [
    ({}, 2),
    ({"column": 0}, 1),
    ({"column": [0]}, 1),
    ({"column": (0, 1)}, 2),
    ({"column": [0, 1]}, 2),
    ({"column": np.array([True, False])}, 1),
    ({"matrix": np.eye(2)}, 2),
    ({"matrix": np.array([[1.0, 0.0]])}, 1),
    # column takes precedence over matrix, so this selects two
    # regressors through the column branch.
    ({"matrix": np.eye(2), "column": [0, 1]}, 2),
]
UNIFORM_SELECTION_IDS = [
    "no_selection",
    "single_integer",
    "single_integer_sequence",
    "tuple",
    "integer_sequence",
    "boolean_mask",
    "matrix_two_rows",
    "matrix_one_row",
    "column_wins_over_matrix",
]


@pytest.mark.parametrize(
    ("results", "n_dispersion"),
    [(RESULTS, 1), (RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
    ids=["one_dimensional", "two_columns", "three_columns"],
)
@pytest.mark.parametrize(
    ("kwargs", "dim"), UNIFORM_SELECTIONS, ids=UNIFORM_SELECTION_IDS
)
@pytest.mark.ai_generated
def test_vcov_uniform_shape_does_not_depend_on_the_arguments(
    results, n_dispersion, kwargs, dim
):
    """Test that uniform=True gives (n_dispersion, dim, dim) every time."""
    assert results.vcov(uniform=True, **kwargs).shape == (
        n_dispersion,
        dim,
        dim,
    )


@pytest.mark.parametrize(
    "results",
    [RESULTS, RESULTS_2_COLUMNS, RESULTS_3_UNRELATED],
    ids=["one_dimensional", "two_columns", "three_columns"],
)
@pytest.mark.parametrize(
    ("kwargs", "dim"), UNIFORM_SELECTIONS, ids=UNIFORM_SELECTION_IDS
)
@pytest.mark.ai_generated
def test_vcov_uniform_agrees_with_one_dispersion_at_a_time(
    results, kwargs, dim
):
    """Test the stack against the shape that was already correct.

    Asking for one scalar dispersion at a time is the call that never
    had to choose between a matrix and a per-column axis, so it is the
    oracle each matrix of the stack is compared against.
    """
    stack = results.vcov(uniform=True, **kwargs)

    for i, dispersion in enumerate(np.ravel(results.dispersion)):
        one_at_a_time = results.vcov(
            dispersion=dispersion, uniform=False, **kwargs
        )
        assert_array_almost_equal(
            stack[i], np.reshape(one_at_a_time, (dim, dim))
        )


@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_3_COLUMNS, 3), (RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_uniform_matches_the_reshaped_dispersion_stack(
    results, n_columns
):
    """Test that uniform=True returns what the reshape already returned.

    ``dispersion[:, None, None]`` is the spelling that gives one matrix
    per column of data today, so the new contract has to agree with it
    rather than introduce a third answer.
    """
    reshaped = results.vcov(
        dispersion=np.asarray(results.dispersion)[:, None, None],
        uniform=False,
    )

    assert reshaped.shape == (n_columns, 2, 2)
    assert_array_almost_equal(results.vcov(uniform=True), reshaped)


@pytest.mark.parametrize(
    "results",
    [RESULTS_2_COLUMNS, RESULTS_3_UNRELATED],
    ids=["two_columns", "three_columns"],
)
@pytest.mark.ai_generated
def test_vcov_uniform_matrix_branch_only_moves_the_dispersion_axis(results):
    """Test that the matrix branch keeps its numbers and moves one axis.

    The matrix branch already returns one matrix per column of data,
    with the per-column axis last. Uniform puts that axis first, and
    nothing else about the result changes.
    """
    legacy = results.vcov(matrix=np.eye(2), uniform=False)

    uniform = results.vcov(matrix=np.eye(2), uniform=True)

    assert legacy.shape[-1] == uniform.shape[0]
    assert np.array_equal(uniform, np.moveaxis(legacy, -1, 0))


@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_3_COLUMNS, 3), (RESULTS_2_COLUMNS, 2)],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"column": [0, 1]},
        {"matrix": np.eye(2), "column": [0, 1]},
    ],
    ids=["no_selection", "integer_sequence", "column_wins_over_matrix"],
)
@pytest.mark.ai_generated
def test_vcov_uniform_answers_what_the_guard_rejects(
    results, n_columns, kwargs
):
    """Test that the calls with no single-matrix answer have a stack one."""
    with pytest.raises(ValueError, match="one covariance matrix per disp"):
        results.vcov(uniform=False, **kwargs)

    stack = results.vcov(uniform=True, **kwargs)

    assert stack.shape == (n_columns, 2, 2)
    assert_array_almost_equal(
        stack, [results.cov * d for d in results.dispersion]
    )


@pytest.mark.ai_generated
def test_vcov_uniform_stacks_an_explicit_dispersion_on_one_dimensional_data():
    """Test that an explicit list of dispersions stacks on a 1-D fit too."""
    dispersion = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="one covariance matrix per disp"):
        RESULTS.vcov(dispersion=dispersion, uniform=False)

    stack = RESULTS.vcov(dispersion=dispersion, uniform=True)

    assert stack.shape == (2, 2, 2)
    assert_array_almost_equal(stack[0], RESULTS.cov)
    assert_array_almost_equal(stack[1], RESULTS.cov * 2.0)


@pytest.mark.parametrize(
    "column",
    [
        np.array([[0, 1]]),
        np.arange(2).reshape(2, 1),
        # numpy reads a 0-d boolean as a mask over the whole array
        # rather than as one regressor.
        np.array(True),
    ],
    ids=["two_dimensional_row", "two_dimensional_column", "zero_d_boolean"],
)
@pytest.mark.ai_generated
def test_vcov_uniform_rejects_selectors_that_are_not_one_regressor_each(
    column,
):
    """Test that a selector which is not a list of regressors raises."""
    with pytest.raises(ValueError, match="column must be an integer"):
        RESULTS.vcov(column=column, uniform=True)


@pytest.mark.parametrize(
    ("results", "n_dispersion"),
    [(RESULTS, 1), (RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
    ids=["one_dimensional", "two_columns", "three_columns"],
)
@pytest.mark.ai_generated
def test_vcov_default_returns_the_uniform_shape(results, n_dispersion):
    """Test that a call naming no shape gets one matrix per dispersion."""
    n_regressors = results.theta.shape[0]

    assert results.vcov().shape == (n_dispersion, n_regressors, n_regressors)
    assert_array_equal(results.vcov(), results.vcov(uniform=True))
    assert_array_equal(
        results.vcov(column=0), results.vcov(column=0, uniform=True)
    )
    assert_array_equal(
        results.vcov(matrix=np.eye(n_regressors)),
        results.vcov(matrix=np.eye(n_regressors), uniform=True),
    )


@pytest.mark.ai_generated
def test_vcov_default_does_not_warn():
    """Test that the call which used to warn no longer does.

    Only the default is pinned here. ``uniform=False`` is deliberately
    left unpinned, because it is scheduled to warn from 0.15.0 and a
    test asserting silence would have to be deleted to let that
    happen.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        RESULTS.vcov()
        RESULTS.vcov(uniform=True)


@pytest.mark.ai_generated
def test_vcov_none_means_the_default_not_the_deprecated_shapes():
    """Test that the old sentinel lands on the new default.

    The commit that added the keyword documented ``None`` as meaning
    ``False`` and warning. The warning is gone, so reading ``None`` for
    truthiness would now drop the caller into the deprecated branch
    silently instead. The keyword has not shipped, so this reverses a
    docstring rather than a release.
    """
    assert_array_equal(
        RESULTS_3_UNRELATED.vcov(uniform=None),
        RESULTS_3_UNRELATED.vcov(uniform=True),
    )


@pytest.mark.parametrize(
    "results",
    [RESULTS_2_COLUMNS, RESULTS_3_UNRELATED],
    ids=["two_columns", "three_columns"],
)
@pytest.mark.ai_generated
def test_vcov_guard_names_the_escape_that_still_exists(results):
    """Test that the message points at a call the caller can make.

    The guard is only reachable by asking for the older shapes, so
    telling the caller to pass ``uniform=True`` would be telling them
    to undo the keyword they just wrote.
    """
    dispersion = np.asarray(results.dispersion)

    with pytest.raises(ValueError, match="you asked for with uniform=False"):
        results.vcov(dispersion=dispersion, uniform=False)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (
            RESULTS,
            {
                "t": (2,),
                "t_0": (),
                "conf_int": (2, 2),
                "Tcontrast": (),
                "Fcontrast": (1, 1, 1),
            },
        ),
        (
            RESULTS_3_UNRELATED,
            {
                "t": (2, 3),
                "t_0": (3,),
                "conf_int": (2, 2, 3),
                "Tcontrast": (3,),
                "Fcontrast": (1, 1, 3),
            },
        ),
    ],
    ids=["one_dimensional", "three_columns"],
)
@pytest.mark.ai_generated
def test_methods_reading_vcov_keep_their_own_shapes(results, expected):
    """Test that the methods built on vcov are unmoved by the default.

    Each of them asks for ``uniform=False``. A caller that lost that
    keyword would not raise: the uniform stack broadcasts into extra
    axes, so the mistake would be silent and the numbers would still
    look like numbers. These are the shapes that catch it.
    """
    assert results.t().shape == expected["t"]
    assert np.shape(results.t(0)) == expected["t_0"]
    assert results.conf_int().shape == expected["conf_int"]
    assert np.shape(results.Tcontrast([1, 0]).t) == expected["Tcontrast"]
    assert (
        np.shape(results.Fcontrast([1, 0]).covariance) == expected["Fcontrast"]
    )


@pytest.mark.parametrize("dispersion", [None, 1.0])
@pytest.mark.ai_generated
def test_vcov_uniform_rejects_a_matrix_of_rank_3_or_more(dispersion):
    """Test that a higher rank contrast raises rather than returning.

    Such a matrix keeps its extra axes through the product, so it
    cannot give one square block per dispersion value. It used to come
    back with those axes still on it, or fail inside numpy, depending
    on both its shape and the number of dispersion values.
    """
    matrix = np.ones((2, RESULTS_2_COLUMNS.theta.shape[0], 2))
    kwargs = {} if dispersion is None else {"dispersion": dispersion}

    with pytest.raises(ValueError, match="matrix must be 1-D or 2-D"):
        RESULTS_2_COLUMNS.vcov(matrix=matrix, **kwargs)

    # The older shapes are unchanged: they still carry the extra axes.
    legacy = RESULTS_2_COLUMNS.vcov(matrix=matrix, uniform=False, **kwargs)
    assert legacy.ndim > 3


# C: the older shapes survive one release behind the keyword.
@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_uniform_false_still_returns_the_older_shapes(results, n_columns):
    """Test that uniform=False is the one-line migration it promises."""
    assert results.vcov(column=0, uniform=False).shape == (n_columns,)
    assert results.vcov(matrix=np.eye(2), uniform=False).shape == (
        2,
        2,
        n_columns,
    )
    assert results.vcov(column=0, dispersion=1.0, uniform=False).shape == ()


# D: one assertion that is exact rather than almost equal.
@pytest.mark.parametrize(
    "results", [RESULTS, RESULTS_2_COLUMNS, RESULTS_3_UNRELATED]
)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"column": 0}, {"column": [0, 1]}, {"matrix": np.eye(2)}],
    ids=["no_selection", "single_integer", "integer_sequence", "matrix"],
)
@pytest.mark.ai_generated
def test_vcov_uniform_values_are_exact_not_merely_close(results, kwargs):
    """Test the stack against the one-dispersion call bit for bit."""
    stack = results.vcov(uniform=True, **kwargs)

    for i, dispersion in enumerate(np.ravel(results.dispersion)):
        one = results.vcov(dispersion=dispersion, uniform=False, **kwargs)
        assert_array_equal(stack[i], np.reshape(one, stack[i].shape))


# A: the other argument, which only the matrix branch reads.
@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_uniform_reads_other_like_the_matrix_branch(results, n_columns):
    """Test that other is still the right hand side of the product."""
    matrix = np.eye(2)[:1]
    other = np.eye(2)[1:]

    uniform = results.vcov(matrix=matrix, other=other, uniform=True)
    legacy = results.vcov(matrix=matrix, other=other, uniform=False)

    assert uniform.shape == (n_columns, 1, 1)
    assert_array_equal(uniform, np.moveaxis(legacy, -1, 0))


@pytest.mark.ai_generated
def test_vcov_uniform_accepts_a_matrix_object():
    """Test that a contrast passed as an ``np.matrix`` still works.

    ``*`` means matrix multiplication for ``np.matrix``, so the block
    has to be coerced before it is scaled by the dispersions. The
    result is a plain array either way, since a stack of matrices
    cannot be an ``np.matrix``.
    """
    results = RESULTS_3_UNRELATED
    with warnings.catch_warnings():
        # numpy itself discourages the subclass; the point here is only
        # that a caller who still has one does not get a shape error.
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        contrast = np.asmatrix(np.eye(2))

        stack = results.vcov(matrix=contrast, uniform=True)

    assert stack.shape == (3, 2, 2)
    assert_array_equal(stack, results.vcov(matrix=np.eye(2), uniform=True))


# B: a dispersion that arrives with axes of its own.
@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_uniform_ignores_how_the_dispersion_is_shaped(results, n_columns):
    """Test that only the dispersion values matter, not their layout."""
    flat = np.asarray(results.dispersion)
    stack = results.vcov(dispersion=flat, uniform=True)

    assert stack.shape == (n_columns, 2, 2)
    for shaped in (flat[:, None], flat[None, :], flat[:, None, None]):
        assert_array_equal(
            results.vcov(dispersion=shaped, uniform=True), stack
        )


# G: column beats matrix, with a matrix that would give a different answer.
@pytest.mark.parametrize(
    ("results", "n_columns"),
    [(RESULTS_2_COLUMNS, 2), (RESULTS_3_UNRELATED, 3)],
)
@pytest.mark.ai_generated
def test_vcov_uniform_column_wins_over_matrix(results, n_columns):
    """Test the precedence with a matrix of a different width than column."""
    # A one row matrix would give dim 1. column names both regressors, so
    # a result of dim 2 can only have come from the column branch.
    both = results.vcov(matrix=np.eye(2)[:1], column=[0, 1], uniform=True)

    assert both.shape == (n_columns, 2, 2)
    assert_array_equal(both, results.vcov(column=[0, 1], uniform=True))


# K: the matrix branch smear that uniform corrects.
@pytest.mark.ai_generated
def test_vcov_uniform_stacks_where_the_matrix_branch_smeared():
    """Test the case where the dispersion count equals the contrast rows.

    With as many dispersion values as the contrast has rows, and the
    dispersion carrying an axis of its own, the older matrix branch
    broadcast the dispersions along the covariance matrix and returned
    one matrix. Uniform returns one matrix per dispersion value.
    """
    results = RESULTS_2_COLUMNS
    dispersion = np.asarray(results.dispersion)[:, None]

    smeared = results.vcov(
        matrix=np.eye(2), dispersion=dispersion, uniform=False
    )
    stacked = results.vcov(
        matrix=np.eye(2), dispersion=dispersion, uniform=True
    )

    assert smeared.shape == (2, 2, 1)
    assert stacked.shape == (2, 2, 2)
    assert_array_almost_equal(
        stacked, [results.cov * d for d in results.dispersion]
    )


# A three regressor design, so a selector can name something that is not
# a prefix of the regressors.
X_THREE_REGRESSORS = np.c_[
    np.arange(N, dtype=float), np.arange(N, dtype=float) ** 2, np.ones((N,))
]
RESULTS_3_REGRESSORS = OLSModel(X_THREE_REGRESSORS).fit(Y_3_COLUMNS_UNRELATED)


# M2: selectors that are not a prefix of the regressors.
@pytest.mark.parametrize(
    ("column", "wanted"),
    [
        (1, [1]),
        ([2], [2]),
        ([0, 2], [0, 2]),
        ((2, 0), [2, 0]),
        (np.array([False, True, True]), [1, 2]),
    ],
    ids=["bare_1", "sequence_2", "skip_the_middle", "out_of_order", "mask"],
)
@pytest.mark.ai_generated
def test_vcov_uniform_selects_the_regressors_it_was_given(column, wanted):
    """Test that the block is the named regressors, not the first few."""
    results = RESULTS_3_REGRESSORS
    stack = results.vcov(column=column, uniform=True)

    assert stack.shape == (3, len(wanted), len(wanted))
    for i, dispersion in enumerate(results.dispersion):
        for a, ia in enumerate(wanted):
            for b, ib in enumerate(wanted):
                assert stack[i, a, b] == results.cov[ia, ib] * dispersion


# T2: matrix and other of different heights give a block that is not square.
@pytest.mark.ai_generated
def test_vcov_uniform_block_is_not_square_when_other_differs():
    """Test the shape when other has a different number of rows."""
    results = RESULTS_3_REGRESSORS
    matrix = np.eye(3)[:1]
    other = np.eye(3)[:2]

    stack = results.vcov(matrix=matrix, other=other, uniform=True)

    assert stack.shape == (3, 1, 2)
    assert_array_equal(
        stack,
        np.moveaxis(
            results.vcov(matrix=matrix, other=other, uniform=False), -1, 0
        ),
    )


@pytest.mark.ai_generated
def test_vcov_uniform_dim_is_the_block_not_the_regressor_count():
    """Test that a contrast row gives a one by one block.

    Through ``matrix`` the block is the contrast's own dimension, so a
    single row spanning two regressors is still one by one, and the
    ``dim`` in the documented shape is not a count of regressors.
    """
    results = RESULTS_3_UNRELATED
    spanning_two = np.array([[1.0, -1.0]])

    stack = results.vcov(matrix=spanning_two, uniform=True)

    assert stack.shape == (3, 1, 1)
    assert results.vcov(matrix=np.eye(2), uniform=True).shape == (3, 2, 2)


@pytest.mark.ai_generated
def test_compute_contrast_does_not_warn():
    """Test that the contrast path runs clean under -W error.

    It is the only caller of ``vcov`` outside this module. Nothing in
    ``vcov`` warns now, so there is no leak left for this to catch
    today; it is here for the ``uniform=False`` warning scheduled for
    0.15.0, which would reach a user of ``compute_contrast`` who could
    do nothing about it.
    """
    labels = np.zeros(Y_3_COLUMNS_UNRELATED.shape[1])
    results = {0.0: OLSModel(X_CORRELATED).fit(Y_3_COLUMNS_UNRELATED)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        compute_contrast(labels, results, np.array([1.0, 0.0]), stat_type="t")
        compute_contrast(labels, results, np.eye(2), stat_type="F")
