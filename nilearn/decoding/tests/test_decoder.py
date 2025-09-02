"""Test the decoder module.

Order of tests from top to bottom:

- helper functions
- fixtures
- classification
- regression
- multiclass

"""

# ruff: noqa: ARG001

import collections
import numbers
import warnings

import numpy as np
import pytest
from nibabel import save
from numpy.testing import assert_array_almost_equal
from sklearn import clone
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    LassoCV,
    LogisticRegressionCV,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    check_scoring,
    get_scorer,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVR, LinearSVC
from sklearn.utils.estimator_checks import (
    ignore_warnings,
    parametrize_with_checks,
)

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.param_validation import (
    _get_mask_extent,
    check_feature_screening,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _rng
from nilearn.decoding import (
    Decoder,
    DecoderRegressor,
    FREMClassifier,
    FREMRegressor,
)
from nilearn.decoding.decoder import (
    SUPPORTED_ESTIMATORS,
    _BaseDecoder,
    _check_estimator,
    _check_param_grid,
    _parallel_fit,
    _wrap_param_grid,
)
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.maskers import NiftiMasker, SurfaceMasker

N_SAMPLES = 80

ESTIMATOR_REGRESSION = ("ridge", "svr")


ESTIMATORS_TO_CHECK = [
    Decoder(),
    DecoderRegressor(),
    FREMClassifier(),
    FREMRegressor(),
]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def _make_binary_classification_test_data(n_samples=N_SAMPLES, dim=5):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=dim**3,
        scale=3.0,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    X, mask = to_niimgs(X, [dim, dim, dim])
    return X, y, mask


@pytest.fixture()
def rand_x_y(rng):
    X = rng.random((100, 10))
    Y = np.hstack([[-1] * 50, [1] * 50])
    return X, Y


def _make_multiclass_classification_test_data(n_samples=40, dim=5):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=dim**3,
        scale=3.0,
        n_informative=5,
        n_classes=4,
        random_state=42,
    )
    X, mask = to_niimgs(X, [dim, dim, dim])
    return X, y, mask


@pytest.fixture(scope="session")
def tiny_binary_classification_data():
    """Use for testing errors.

    This fixture aims to return a very small data set
    because it will only be used for the tests
    that check error handling like input validation.
    """
    return _make_binary_classification_test_data(n_samples=20)


@pytest.fixture
def binary_classification_data():
    """Use for test where classification is actually performed."""
    return _make_binary_classification_test_data(n_samples=N_SAMPLES)


def _make_regression_test_data(n_samples=N_SAMPLES, dim=5):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=dim**3,
        n_informative=dim,
        noise=1.5,
        bias=1.0,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)
    X, mask = to_niimgs(X, [dim, dim, dim])
    return X, y, mask


@pytest.fixture
def regression_data():
    return _make_regression_test_data(n_samples=N_SAMPLES, dim=5)


@pytest.fixture
def multiclass_data():
    return _make_multiclass_classification_test_data(n_samples=N_SAMPLES)


@pytest.mark.parametrize(
    "regressor, param",
    [
        (RidgeCV(), ["alphas"]),
        (SVR(kernel="linear"), ["C"]),
        (LassoCV(), ["n_alphas"]),
    ],
)
def test_check_param_grid_regression(regressor, param, rng):
    """Test several estimators.

    Each one with its specific regularization parameter.
    """
    X = rng.random((N_SAMPLES, 10))
    Y = rng.random(N_SAMPLES)

    param_grid = _check_param_grid(regressor, X, Y, None)

    assert list(param_grid.keys()) == list(param)


@pytest.mark.parametrize(
    "classifier, param",
    [
        (LogisticRegressionCV(penalty="l1"), ["Cs"]),
        (LogisticRegressionCV(penalty="l2"), ["Cs"]),
        (RidgeClassifierCV(), ["alphas"]),
    ],
)
def test_check_param_grid_classification(rand_x_y, classifier, param):
    """Test several estimators.

    Each one with its specific regularization parameter.
    """
    X, Y = rand_x_y

    param_grid = _check_param_grid(classifier, X, Y, None)

    assert list(param_grid.keys()) == list(param)


@ignore_warnings
@pytest.mark.parametrize(
    "param_grid_input",
    [
        {"C": [1, 10, 100]},
        {"Cs": [1, 10, 100]},
        [{"C": [1, 10, 100]}, {"fit_intercept": [False]}],
    ],
)
def test_check_param_grid_replacement(rand_x_y, param_grid_input):
    X, Y = rand_x_y
    param_to_replace = "C"
    param_replaced = "Cs"
    param_grid_output = _check_param_grid(
        LogisticRegressionCV(),
        X,
        Y,
        param_grid_input,
    )
    for params in ParameterGrid(param_grid_output):
        assert param_to_replace not in params
        if param_replaced not in params:
            assert params in ParameterGrid(param_grid_input)


@pytest.mark.parametrize("estimator", ["log_l1", RandomForestClassifier()])
def test_non_supported_estimator_error(rand_x_y, estimator):
    """Raise the error when using a non supported estimator."""
    X, Y = rand_x_y

    with pytest.raises(
        ValueError, match="Invalid estimator. The supported estimators are:"
    ):
        _check_param_grid(estimator, X, Y, None)


def test_check_parameter_grid_is_empty(rand_x_y):
    X, Y = rand_x_y
    dummy_classifier = DummyClassifier(random_state=0)

    param_grid = _check_param_grid(dummy_classifier, X, Y, None)

    assert param_grid == {}


@ignore_warnings
@pytest.mark.parametrize(
    "param_grid",
    [
        {"alphas": [1, 10, 100, 1000]},
        {"alphas": [1, 10, 100, 1000], "fit_intercept": [True, False]},
        {"fit_intercept": [True, False]},
        {"alphas": [[1, 10, 100, 1000]]},
        {"alphas": (1, 10, 100, 1000)},
        {"alphas": [(1, 10, 100, 1000)]},
        {"alphas": ((1, 10, 100, 1000),)},
        {"alphas": np.array([1, 10, 100, 1000])},
        {"alphas": [np.array([1, 10, 100, 1000])]},
        [{"alphas": [1, 10]}, {"alphas": [[100, 1000]]}],
        [{"alphas": [1, 10]}, {"fit_intercept": [True, False]}],
    ],
)
def test_wrap_param_grid(param_grid):
    param_name = "alphas"
    original_grid = ParameterGrid(param_grid)
    wrapped_grid = ParameterGrid(_wrap_param_grid(param_grid, param_name))
    for grid_row in wrapped_grid:
        if param_name in grid_row:
            param_value = grid_row[param_name]
            assert isinstance(param_value, collections.abc.Iterable)
            assert all(
                isinstance(item, numbers.Number) for item in param_value
            )
        else:
            assert grid_row in original_grid


@pytest.mark.parametrize(
    "param_grid, need_wrap",
    [
        ({"alphas": [1, 10, 100, 1000]}, True),
        ({"alphas": [[1, 10, 100, 1000]]}, False),
    ],
)
def test_wrap_param_grid_warning(param_grid, need_wrap):
    expected_warning_substring = "should be a sequence of iterables"

    if need_wrap:
        with pytest.warns(UserWarning, match=expected_warning_substring):
            _wrap_param_grid(param_grid, param_name="alphas")

    else:
        with warnings.catch_warnings(record=True) as raised_warnings:
            _wrap_param_grid(param_grid, param_name="alphas")
        warning_messages = [
            str(warning.message) for warning in raised_warnings
        ]

        found_warning = any(
            expected_warning_substring in x for x in warning_messages
        )

        assert not found_warning


def test_wrap_param_grid_is_none():
    assert _wrap_param_grid(None, "alphas") is None


@pytest.mark.parametrize(
    "model", [DecoderRegressor, Decoder, FREMRegressor, FREMClassifier]
)
def test_check_inputs_length(model):
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    # Remove ten samples from y
    y = y[:-10]

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model(mask=mask, screening_percentile=100.0).fit(X_, y)


@pytest.mark.parametrize(
    "estimator",
    [
        "svc",
        "svc_l2",
        "svc_l1",
        "logistic",
        "logistic_l1",
        "logistic_l2",
        "ridge",
        "ridge_classifier",
        "ridge_regressor",
        "svr",
        "dummy_classifier",
        "dummy_regressor",
    ],
)
def test_check_supported_estimator(estimator):
    """Check if the estimator is one of the supported estimators."""
    expected_warning = (
        "Use a custom estimator at your own risk "
        "of the process not working as intended."
    )

    with warnings.catch_warnings(record=True) as raised_warnings:
        _check_estimator(_BaseDecoder(estimator=estimator).estimator)
    warning_messages = [str(warning.message) for warning in raised_warnings]

    assert expected_warning not in warning_messages


@pytest.mark.parametrize("estimator", ["ridgo", "svb"])
def test_check_unsupported_estimator(estimator):
    """Check if the estimator is one of the supported estimators.

    If not, if it is a string and if not in supported ones,
    then raise the error.
    """
    with pytest.raises(ValueError, match="Invalid estimator"):
        _check_estimator(_BaseDecoder(estimator=estimator).estimator)

    expected_warning = (
        "Use a custom estimator at your own risk "
        "of the process not working as intended."
    )
    custom_estimator = RandomForestClassifier()
    with pytest.warns(UserWarning, match=expected_warning):
        _check_estimator(_BaseDecoder(estimator=custom_estimator).estimator)


def test_parallel_fit(rand_x_y):
    """Check that results of _parallel_fit is the same \
    for different controlled param_grid.
    """
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=5,
        noise=0.2,
        random_state=42,
    )
    train = range(80)

    _, y_classification = rand_x_y
    test = range(80, len(y_classification))

    estimator = SVR(kernel="linear")

    # define a scorer
    scorer = check_scoring(estimator, "r2")

    # Define a screening selector
    selector = check_feature_screening(
        screening_percentile=None, mask_img=None, is_classification=False
    )

    outputs = []
    for params in [[1e-1, 1e0, 1e1], [1e-1, 1e0, 5e0, 1e1]]:
        param_grid = {"C": np.array(params)}
        outputs.append(
            list(
                _parallel_fit(
                    estimator=SVR(kernel="linear"),
                    X=X,
                    y=y,
                    train=train,
                    test=test,
                    param_grid=param_grid,
                    scorer=scorer,
                    mask_img=None,
                    class_index=1,
                    selector=selector,
                    clustering_percentile=100,
                )
            )
        )

    # check that every element of the output tuple is the same for both tries
    for a, b in zip(outputs[0], outputs[1]):
        if isinstance(a, np.ndarray):
            assert_array_almost_equal(a, b)
        else:
            assert a == b


@ignore_warnings
@pytest.mark.parametrize(
    "param_values",
    (
        [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        [[0.001, 0.01, 0.1, 1, 10, 100, 1000]],
    ),
)
@pytest.mark.parametrize(
    "estimator, param_name, fitted_param_name, is_classification",
    [
        (RidgeCV(), "alphas", "best_alpha", False),
        (RidgeClassifierCV(), "alphas", "best_alpha", True),
        (LogisticRegressionCV(), "Cs", "best_C", True),
        (LassoCV(), "alphas", "best_alpha", False),
    ],
)
def test_parallel_fit_builtin_cv(
    rand_x_y,
    estimator,
    param_name,
    fitted_param_name,
    is_classification,
    param_values,
):
    """Check that the `fitted_param_name` output of _parallel_fit is \
       a single value even if param_grid is wrapped in a list \
       for models with built-in CV.
    """
    # y will be replaced if this is a classification
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=20,
        n_informative=5,
        noise=0.2,
        random_state=42,
    )

    # train/test indices
    n_samples_train = int(0.8 * N_SAMPLES)
    train = range(n_samples_train)
    test = range(n_samples_train, N_SAMPLES)

    # define a screening selector
    selector = check_feature_screening(
        screening_percentile=None, mask_img=None, is_classification=False
    )

    # create appropriate scorer and update y for classification
    if is_classification:
        scorer = check_scoring(estimator, "accuracy")
        _, y = rand_x_y
    else:
        scorer = check_scoring(estimator, "r2")

    param_grid = {param_name: param_values}
    _, _, _, best_param, _, _ = _parallel_fit(
        estimator=estimator,
        X=X,
        y=y,
        train=train,
        test=test,
        param_grid=param_grid,
        scorer=scorer,
        mask_img=None,
        class_index=1,
        selector=selector,
        clustering_percentile=100,
    )

    assert isinstance(best_param[fitted_param_name], numbers.Number)


@ignore_warnings
def test_decoder_param_grid_sequence(binary_classification_data):
    X, y, _ = binary_classification_data
    n_cv_folds = 10
    param_grid = [
        {
            "penalty": ["l2"],
            "C": [100, 1000],
            "random_state": [42],  # fix the seed for consistent behavior
        },
        {
            "penalty": ["l1"],
            "dual": [False],  # "dual" is not in the first dict
            "C": [100, 10],
            "random_state": [42],  # fix the seed for consistent behavior
        },
    ]

    model = Decoder(param_grid=param_grid, cv=n_cv_folds)
    model.fit(X, y)

    for best_params in model.cv_params_.values():
        for param_list in best_params.values():
            assert len(param_list) == n_cv_folds


@ignore_warnings
def test_decoder_binary_classification_with_masker_object(
    binary_classification_data,
):
    X, y, _ = binary_classification_data

    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "roc_auc"
    assert model.score(X, y) == 1.0
    assert accuracy_score(y, y_pred) > 0.95


@ignore_warnings
def test_decoder_binary_classification_with_logistic_model(
    binary_classification_data,
):
    """Check decoder with predict_proba for scoring with logistic model."""
    X, y, mask = binary_classification_data

    model = Decoder(estimator="logistic_l2", mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@ignore_warnings
@pytest.mark.parametrize("screening_percentile", [100, 20, None])
def test_decoder_binary_classification_screening(
    binary_classification_data, screening_percentile
):
    X, y, mask = binary_classification_data

    model = Decoder(mask=mask, screening_percentile=screening_percentile)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@ignore_warnings
@pytest.mark.parametrize("clustering_percentile", [100, 99])
def test_decoder_binary_classification_clustering(
    binary_classification_data, clustering_percentile
):
    X, y, mask = binary_classification_data

    model = FREMClassifier(
        estimator="logistic_l2",
        mask=mask,
        clustering_percentile=clustering_percentile,
        screening_percentile=90,
        cv=5,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.9


@ignore_warnings
@pytest.mark.parametrize("cv", [KFold(n_splits=5), LeaveOneGroupOut()])
def test_decoder_binary_classification_cross_validation(
    binary_classification_data, cv, rng
):
    X, y, mask = binary_classification_data

    # check cross-validation scheme and fit attribute with groups enabled
    model = Decoder(
        estimator="svc", mask=mask, standardize="zscore_sample", cv=cv
    )
    groups = None
    if isinstance(cv, LeaveOneGroupOut):
        groups = rng.binomial(2, 0.3, size=len(y))
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.9


@ignore_warnings
def test_decoder_dummy_classifier(binary_classification_data):
    n_samples = N_SAMPLES
    X, y, mask = binary_classification_data

    # We make 80% of y to have value of 1.0 to check whether the stratified
    # strategy returns a proportion prediction value of 1.0 of roughly 80%
    proportion = 0.8
    y = np.zeros(n_samples)
    y[: int(proportion * n_samples)] = 1.0

    model = Decoder(estimator="dummy_classifier", mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.sum(y_pred == 1.0) / n_samples - proportion < 0.05


@ignore_warnings
def test_decoder_dummy_classifier_with_callable(binary_classification_data):
    X, y, mask = binary_classification_data

    accuracy_scorer = get_scorer("accuracy")
    model = Decoder(
        estimator="dummy_classifier", mask=mask, scoring=accuracy_scorer
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == accuracy_scorer
    assert model.score(X, y) == accuracy_score(y, y_pred)


@ignore_warnings
def test_decoder_dummy_classifier_strategy_prior():
    X, y, mask = _make_binary_classification_test_data(n_samples=300)

    param = {"strategy": "prior"}
    dummy_classifier = DummyClassifier(random_state=0)
    dummy_classifier.set_params(**param)
    model = Decoder(estimator=dummy_classifier, mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.all(y_pred) == 1.0
    assert roc_auc_score(y, y_pred) == 0.5


@ignore_warnings
def test_decoder_dummy_classifier_strategy_most_frequent():
    X, y, mask = _make_binary_classification_test_data(n_samples=300)

    param = {"strategy": "most_frequent"}
    dummy_classifier = DummyClassifier(random_state=0)
    dummy_classifier.set_params(**param)

    model = Decoder(estimator=dummy_classifier, mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.all(y_pred) == 1.0

    # Returns model coefficients for dummy estimators as None
    assert model.coef_ is None
    # Dummy output are nothing but the attributes of the dummy estimators
    assert model.dummy_output_ is not None
    assert model.cv_scores_ is not None


@ignore_warnings
def test_decoder_dummy_classifier_roc_scoring(binary_classification_data):
    X, y, mask = binary_classification_data

    model = Decoder(estimator="dummy_classifier", mask=mask, scoring="roc_auc")
    model.fit(X, y)

    assert np.mean(model.cv_scores_[0]) >= 0.45


@ignore_warnings
def test_decoder_error_not_implemented(tiny_binary_classification_data):
    X, y, mask = tiny_binary_classification_data

    param = {"strategy": "constant"}
    dummy_classifier = DummyClassifier(random_state=0)
    dummy_classifier.set_params(**param)

    model = Decoder(estimator=dummy_classifier, mask=mask)

    with pytest.raises(NotImplementedError):
        model.fit(X, y)


@ignore_warnings
def test_decoder_error_unknown_scoring_metrics(
    tiny_binary_classification_data,
):
    X, y, mask = tiny_binary_classification_data

    dummy_classifier = DummyClassifier(random_state=0)

    model = Decoder(estimator=dummy_classifier, mask=mask, scoring="foo")

    with pytest.raises(
        ValueError,
        match="The 'scoring' parameter of check_scoring must be a str among",
    ):
        model.fit(X, y)


@ignore_warnings
def test_decoder_dummy_classifier_default_scoring():
    X, y, _ = _make_binary_classification_test_data()

    model = Decoder(estimator="dummy_classifier", scoring=None)

    assert model.scoring is None

    model.fit(X, y)

    assert model.scorer_._score_func == get_scorer("accuracy")._score_func
    assert model.scorer_._sign == get_scorer("accuracy")._sign
    assert model.score(X, y) > 0.5


@ignore_warnings
def test_decoder_classification_string_label():
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, [2, 2, 2])
    labels = ["red", "blue", "green"]
    y_str = [labels[y[i]] for i in range(len(y))]

    model = Decoder(mask=mask)
    model.fit(X, y_str)
    y_pred = model.predict(X)

    assert accuracy_score(y_str, y_pred) > 0.95


@ignore_warnings
@pytest.mark.parametrize("screening_percentile", [100, 20, 1, None])
@pytest.mark.parametrize("estimator", ESTIMATOR_REGRESSION)
def test_decoder_regression_screening(
    regression_data, screening_percentile, estimator
):
    X, y, mask = regression_data

    model = DecoderRegressor(
        estimator=estimator,
        mask=mask,
        screening_percentile=screening_percentile,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert r2_score(y, y_pred) > 0.95


@ignore_warnings
@pytest.mark.parametrize("clustering_percentile", [100, 99])
@pytest.mark.parametrize("estimator", ESTIMATOR_REGRESSION)
def test_decoder_regression_clustering(
    regression_data, clustering_percentile, estimator
):
    X, y, mask = regression_data

    model = FREMRegressor(
        estimator=estimator,
        mask=mask,
        clustering_percentile=clustering_percentile,
        screening_percentile=90,
        cv=10,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "r2"
    assert r2_score(y, y_pred) > 0.95
    assert model.score(X, y) == r2_score(y, y_pred)


@ignore_warnings
def test_decoder_dummy_regression(regression_data):
    X, y, mask = regression_data

    # Regression with dummy estimator
    model = DecoderRegressor(
        estimator="dummy_regressor",
        mask=mask,
        scoring="r2",
        screening_percentile=1,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "r2"
    assert r2_score(y, y_pred) <= 0.0
    assert model.score(X, y) == r2_score(y, y_pred)


@ignore_warnings
def test_decoder_dummy_regression_default_scoring_metric_is_r2(
    regression_data,
):
    """Check that default scoring metric for regression is r2."""
    X, y, mask = regression_data

    model = DecoderRegressor(
        estimator="dummy_regressor", mask=mask, scoring=None
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.score(X, y) == r2_score(y, y_pred)


@ignore_warnings
def test_decoder_dummy_regression_other_strategy(regression_data):
    """Chexk that decoder object use other strategy for dummy regressor."""
    X, y, mask = regression_data

    dummy_regressor = DummyRegressor()
    param = {"strategy": "median"}
    dummy_regressor.set_params(**param)

    model = DecoderRegressor(estimator=dummy_regressor, mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert r2_score(y, y_pred) <= 0.0
    # Returns model coefficients for dummy estimators as None
    assert model.coef_ is None
    # Dummy output are nothing but the attributes of the dummy estimators
    assert model.dummy_output_ is not None
    assert model.cv_scores_ is not None


@ignore_warnings
def test_decoder_multiclass_classification_masker(multiclass_data):
    X, y, _ = multiclass_data

    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@ignore_warnings
def test_decoder_multiclass_classification_masker_dummy_classifier(
    multiclass_data,
):
    X, y, _ = multiclass_data

    model = Decoder(
        estimator="dummy_classifier", mask=NiftiMasker(), scoring="accuracy"
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "accuracy"
    # 4-class classification
    assert accuracy_score(y, y_pred) > 0.2
    assert model.score(X, y) == accuracy_score(y, y_pred)


@ignore_warnings
@pytest.mark.parametrize("screening_percentile", [100, 20, None])
def test_decoder_multiclass_classification_screening(
    multiclass_data, screening_percentile
):
    X, y, mask = multiclass_data

    model = Decoder(mask=mask, screening_percentile=screening_percentile)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@ignore_warnings
@pytest.mark.parametrize("clustering_percentile", [100, 99])
@pytest.mark.parametrize("estimator", ["svc_l2", "svc_l1"])
def test_decoder_multiclass_classification_clustering(
    multiclass_data, clustering_percentile, estimator
):
    X, y, mask = multiclass_data

    model = FREMClassifier(
        estimator=estimator,
        mask=mask,
        clustering_percentile=clustering_percentile,
        screening_percentile=90,
        cv=5,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "roc_auc"
    assert accuracy_score(y, y_pred) > 0.9


@ignore_warnings
@pytest.mark.parametrize("cv", [KFold(n_splits=5), LeaveOneGroupOut()])
def test_decoder_multiclass_classification_cross_validation(
    multiclass_data, cv
):
    X, y, mask = multiclass_data

    # check cross-validation scheme and fit attribute with groups enabled
    model = Decoder(
        estimator="svc", mask=mask, standardize="zscore_sample", cv=cv
    )
    groups = None
    if isinstance(cv, LeaveOneGroupOut):
        groups = _rng(0).binomial(2, 0.3, size=len(y))
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.9


def test_decoder_multiclass_classification_apply_mask_shape():
    """Test whether if _apply mask output has the same shape \
    as original matrix.
    """
    dim = 5
    X_init, _ = make_classification(
        n_samples=200,
        n_features=dim**3,
        scale=3.0,
        n_informative=5,
        n_classes=4,
        random_state=42,
    )
    X, _ = to_niimgs(X_init, [dim, dim, dim])

    model = Decoder(mask=NiftiMasker())

    X_masked = model._apply_mask(X)

    assert X_masked.shape == X_init.shape


def test_decoder_multiclass_classification_apply_mask_attributes(affine_eye):
    """Test whether model.masker_ have some desire attributes \
    manually set after calling _apply_mask.

    By default these parameters are set to None;
    """
    X, _, _ = _make_multiclass_classification_test_data()

    target_affine = 2 * affine_eye
    target_shape = (1, 1, 1)
    t_r = 1
    high_pass = 1
    low_pass = 2
    smoothing_fwhm = 0.5

    model = Decoder(
        target_affine=target_affine,
        target_shape=target_shape,
        t_r=t_r,
        high_pass=high_pass,
        low_pass=low_pass,
        smoothing_fwhm=smoothing_fwhm,
    )

    model._apply_mask(X)

    assert np.any(model.masker_.target_affine == target_affine)
    assert model.masker_.target_shape == target_shape
    assert model.masker_.t_r == t_r
    assert model.masker_.high_pass == high_pass
    assert model.masker_.low_pass == low_pass
    assert model.masker_.smoothing_fwhm == smoothing_fwhm


def test_decoder_multiclass_error_incorrect_cv(multiclass_data):
    """Check whether ValueError is raised when cv is not set correctly."""
    X, y, _ = multiclass_data

    for cv in ["abc", LinearSVC(dual=True)]:
        model = Decoder(mask=NiftiMasker(), cv=cv)
        with pytest.raises(ValueError, match="Expected cv as an integer"):
            model.fit(X, y)


@ignore_warnings
def test_decoder_multiclass_warnings(multiclass_data):
    X, y, _ = multiclass_data
    groups = _rng(0).binomial(2, 0.3, size=len(y))

    # Check whether decoder raised warning when groups is set to specific
    # value but CV Splitter is not set
    expected_warning = (
        "groups parameter is specified but "
        "cv parameter is not set to custom CV splitter. "
        "Using default object LeaveOneGroupOut()."
    )
    with pytest.warns(UserWarning, match=expected_warning):
        model = Decoder(mask=NiftiMasker())
        model.fit(X, y, groups=groups)

    # Check that warning is raised when n_features is lower than 50 after
    # screening and clustering for FREM
    with pytest.warns(
        UserWarning, match=".*decoding model will be trained only.*"
    ):
        model = FREMClassifier(
            clustering_percentile=10,
            screening_percentile=10,
            mask=NiftiMasker(),
            cv=1,
        )
        model.fit(X, y)


def test_decoder_tags_classification():
    """Check value returned by _more_tags."""
    model = Decoder()
    # TODO (sklearn >= 1.6.0) remove if block
    if SKLEARN_LT_1_6:
        assert model.__sklearn_tags__()["require_y"] is True
    else:
        assert model.__sklearn_tags__().target_tags.required is True


def test_decoder_tags_regression():
    """Check value returned by _more_tags."""
    model = DecoderRegressor()
    # remove if block when bumping sklearn_version to > 1.5
    if SKLEARN_LT_1_6:
        assert model.__sklearn_tags__()["multioutput"] is True
    else:
        assert model.__sklearn_tags__().target_tags.multi_output is True


@ignore_warnings
@pytest.mark.timeout(0)
def test_decoder_strings_filepaths_input(
    tiny_binary_classification_data, tmp_path
):
    """Smoke test for decoder methods to accept list of paths as input.

    See https://github.com/nilearn/nilearn/issues/4226
    """
    X, y, _ = tiny_binary_classification_data
    X_paths = [tmp_path / f"niimg{i}.nii" for i in range(X.shape[-1])]
    for i, nii_path in enumerate(X_paths):
        save(X.slicer[..., i], nii_path)

    model = Decoder(mask=NiftiMasker())
    model.fit(X_paths, y)
    model.predict(X_paths)
    model.score(X_paths, y)


# ------------------------ surface tests ------------------------------------ #


@pytest.fixture()
def _make_surface_class_data(rng, surf_img_2d, n_samples=50):
    """Create a surface image classification for testing."""
    y = rng.choice([0, 1], size=n_samples)
    return surf_img_2d(n_samples), y


@pytest.fixture()
def _make_surface_reg_data(rng, surf_img_2d, n_samples=50):
    """Create a surface image regression for testing."""
    y = rng.random(n_samples)
    return surf_img_2d(n_samples), y


def test_decoder_apply_mask_surface(_make_surface_class_data):
    """Test _apply_mask on surface image."""
    X, _ = _make_surface_class_data
    model = Decoder(mask=SurfaceMasker())
    X_masked = model._apply_mask(X)

    assert X_masked.shape == X.shape[::-1]
    assert type(model.mask_img_).__name__ == "SurfaceImage"


@ignore_warnings
def test_decoder_screening_percentile_surface_default(
    _make_surface_class_data,
):
    """Test default screening percentile with surface image."""
    X, y = _make_surface_class_data

    model = Decoder(mask=SurfaceMasker())
    model.fit(X, y)
    assert model.screening_percentile_ == 20


@ignore_warnings
@pytest.mark.parametrize("perc", [None, 100, 0])
def test_decoder_screening_percentile_surface(perc, _make_surface_class_data):
    """Test passing screening percentile with surface image."""
    X, y = _make_surface_class_data

    model = Decoder(mask=SurfaceMasker(), screening_percentile=perc)
    model.fit(X, y)
    if perc is None:
        assert model.screening_percentile_ == 100
    else:
        assert model.screening_percentile_ == perc


@ignore_warnings
@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_decoder_adjust_screening_less_than_mask_surface(
    surf_mask_dim,
    surf_mask_1d,
    surf_mask_2d,
    _make_surface_class_data,
    screening_percentile=30,
):
    """When mask size is less than or equal to screening percentile wrt to
    the mesh size, it is adjusted to the ratio of mesh to mask.
    """
    img, y = _make_surface_class_data
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    mask_n_vertices = _get_mask_extent(surf_mask)
    mesh_n_vertices = img.mesh.n_vertices
    mask_to_mesh_ratio = (mask_n_vertices / mesh_n_vertices) * 100
    assert screening_percentile <= mask_to_mesh_ratio
    decoder = Decoder(
        mask=surf_mask,
        param_grid={"C": [0.01, 0.1]},
        cv=3,
        screening_percentile=screening_percentile,
    )
    decoder.fit(img, y)
    adjusted = decoder.screening_percentile_
    assert adjusted == screening_percentile * (
        mesh_n_vertices / mask_n_vertices
    )


@ignore_warnings
@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_decoder_adjust_screening_greater_than_mask_surface(
    surf_mask_dim,
    surf_mask_1d,
    surf_mask_2d,
    _make_surface_class_data,
    screening_percentile=80,
):
    """When mask size is greater than screening percentile wrt to the mesh
    size, it is changed to 100% of mask.
    """
    img, y = _make_surface_class_data
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    mask_n_vertices = _get_mask_extent(surf_mask)
    mesh_n_vertices = img.mesh.n_vertices
    mask_to_mesh_ratio = (mask_n_vertices / mesh_n_vertices) * 100
    assert screening_percentile > mask_to_mesh_ratio
    decoder = Decoder(
        mask=surf_mask_1d,
        param_grid={"C": [0.01, 0.1]},
        cv=3,
        screening_percentile=screening_percentile,
    )
    decoder.fit(img, y)
    adjusted = decoder.screening_percentile_
    assert adjusted == 100


@ignore_warnings
def test_decoder_predict_score_surface(_make_surface_class_data):
    """Test classification predict and scoring for surface image."""
    X, y = _make_surface_class_data
    model = Decoder(mask=SurfaceMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "roc_auc"

    model.score(X, y)
    acc = accuracy_score(y, y_pred)
    assert 0.3 < acc < 0.7


@ignore_warnings
def test_decoder_regressor_predict_score_surface(_make_surface_reg_data):
    """Test regression predict and scoring for surface image."""
    X, y = _make_surface_reg_data
    model = DecoderRegressor(mask=SurfaceMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "r2"

    model.score(X, y)
    r2 = r2_score(y, y_pred)
    assert r2 <= 0


@ignore_warnings
@pytest.mark.parametrize("frem", [FREMRegressor, FREMClassifier])
def test_frem_decoder_fit_surface(
    frem,
    _make_surface_class_data,
    surf_mask_1d,
):
    """Test fit for using FREM decoding with surface image."""
    X, y = _make_surface_class_data
    model = frem(mask=surf_mask_1d, clustering_percentile=90)
    model.fit(X, y)


# ------------------------ test decoder vs sklearn -------------------------- #


@ignore_warnings
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "classifier_penalty",
    ["svc_l1", "svc_l2", "logistic_l1", "logistic_l2", "ridge_classifier"],
)
def test_decoder_vs_sklearn(
    classifier_penalty, strings_to_sklearn=SUPPORTED_ESTIMATORS
):
    """Compare scores from nilearn Decoder with sklearn classifiers."""
    X, y, mask = _make_multiclass_classification_test_data(
        n_samples=100, dim=10
    )
    n_classes = len(np.unique(y))
    # default cross-validation in nilearn is StratifiedKFold
    # with 10 splits
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # default scoring is accuracy
    scorer = check_scoring(strings_to_sklearn[classifier_penalty], "accuracy")

    ## nilearn decoding
    nilearn_decoder = Decoder(
        estimator=classifier_penalty,
        mask=mask,
        standardize=True,
        cv=cv,
        scoring=scorer,
        screening_percentile=100,  # disable screening
    )
    nilearn_decoder.fit(X, y)
    scores_nilearn = nilearn_decoder.cv_scores_

    ## start decoding with sklearn
    masker = NiftiMasker(mask_img=mask, standardize=True)
    X_transformed = masker.fit_transform(X)

    sklearn_classifier = strings_to_sklearn[classifier_penalty]
    scores_sklearn = {c: [] for c in range(n_classes)}
    # convert multiclass to n_classes binary classifications
    label_binarizer = LabelBinarizer()
    y_binary = label_binarizer.fit_transform(y)
    for klass in range(n_classes):
        for count, (train_idx, test_idx) in enumerate(
            cv.split(X_transformed, y)
        ):
            X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
            y_train, y_test = (
                y_binary[train_idx, klass],
                y_binary[test_idx, klass],
            )
            # set best hyperparameters for each fold
            sklearn_classifier = _set_best_hyperparameters(
                klass,
                sklearn_classifier,
                nilearn_decoder,
                classifier_penalty,
                count,
            )
            sklearn_classifier.fit(X_train, y_train)
            score = scorer(sklearn_classifier, X_test, y_test)
            scores_sklearn[klass].append(score)

    # Flatten scores
    flat_sklearn_scores = np.concatenate(list(scores_sklearn.values()))
    flat_nilearn_scores = np.concatenate(list(scores_nilearn.values()))

    # check average scores are within 2% of each other
    assert np.isclose(
        np.mean(flat_sklearn_scores), np.mean(flat_nilearn_scores), atol=0.02
    )


def _set_best_hyperparameters(
    klass, sklearn_classifier, nilearn_decoder, classifier_penalty, count
):
    if classifier_penalty in ["svc_l1", "svc_l2"]:
        # LinearSVC does not have a CV variant, so we use exactly the
        # parameter selected by nilearn
        sklearn_classifier = clone(sklearn_classifier).set_params(
            C=nilearn_decoder.cv_params_[klass]["C"][count]
        )
    elif classifier_penalty in ["logistic_l1", "logistic_l2"]:
        # this sets the list of Cs as coded within nilearn and
        # LogisticRegressionCV will select the best one using
        # cross-validation
        sklearn_classifier = clone(sklearn_classifier).set_params(
            Cs=nilearn_decoder.cv_params_[klass]["Cs"][count],
        )
    elif classifier_penalty in ["ridge_classifier"]:
        # same as logistic regression
        sklearn_classifier = clone(sklearn_classifier).set_params(
            alphas=nilearn_decoder.cv_params_[klass]["alphas"][count]
        )
    return sklearn_classifier


@ignore_warnings
@pytest.mark.parametrize("regressor", ["svr", "lasso", "ridge"])
def test_regressor_vs_sklearn(
    regressor, strings_to_sklearn=SUPPORTED_ESTIMATORS
):
    """Compare scores from nilearn DecoderRegressor with sklearn regressors."""
    X, y, mask = _make_regression_test_data(n_samples=100, dim=10)
    # for regression default cv in nilearn is KFold with 10 splits
    # shuffling is False by default but we use it here with a fixed seed
    # to reduce variability in the test
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    # r2 is the default scoring for regression
    scorer = check_scoring(strings_to_sklearn[regressor], "r2")

    ## nilearn decoding
    nilearn_regressor = DecoderRegressor(
        estimator=regressor,
        mask=mask,
        standardize=True,
        cv=cv,
        scoring=scorer,
        screening_percentile=100,  # disable screening
    )
    nilearn_regressor.fit(X, y)
    scores_nilearn = nilearn_regressor.cv_scores_["beta"]

    ## start decoding with sklearn
    masker = NiftiMasker(mask_img=mask, standardize=True)
    X_transformed = masker.fit_transform(X)

    sklearn_regressor = strings_to_sklearn[regressor]
    scores_sklearn = []

    for count, (train_idx, test_idx) in enumerate(cv.split(X_transformed, y)):
        X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
        y_train, y_test = (y[train_idx], y[test_idx])
        # set best hyperparameters for each fold
        if regressor == "svr":
            # SVR does not have a CV variant, so we use exactly the
            # parameter selected by nilearn
            sklearn_regressor = clone(sklearn_regressor).set_params(
                C=nilearn_regressor.cv_params_["beta"]["C"][count]
            )
        elif regressor == "lasso":
            # this sets n_alphas as coded within nilearn and
            # LassoCV will select the best one using cross-validation
            sklearn_regressor = clone(sklearn_regressor).set_params(
                n_alphas=nilearn_regressor.cv_params_["beta"]["n_alphas"][
                    count
                ],
            )
        elif regressor in ["ridge"]:
            # same as lasso but with alphas
            sklearn_regressor = clone(sklearn_regressor).set_params(
                alphas=nilearn_regressor.cv_params_["beta"]["alphas"][count]
            )
        sklearn_regressor.fit(X_train, y_train)
        score = scorer(sklearn_regressor, X_test, y_test)
        scores_sklearn.append(score)

    # check average scores are within 1% of each other
    assert np.isclose(
        np.mean(scores_sklearn), np.mean(scores_nilearn), atol=0.01
    )
    # also check individual scores are within 1% of each other
    assert np.allclose(scores_sklearn, scores_nilearn, atol=0.01)
