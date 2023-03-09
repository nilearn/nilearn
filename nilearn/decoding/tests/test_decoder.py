"""Test the decoder module."""

# Author: Andres Hoyos-Idrobo
#         Binh Nguyen
#         Thomas Bazeiile
#
# License: simplified BSD

import warnings

import numpy as np
import pytest
from nilearn._utils.param_validation import check_feature_screening
from nilearn.decoding.decoder import (
    Decoder,
    DecoderRegressor,
    FREMClassifier,
    FREMRegressor,
    _BaseDecoder,
    _check_estimator,
    _check_param_grid,
    _parallel_fit,
)
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.maskers import NiftiMasker
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, get_scorer, r2_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC

try:
    from sklearn.metrics import check_scoring
except ImportError:
    # for scikit-learn 0.18 and 0.19
    from sklearn.metrics.scorer import check_scoring

# Regression
ridge = RidgeCV()
svr = SVR(kernel="linear")
# Classification
svc = LinearSVC()
logistic_l1 = LogisticRegression(penalty="l1")
logistic_l2 = LogisticRegression(penalty="l2")
ridge_classifier = RidgeClassifierCV()
random_forest = RandomForestClassifier()

dummy_classifier = DummyClassifier(random_state=0)
dummy_regressor = DummyRegressor()

regressors = {"ridge": (ridge, []), "svr": (svr, "C")}
classifiers = {
    "svc": (svc, "C"),
    "logistic_l1": (logistic_l1, "C"),
    "logistic_l2": (logistic_l2, "C"),
    "ridge_classifier": (ridge_classifier, []),
}
# Create a test dataset
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
# Create different targets
y_regression = rng.rand(100)
y_classification = np.hstack([[-1] * 50, [1] * 50])
y_classification_str = np.hstack([["face"] * 50, ["house"] * 50])
y_multiclass = np.hstack([[0] * 35, [1] * 30, [2] * 35])


def test_check_param_grid():
    """Test several estimators.

    Each one with its specific regularization parameter.
    """
    # Regression
    for _, (regressor, param) in regressors.items():
        param_grid = _check_param_grid(regressor, X, y_regression, None)
        assert list(param_grid.keys()) == list(param)
    # Classification
    for _, (classifier, param) in classifiers.items():
        param_grid = _check_param_grid(classifier, X, y_classification, None)
        assert list(param_grid.keys()) == list(param)

    # Using a non-linear estimator to raise the error
    for estimator in ["log_l1", random_forest]:
        pytest.raises(
            ValueError, _check_param_grid, estimator, X, y_classification, None
        )

    # Test return parameter grid is empty
    param_grid = _check_param_grid(dummy_classifier, X, y_classification, None)
    assert param_grid == {}


def test_check_inputs_length():
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    # Remove ten samples from y
    y = y[:-10]

    for model in [DecoderRegressor, Decoder, FREMRegressor, FREMClassifier]:
        pytest.raises(
            ValueError, model(mask=mask, screening_percentile=100.0).fit, X_, y
        )


def test_check_estimator():
    """Check if the estimator is one of the supported estimators.

    If not, if it is a string, and if not, then raise the error.
    """
    supported_estimators = [
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
    ]
    unsupported_estimators = ["ridgo", "svb"]
    expected_warning = (
        "Use a custom estimator at your own risk "
        "of the process not working as intended."
    )

    with warnings.catch_warnings(record=True) as raised_warnings:
        for estimator in supported_estimators:
            _check_estimator(_BaseDecoder(estimator=estimator).estimator)
    warning_messages = [str(warning.message) for warning in raised_warnings]
    assert expected_warning not in warning_messages

    for estimator in unsupported_estimators:
        pytest.raises(
            ValueError,
            _check_estimator,
            _BaseDecoder(estimator=estimator).estimator,
        )
    custom_estimator = random_forest
    pytest.warns(
        UserWarning,
        _check_estimator,
        _BaseDecoder(estimator=custom_estimator).estimator,
    )


def test_parallel_fit():
    """Check that results of _parallel_fit is the same \
    for different controlled param_grid."""
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=5,
        noise=0.2,
        random_state=42,
    )
    train = range(80)
    test = range(80, len(y_classification))
    outputs = []
    estimator = svr
    svr_params = [[1e-1, 1e0, 1e1], [1e-1, 1e0, 5e0, 1e1]]
    scorer = check_scoring(estimator, "r2")  # define a scorer
    # Define a screening selector
    selector = check_feature_screening(
        screening_percentile=None, mask_img=None, is_classification=False
    )
    for params in svr_params:
        param_grid = {"C": np.array(params)}
        outputs.append(
            list(
                _parallel_fit(
                    estimator=estimator,
                    X=X,
                    y=y,
                    train=train,
                    test=test,
                    param_grid=param_grid,
                    is_classification=False,
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
            np.testing.assert_array_almost_equal(a, b)
        else:
            assert a == b


def _make_binary_classification_test_data(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=125,
        scale=3.0,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    X, mask = to_niimgs(X, [5, 5, 5])
    return X, y, mask


def test_decoder_binary_classification_with_masker_object():
    X, y, mask = _make_binary_classification_test_data(n_samples=200)

    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "roc_auc"
    assert model.score(X, y) == 1.0
    assert accuracy_score(y, y_pred) > 0.95


def test_decoder_binary_classification_with_logistic_model():
    """Check decoder with predict_proba for scoring with logistic model."""
    X, y, mask = _make_binary_classification_test_data(n_samples=200)

    model = Decoder(estimator="logistic_l2", mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@pytest.mark.parametrize("screening_percentile", [100, 20, None])
def test_decoder_binary_classification_screening(screening_percentile):
    X, y, mask = _make_binary_classification_test_data(n_samples=200)

    model = Decoder(mask=mask, screening_percentile=screening_percentile)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@pytest.mark.parametrize("clustering_percentile", [100, 99])
def test_decoder_binary_classification_clustering(clustering_percentile):
    X, y, mask = _make_binary_classification_test_data(n_samples=200)

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


@pytest.mark.parametrize("cv", [KFold(n_splits=5), LeaveOneGroupOut()])
def test_decoder_binary_classification_cross_validation(cv):
    X, y, mask = _make_binary_classification_test_data(n_samples=200)

    # check cross-validation scheme and fit attribute with groups enabled
    rand_local = np.random.RandomState(42)

    model = Decoder(estimator="svc", mask=mask, standardize=True, cv=cv)
    groups = None
    if isinstance(cv, LeaveOneGroupOut):
        groups = rand_local.binomial(2, 0.3, size=len(y))
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.9


def test_decoder_dummy_classifier():
    n_samples = 400
    X, y, mask = _make_binary_classification_test_data(n_samples=n_samples)

    # We make 80% of y to have value of 1.0 to check whether the stratified
    # strategy returns a proportion prediction value of 1.0 of roughly 80%
    proportion = 0.8
    y = np.zeros(n_samples)
    y[: int(proportion * n_samples)] = 1.0

    model = Decoder(estimator="dummy_classifier", mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.sum(y_pred == 1.0) / n_samples - proportion < 0.05


def test_decoder_dummy_classifier_with_callable():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    accuracy_scorer = get_scorer("accuracy")
    model = Decoder(
        estimator="dummy_classifier", mask=mask, scoring=accuracy_scorer
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == accuracy_scorer
    assert model.score(X, y) == accuracy_score(y, y_pred)


def test_decoder_error_model_not_fitted():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    model = Decoder(estimator="dummy_classifier", mask=mask)
    with pytest.raises(
        NotFittedError, match="This Decoder instance is not fitted yet."
    ):
        model.score(X, y)


def test_decoder_dummy_classifier_strategy_prior():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    param = dict(strategy="prior")
    dummy_classifier.set_params(**param)
    model = Decoder(estimator=dummy_classifier, mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.all(y_pred) == 1.0
    assert roc_auc_score(y, y_pred) == 0.5


def test_decoder_dummy_classifier_strategy_most_frequent():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    param = dict(strategy="most_frequent")
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


def test_decoder_dummy_classifier_roc_scoring():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    model = Decoder(estimator="dummy_classifier", mask=mask, scoring="roc_auc")
    model.fit(X, y)

    assert np.mean(model.cv_scores_[0]) >= 0.45


def test_decoder_error_not_implemented():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    param = dict(strategy="constant")
    dummy_classifier.set_params(**param)
    model = Decoder(estimator=dummy_classifier, mask=mask)

    pytest.raises(NotImplementedError, model.fit, X, y)


def test_decoder_error_unknown_scoring_metrics():
    X, y, mask = _make_binary_classification_test_data(n_samples=400)

    model = Decoder(estimator=dummy_classifier, mask=mask, scoring="foo")

    with pytest.raises(ValueError, match="'foo' is not a valid scoring value"):
        model.fit(X, y)


def test_decoder_dummy_classifier_default_scoring():
    X, y, _ = _make_binary_classification_test_data(n_samples=400)

    model = Decoder(estimator="dummy_classifier", scoring=None)

    assert model.scoring is None

    model.fit(X, y)

    assert model.scorer_._score_func == get_scorer("accuracy")._score_func
    assert model.scorer_._sign == get_scorer("accuracy")._sign
    assert model.score(X, y) > 0.5


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


def _make_regression_test_data(n_samples, dim):
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


@pytest.mark.parametrize("screening_percentile", [100, 20, 1, None])
def test_decoder_regression_screening(screening_percentile):
    X, y, mask = _make_regression_test_data(n_samples=100, dim=30)

    for reg in regressors:
        model = DecoderRegressor(
            estimator=reg,
            mask=mask,
            screening_percentile=screening_percentile,
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        assert r2_score(y, y_pred) > 0.95


@pytest.mark.parametrize("clustering_percentile", [100, 99])
def test_decoder_regression_clustering(clustering_percentile):
    X, y, mask = _make_regression_test_data(n_samples=100, dim=5)

    for reg in regressors:
        model = FREMRegressor(
            estimator=reg,
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


def test_decoder_dummy_regression():
    X, y, mask = _make_regression_test_data(n_samples=100, dim=30)

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

    # Check that default scoring metric for regression is r2
    model = DecoderRegressor(
        estimator="dummy_regressor", mask=mask, scoring=None
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    assert model.score(X, y) == r2_score(y, y_pred)

    # decoder object use other strategy for dummy regressor
    param = dict(strategy="median")
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


def _make_multiclass_classification_test_data(n_samples=200):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=125,
        scale=3.0,
        n_informative=5,
        n_classes=4,
        random_state=42,
    )
    X, mask = to_niimgs(X, [5, 5, 5])
    return X, y, mask


def test_decoder_multiclass_classification_masker():
    X, y, _ = _make_multiclass_classification_test_data()

    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


def test_decoder_multiclass_classification_masker_dummy_classifier():
    X, y, _ = _make_multiclass_classification_test_data()

    model = Decoder(
        estimator="dummy_classifier", mask=NiftiMasker(), scoring="accuracy"
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert model.scoring == "accuracy"
    # 4-class classification
    assert accuracy_score(y, y_pred) > 0.2
    assert model.score(X, y) == accuracy_score(y, y_pred)


@pytest.mark.parametrize("screening_percentile", [100, 20, None])
def test_decoder_multiclass_classification_screening(screening_percentile):
    X, y, mask = _make_multiclass_classification_test_data()

    model = Decoder(mask=mask, screening_percentile=screening_percentile)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert accuracy_score(y, y_pred) > 0.95


@pytest.mark.parametrize("clustering_percentile", [100, 99])
@pytest.mark.parametrize("estimator", ["svc_l2", "svc_l1"])
def test_decoder_multiclass_classification_clustering(
    clustering_percentile, estimator
):
    X, y, mask = _make_multiclass_classification_test_data()

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


@pytest.mark.parametrize("cv", [KFold(n_splits=5), LeaveOneGroupOut()])
def test_decoder_multiclass_classification_cross_validation(cv):
    X, y, mask = _make_multiclass_classification_test_data()

    # check cross-validation scheme and fit attribute with groups enabled
    rand_local = np.random.RandomState(42)

    model = Decoder(estimator="svc", mask=mask, standardize=True, cv=cv)
    groups = None
    if isinstance(cv, LeaveOneGroupOut):
        groups = rand_local.binomial(2, 0.3, size=len(y))
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)
    assert accuracy_score(y, y_pred) > 0.9


def test_decoder_apply_mask():
    X_init, y = make_classification(
        n_samples=200,
        n_features=125,
        scale=3.0,
        n_informative=5,
        n_classes=4,
        random_state=42,
    )
    X, _ = to_niimgs(X_init, [5, 5, 5])

    model = Decoder(mask=NiftiMasker())

    X_masked = model._apply_mask(X)

    # test whether if _apply mask output has the same shape as original matrix
    assert X_masked.shape == X_init.shape

    # test whether model.masker_ have some desire attributes manually set after
    # calling _apply_mask; by default these parameters are set to None
    target_affine = 2 * np.eye(4)
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


def test_decoder_split_cv():
    X, y, _ = _make_multiclass_classification_test_data()
    rand_local = np.random.RandomState(42)
    groups = rand_local.binomial(2, 0.3, size=len(y))

    # Check whether ValueError is raised when cv is not set correctly
    for cv in ["abc", LinearSVC()]:
        model = Decoder(mask=NiftiMasker(), cv=cv)
        pytest.raises(ValueError, model.fit, X, y)

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
    with pytest.warns(UserWarning, match=".*screening_percentile parameters"):
        model = FREMClassifier(
            clustering_percentile=10,
            screening_percentile=10,
            mask=NiftiMasker(),
            cv=1,
        )
        model.fit(X, y)
