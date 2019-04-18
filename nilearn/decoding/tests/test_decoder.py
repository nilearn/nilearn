"""
Test the decoder module
"""

# Author: Andres Hoyos-Idrobo
# License: simplified BSD

import numpy as np
from nose.tools import assert_equal, assert_raises, assert_true
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.svm import SVR, LinearSVC

from nilearn.decoding.decoder import (BaseDecoder, Decoder, DecoderRegressor,
                                      _check_param_grid)
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.input_data import NiftiMasker


def test_check_param_grid():
    # testing several estimators, each one with its specific regularization
    # parameter
    # Regression
    ridge = Ridge()
    svr = SVR(kernel='linear')
    # Classification
    svc = LinearSVC()
    logistic_l1 = LogisticRegression(penalty='l1')
    logistic_l2 = LogisticRegression(penalty='l2')
    ridge_classifier = RidgeClassifier()
    random_forest = RandomForestClassifier()

    regressors = {'ridge': (ridge, 'alpha'),
                  'svr': (svr, 'C')}
    classifiers = {'svc': (svc, 'C'),
                   'logistic_l1': (logistic_l1, 'C'),
                   'logistic_l2': (logistic_l2, 'C'),
                   'ridge_classifier': (ridge_classifier, 'alpha')}
    # Create a test dataset
    rand = np.random.RandomState(0)
    X = rand.rand(100, 10)
    # Create different targets
    y_regression = rand.rand(100)
    y_classif = np.hstack([[-1] * 50, [1] * 50])
    y_classif_str = np.hstack([['face'] * 50, ['house'] * 50])
    y_multiclass = np.hstack([[0] * 35, [1] * 30, [2] * 35])

    # Regression
    for _, (regressor, param) in regressors.items():
        param_grid = _check_param_grid(regressor, X, y_regression, None)
        assert_equal(list(param_grid.keys())[0], param)
    # Classification
    for _, (classifier, param) in classifiers.items():
        param_grid = _check_param_grid(classifier, X, y_classif, None)
        assert_equal(list(param_grid.keys())[0], param)

    # Using a non-linear estimator to raise the error
    for estimator in ['log_l1', random_forest]:
        assert_raises(ValueError, _check_param_grid, estimator, X,
                      y_classif, None)


def test_check_inputs_length():
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    # Remove ten samples from y
    y = y[:-10]

    for model in [DecoderRegressor, Decoder]:
        assert_raises(ValueError, model(mask=mask,
                                        screening_percentile=100.).fit, X_, y)


def test_check_estimator():
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    for estimator in ['log_l1', 'log_l2', 'ridgo']:
        assert_raises(ValueError, BaseDecoder(estimator=estimator).fit, X_, y)


def test_decoder_classification():
    X, y = make_classification(n_samples=200, n_features=125, scale=3.0,
                               n_informative=5, n_classes=4, random_state=42)
    X, mask = to_niimgs(X, [5, 5, 5])

    # check classification with masker object
    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_true(accuracy_score(y, y_pred) > 0.95)

    # check differnet screening_percentile value
    for screening_percentile in [100, 20]:
        model = Decoder(mask=mask, screening_percentile=screening_percentile)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert_true(accuracy_score(y, y_pred) > 0.95)

    # check cross-validation scheme and fit attribute with groups enabled
    for cv in [KFold(n_splits=5), LeaveOneGroupOut()]:
        model = Decoder(estimator='svc', mask=mask,
                        standardize=True, cv=cv)
        if isinstance(cv, LeaveOneGroupOut):
            groups = np.random.binomial(2, 0.3, size=len(y))
        else:
            groups = None
        model.fit(X, y, groups=groups)
        assert_true(accuracy_score(y, y_pred) > 0.9)


def test_decoder_classification_string_label():
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, [2, 2, 2])
    labels = ['red', 'blue', 'green']
    y_str = [labels[y[i]] for i in range(len(y))]

    model = Decoder(mask=mask)
    model.fit(X, y_str)
    y_pred = model.predict(X)
    assert_true(accuracy_score(y_str, y_pred) > 0.95)


def test_decoder_regression():
    X, y = make_regression(n_samples=200, n_features=125,
                           n_informative=5, noise=0.2, random_state=42)
    X, mask = to_niimgs(X, [5, 5, 5])
    for screening_percentile in [100, 20]:
        model = DecoderRegressor(estimator='ridge', mask=mask,
                                 screening_percentile=screening_percentile)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert_true(r2_score(y, y_pred) > 0.95)
