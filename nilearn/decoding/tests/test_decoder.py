"""
Test the decoder module
"""

# Author: Andres Hoyos-Idrobo
# License: simplified BSD

from nose.tools import assert_equal, assert_true, assert_raises
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVR
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  Ridge)
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.decoding.decoder import Decoder, DecoderRegressor, BaseDecoder
from nilearn.decoding.decoder import _check_param_grid


# Create a test dataset
rand = np.random.RandomState(0)
X = rand.rand(100, 10)
# Create different targets
y_regression = rand.rand(100)
y_classif = np.hstack([[-1] * 50, [1] * 50])
y_classif_str = np.hstack([['face'] * 50, ['house'] * 50])
y_multiclass = np.hstack([[0] * 35, [1] * 30, [2] * 35])

# Test estimators
# Regression
ridge = Ridge()
svr = SVR(kernel='linear')
# Classification
svc = LinearSVC()
logistic_l1 = LogisticRegression(penalty='l1')
logistic_l2 = LogisticRegression(penalty='l2')
ridge_classifier = RidgeClassifier()
random_forest = RandomForestClassifier()


def test_check_param_grid():
    # testing several estimators, each one with its specific regularization
    # parameter
    regressors = {'ridge': (ridge, 'alpha'),
                  'svr': (svr, 'C')}
    classifiers = {'svc': (svc, 'C'),
                   'logistic_l1': (logistic_l1, 'C'),
                   'logistic_l2': (logistic_l2, 'C'),
                   'ridge_classifier': (ridge_classifier, 'alpha')}
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


def test_checking_inputs_length():
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    # Remove ten samples from y
    y = y[:-10]

    for model in [DecoderRegressor, Decoder]:
        assert_raises(ValueError, model(mask=mask,
                                        screening_percentile=100.).fit, X_, y)


def test_decoder_prediction():
    iris = load_iris()
    X, y = iris.data, iris.target
    X, mask = to_niimgs(X, [2, 2, 2])

    for screening_percentile in [100, 20]:
        model = Decoder(mask=mask)
        model.fit(X, y)
        # checking its overfit
        y_pred = model.predict(X)
        assert_true(accuracy_score(y, y_pred) > 0.95)


def test_check_estimator():
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask = to_niimgs(X, (2, 2, 2))

    for estimator in ['log_l1', 'log_l2', 'ridgo']:
        assert_raises(ValueError, BaseDecoder(estimator=estimator).fit, X_, y)
