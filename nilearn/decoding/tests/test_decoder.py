"""
Test the decoder module
"""

# Author: Andres Hoyos-Idrobo
#         Binh Nguyen
#         Thomas Bazeiile
#
# License: simplified BSD

import warnings

import sklearn
import numpy as np
from nilearn.decoding.decoder import (_BaseDecoder, Decoder, DecoderRegressor,
                                      _check_estimator, _check_param_grid, 
                                      _parallel_fit)
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.input_data import NiftiMasker
from nose.tools import assert_equal, assert_raises, assert_true, assert_warns
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV, RidgeClassifierCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.svm import SVR, LinearSVC
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import check_scoring
except ImportError:
    # for scikit-learn 0.18 and 0.19
    from sklearn.metrics.scorer import check_scoring


# Regression
ridge = RidgeCV()
svr = SVR(kernel='linear')
# Classification
svc = LinearSVC()
logistic_l1 = LogisticRegression(penalty='l1')
logistic_l2 = LogisticRegression(penalty='l2')
ridge_classifier = RidgeClassifierCV()
random_forest = RandomForestClassifier()

regressors = {'ridge': (ridge, []),
              'svr': (svr, 'C')}
classifiers = {'svc': (svc, 'C'),
               'logistic_l1': (logistic_l1, 'C'),
               'logistic_l2': (logistic_l2, 'C'),
               'ridge_classifier': (ridge_classifier, [])}
# Create a test dataset
rand = np.random.RandomState(0)
X = rand.rand(100, 10)
# Create different targets
y_regression = rand.rand(100)
y_classification = np.hstack([[-1] * 50, [1] * 50])
y_classification_str = np.hstack([['face'] * 50, ['house'] * 50])
y_multiclass = np.hstack([[0] * 35, [1] * 30, [2] * 35])


def test_check_param_grid():
    """testing several estimators, each one with its specific regularization
    parameter
    """

    # Regression
    for _, (regressor, param) in regressors.items():
        param_grid = _check_param_grid(regressor, X, y_regression, None)
        assert_equal(list(param_grid.keys()), list(param))
    # Classification
    for _, (classifier, param) in classifiers.items():
        param_grid = _check_param_grid(classifier, X, y_classification, None)
        assert_equal(list(param_grid.keys()), list(param))

    # Using a non-linear estimator to raise the error
    for estimator in ['log_l1', random_forest]:
        assert_raises(ValueError, _check_param_grid, estimator, X,
                      y_classification, None)


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
    """Check if the estimator is one of the supported estimators, and if not,
    if it is a string, and if not, then raise the error
    """

    supported_estimators = ['svc', 'svc_l2', 'svc_l1',
                            'logistic', 'logistic_l1', 'logistic_l2',
                            'ridge', 'ridge_classifier',
                            'ridge_regressor', 'svr']
    unsupported_estimators = ['ridgo', 'svb']
    expected_warning = ('Use a custom estimator at your own risk '
                        'of the process not working as intended.')

    with warnings.catch_warnings(record=True) as raised_warnings:
        for estimator in supported_estimators:
            _check_estimator(_BaseDecoder(estimator=estimator).estimator)
    warning_messages = [str(warning.message) for warning in raised_warnings]
    assert expected_warning not in warning_messages
    
    for estimator in unsupported_estimators:
        assert_raises(ValueError, _check_estimator,
                      _BaseDecoder(estimator=estimator).estimator)
    custom_estimator = random_forest
    assert_warns(UserWarning, _check_estimator,
                 _BaseDecoder(estimator=custom_estimator).estimator)


def test_parallel_fit():
    """The goal of this test is to check that results of _parallel_fit is the
    same for differnet controlled param_grid
    """

    X, y = make_regression(n_samples=100, n_features=20,
                           n_informative=5, noise=0.2, random_state=42)
    train = range(80)
    test = range(80, len(y_classification))
    outputs = []
    estimator = svr
    svr_params = [[1e-1, 1e0, 1e1], [1e-1, 1e0, 5e0, 1e1]]
    # define a scorer
    scorer = check_scoring(estimator, 'r2')

    for params in svr_params:
        param_grid = {}
        param_grid['C'] = np.array(params)
        outputs.append(list(_parallel_fit(estimator=estimator, X=X, y=y,
                                          train=train, test=test,
                                          param_grid=param_grid,
                                          is_classification=False, 
                                          scorer=scorer,
                                          mask_img=None, class_index=1,
                                          screening_percentile=None)))
    # check that every element of the output tuple is the same for both tries
    for a, b in zip(outputs[0], outputs[1]):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_almost_equal(a, b)
        else:
            assert_equal(a, b)


def test_decoder_binary_classification():
    X, y = make_classification(n_samples=200, n_features=125, scale=3.0,
                               n_informative=5, n_classes=2, random_state=42)
    X, mask = to_niimgs(X, [5, 5, 5])

    # check classification with masker object
    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_true(accuracy_score(y, y_pred) > 0.95)

    # decoder object use predict_proba for scoring with logistic model
    model = Decoder(estimator='logistic_l2', mask=mask)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_true(accuracy_score(y, y_pred) > 0.95)

    # check different screening_percentile value
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
            groups = rand.binomial(2, 0.3, size=len(y))
        else:
            groups = None
        model.fit(X, y, groups=groups)
        assert_true(accuracy_score(y, y_pred) > 0.9)


def test_decoder_multiclass_classification():
    X, y = make_classification(n_samples=200, n_features=125, scale=3.0,
                               n_informative=5, n_classes=4, random_state=42)
    X, mask = to_niimgs(X, [5, 5, 5])

    # check classification with masker object
    model = Decoder(mask=NiftiMasker())
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_true(accuracy_score(y, y_pred) > 0.95)

    # check different screening_percentile value
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
            groups = rand.binomial(2, 0.3, size=len(y))
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
    X = StandardScaler().fit_transform(X)
    y = (y - y.mean()) / y.std()
    X, mask = to_niimgs(X, [5, 5, 5])
    for regressor_ in regressors:
        for screening_percentile in [100, 20]:
            model = DecoderRegressor(estimator=regressor_, mask=mask,
                                     screening_percentile=screening_percentile)
            model.fit(X, y)
            y_pred = model.predict(X)
            assert_true(r2_score(y, y_pred) > 0.95)


def test_decoder_apply_mask():
    X_init, y = make_classification(
        n_samples=200, n_features=125, scale=3.0,
        n_informative=5, n_classes=4, random_state=42)
    X, _ = to_niimgs(X_init, [5, 5, 5])
    model = Decoder(mask=NiftiMasker())

    X_masked = model._apply_mask(X)

    # test whether if _apply mask output has the same shape as original matrix
    assert_equal(X_masked.shape, X_init.shape)

    # test whethere model.masker_ have some desire attributes manually set 
    # after calling _apply_mask; by default these parameters are set to None
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
        smoothing_fwhm=smoothing_fwhm

    )
    X_masked = model._apply_mask(X)
    
    assert_true(np.any(model.masker_.target_affine == target_affine))
    assert_equal(model.masker_.target_shape, target_shape)
    assert_equal(model.masker_.t_r, t_r)
    assert_equal(model.masker_.high_pass, high_pass)
    assert_equal(model.masker_.low_pass, low_pass)
    assert_equal(model.masker_.smoothing_fwhm, smoothing_fwhm)
