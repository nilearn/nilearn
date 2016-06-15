"""
Test the decoder module
"""

# Author: Andres Hoyos-Idrobo
# License: simplified BSD

from nose.tools import (assert_equal, assert_true, assert_false,
                        assert_raises)
import warnings
import os
import numpy as np
import nibabel
from sklearn.svm import LinearSVC, SVR
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  Ridge)
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from nilearn.input_data import NiftiMasker
from nilearn.decoding.tests.test_same_api import to_niimgs
# from nilearn.image import index_img
from nilearn._utils.testing import assert_warns
from nilearn.decoding.decoder import (Decoder, MNI152_BRAIN_VOLUME,
                                      _check_estimator,
                                      _check_param_grid,
                                      _check_masking,
                                      _check_scorer,
                                      _check_feature_screening,
                                      _get_mask_volume)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_decoder_score():
    """
    Testing the decoder score method
    """

    # using the isis dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    y = 2 * (y > 0) - 1
    X_, mask_img = to_niimgs(X, (2, 2, 2))

    masker = NiftiMasker(mask_img=mask_img).fit()

    # testing two classifiers
    for estimator in ['svc', 'ridge_classifier']:
        for mask in [None, masker]:
            gnc = Decoder(estimator=estimator, mask=mask,
                          standardize=False,
                          screening_percentile=100.)

            assert_raises(ValueError, gnc.predict, X_)

            gnc.fit(X_, y)
            accuracy = gnc.score(X_, y)
            assert_equal(accuracy, accuracy_score(y, gnc.predict(X_)))


def test_check_masking():

    # Create toy mask_img
    mask = np.ones((5, 5, 5), np.bool)
    mask_img = nibabel.Nifti1Image(mask.astype(np.int),
                                   np.eye(4))

    # Using two different smoothing_fwhm to compare overriding of this param
    # after masker fitting
    smoothing_fwhm = 4
    smoothing_fwhm_test = 8

    kwargs = {'target_affine': np.eye(4),
              'target_shape': (5, 5, 5),
              'standardize': True,
              'mask_strategy': 'epi',
              'memory': None,
              'memory_level': 1}

    # the masker shold be fitted
    masker_test_1 = NiftiMasker(smoothing_fwhm=smoothing_fwhm_test, **kwargs)
    # assigning a mask_img
    masker_test_2 = NiftiMasker(mask_img=mask_img,
                                smoothing_fwhm=smoothing_fwhm_test, **kwargs)
    # assigning all the params
    masker_test_3 = NiftiMasker(mask_img=mask_img,
                                smoothing_fwhm=smoothing_fwhm_test,
                                **kwargs).fit()

    # Testing various mask inputs
    masks = [None, masker_test_1, masker_test_2, masker_test_3]
    for mask in masks:
        masker = _check_masking(mask, smoothing_fwhm=smoothing_fwhm, **kwargs)
        if mask is None:
            assert_true(isinstance(masker, BaseEstimator))
        else:
            assert_true(masker.smoothing_fwhm == smoothing_fwhm_test)


# Crate a test dataset
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


def test_check_estimator():

    regressors = [ridge, svr]
    classifiers = [svc, logistic_l1, logistic_l2, ridge_classifier]

    # Regression
    for regressor in regressors:
        is_classification, is_binary, classes, classes_to_predict = \
            _check_estimator(regressor, y_regression, None)

        is_classification, is_binary, classes, classes_to_predict = \
            _check_estimator(regressor, y_classif, None)

        assert_false(is_classification)

    # Classification
    for classifier in classifiers:
        is_classification, is_binary, classes, classes_to_predict = \
            _check_estimator(classifier, y_regression, None)

        is_classification, is_binary, classes, classes_to_predict = \
            _check_estimator(classifier, y_classif, 1)

        is_classification, is_binary, classes, classes_to_predict = \
            _check_estimator(classifier, y_multiclass, None)

        assert_true(is_classification)


class DummyDecoder(BaseEstimator):
    """Dummy decoder to test check scorer"""
    def __init__(self, is_classification=True, is_binary=True):
        self.is_classification_ = is_classification
        self.is_binary_ = is_binary
        # dummy classes
        self.classes_ = ['baseline', 'cellphone']

    def fit(X, y):
        pass

    def predict(X):
        pass


def test_check_scorer():

    for is_binary in [True, False]:

        regressor = DummyDecoder(is_classification=False, is_binary=is_binary)
        classifier = DummyDecoder(is_classification=True, is_binary=is_binary)

        # Regression
        score, scoring, score_func = _check_scorer(regressor, 'r2', None,
                                                   y_regression)
        score, scoring, score_func = _check_scorer(regressor, None, None,
                                                   y_regression)
        score, scoring, score_func = _check_scorer(regressor, 'other_test',
                                                   None, y_regression)

        assert_raises(ValueError, _check_scorer, regressor, 'accuracy', None,
                      y_regression)

        # Classification
        score, scoring, score_func = _check_scorer(classifier, 'accuracy',
                                                   None, y_classif)
        score, scoring, score_func = _check_scorer(classifier, None, None,
                                                   y_classif)
        score, scoring, score_func = _check_scorer(classifier, 'other_test',
                                                   None, y_classif)

        assert_raises(ValueError, _check_scorer, classifier, 'r2', None,
                      y_classif)
        assert_raises(ValueError, _check_scorer, classifier, 'f1_score', None,
                      y_classif_str)
        assert_raises(ValueError, _check_scorer, classifier, 'f1_score',
                      'face', y_classif_str)


def test_feature_screening():

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:

            if screening_percentile == 100 or screening_percentile is None:
                assert_equal(_check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    None)
            elif screening_percentile == 101 or screening_percentile == -1:
                assert_raises(ValueError, _check_feature_screening,
                              screening_percentile, MNI152_BRAIN_VOLUME,
                              is_classif)
            elif screening_percentile == 20:
                assert_true(isinstance(_check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    BaseEstimator))
            else:
                assert_warns(UserWarning, _check_feature_screening,
                             screening_percentile, MNI152_BRAIN_VOLUME * 2,
                             is_classif)


# XXX this test is repeated, taken form test_space_net
def test_get_mask_volume():
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (
            mni152_brain_mask))
