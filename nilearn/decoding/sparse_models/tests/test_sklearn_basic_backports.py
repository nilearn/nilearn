import numpy as np
from nose.tools import assert_true, assert_false
from ...._utils.fixes import LabelBinarizer, is_classifier, is_regressor
from ..cv import (SmoothLassoRegressorCV, SmoothLassoClassifierCV,
                  TVl1RegressorCV, TVl1ClassifierCV)
from ..estimators import (SmoothLassoRegressor, SmoothLassoClassifier,
                          TVl1Regressor, TVl1Classifier)


def test_is_classifier():
    assert_true(is_classifier(SmoothLassoClassifier()))
    assert_true(is_classifier(SmoothLassoClassifierCV()))
    assert_true(is_classifier(TVl1Classifier()))
    assert_true(is_classifier(TVl1ClassifierCV()))
    assert_false(is_regressor(SmoothLassoClassifier()))
    assert_false(is_regressor(SmoothLassoClassifierCV()))
    assert_false(is_regressor(TVl1Classifier()))
    assert_false(is_regressor(TVl1ClassifierCV()))


def test_is_regressor():
    assert_true(is_regressor(SmoothLassoRegressor()))
    assert_true(is_regressor(SmoothLassoRegressorCV()))
    assert_true(is_regressor(TVl1Regressor()))
    assert_true(is_regressor(TVl1RegressorCV()))
    assert_false(is_classifier(SmoothLassoRegressor()))
    assert_false(is_classifier(SmoothLassoRegressorCV()))
    assert_false(is_classifier(TVl1Regressor()))
    assert_false(is_classifier(TVl1RegressorCV()))


def test_labelbinarizer_backport():
    np.testing.assert_array_equal(
        LabelBinarizer().fit_transform(np.arange(4)),
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]))

    np.testing.assert_array_equal(
        LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(np.arange(4)),
        np.array([[1, -1, -1, -1],
                  [-1, 1, -1, -1],
                  [-1, -1, 1, -1],
                  [-1, -1, -1, 1]]))


def test_labelbinarizer_misc_use_cases():
    lb = LabelBinarizer()
    np.testing.assert_array_equal(lb.fit([1, 2, 6, 4, 2]).classes_,
                                  [1, 2, 4, 6])
    np.testing.assert_array_equal(lb.transform([1, 6]),
                                  [[1, 0, 0, 0],
                                   [0, 0, 0, 1]])

    lb = LabelBinarizer()
    # np.testing.assert_array_equal(lb.fit_transform(['yes', 'no', 'no', 'yes']),
    #                               [[1],
    #                                [0],
    #                                [0],
    #                                [1]])

    np.testing.assert_array_equal(lb.fit([[0, 1, 1], [1, 0, 0]]).classes_,
                                  [0, 1])

    clf = LabelBinarizer()
    np.testing.assert_array_equal(clf.fit([1, 2, 6, 4, 2]).classes_,
                                  [1, 2, 4, 6])
    np.testing.assert_array_equal(clf.transform([1, 6]),
                                  [[1., 0., 0., 0.],
                                   [0., 0., 0., 1.]])

    np.testing.assert_array_equal(clf.fit_transform([(1, 2), (3,)]),
                                  [[1., 1., 0.],
                                   [0., 0., 1.]])
    np.testing.assert_array_equal(clf.classes_, [1, 2, 3])
