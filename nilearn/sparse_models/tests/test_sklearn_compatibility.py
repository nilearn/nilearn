import numpy as np
from ..estimators import (TVl1Classifier, TVl1Regressor,
                          SmoothLassoClassifier, SmoothLassoRegressor)
from ..cv import TVl1ClassifierCV, SmoothLassoClassifierCV
from nose.tools import assert_true


def test_get_params():
    # Issue #12 (on github) reported that our objects
    # get_params() methods returned empty dicts.
    for model in [SmoothLassoRegressor, SmoothLassoClassifier,
                  SmoothLassoClassifierCV, TVl1ClassifierCV,
                  TVl1Classifier, TVl1Regressor]:
        kwargs = {}
        if model.__name__.endswith('CV'):
            kwargs['alphas'] = np.logspace(-3, 1, num=5)
        for param in ["max_iter", "alpha", "l1_ratio", "verbose",
                      "callback", "tol", "mask", "memory", "backtracking",
                      "copy_data", "fit_intercept", "alphas"]:
            if model.__name__.endswith("CV"):
                if param == "alpha":
                    continue
            elif param == "alphas":
                continue
            assert_true(param in model(**kwargs).get_params(),
                        msg="Class '%s' doesn't have parameter '%s'." % (
                            model.__name__, param))
