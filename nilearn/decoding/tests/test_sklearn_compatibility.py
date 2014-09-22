import numpy as np
from ..space_net import SpaceNet
from nose.tools import assert_true
import traceback


def test_get_params():
    # Issue #12 (on github) reported that our objects
    # get_params() methods returned empty dicts.

    for penalty in ["smooth-lasso", "tvl1"]:
        for classif in [True, False]:
            kwargs = {}
            for param in ["max_iter", "alpha", "l1_ratio", "verbose",
                          "tol", "mask", "memory", "copy_data",
                          "fit_intercept", "alphas"]:
                m = SpaceNet(mask='dummy',
                             penalty=penalty, classif=classif, **kwargs)
                try:
                    params = m.get_params()
                except AttributeError:
                    if "get_params" in traceback.format_exc():
                        params = m._get_params()
                    else:
                        raise

                assert_true(param in params,
                            msg="%s doesn't have parameter '%s'." % (
                        m, param))
