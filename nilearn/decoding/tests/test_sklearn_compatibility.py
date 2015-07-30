from nilearn.decoding.space_net import BaseSpaceNet
from nose.tools import assert_true
import traceback


def test_get_params():
    # Issue #12 (on github) reported that our objects
    # get_params() methods returned empty dicts.

    for penalty in ["graph-net", "tv-l1"]:
        for is_classif in [True, False]:
            kwargs = {}
            for param in ["max_iter", "alphas", "l1_ratios", "verbose",
                          "tol", "mask", "memory", "fit_intercept", "alphas"]:
                m = BaseSpaceNet(
                    mask='dummy',
                    penalty=penalty,
                    is_classif=is_classif,
                    **kwargs)
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
