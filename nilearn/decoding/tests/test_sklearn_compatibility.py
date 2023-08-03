import traceback

import pytest

from nilearn.decoding.space_net import BaseSpaceNet


@pytest.mark.parametrize("penalty", ["graph-net", "tv-l1"])
@pytest.mark.parametrize("is_classif", [True, False])
@pytest.mark.parametrize(
    "param",
    [
        "max_iter",
        "alphas",
        "l1_ratios",
        "verbose",
        "tol",
        "mask",
        "memory",
        "fit_intercept",
        "alphas",
    ],
)
def test_get_params(penalty, is_classif, param):
    # Issue #12 (on github) reported that our objects
    # get_params() methods returned empty dicts.

    kwargs = {}
    m = BaseSpaceNet(
        mask="dummy",
        penalty=penalty,
        is_classif=is_classif,
        **kwargs,
    )
    try:
        params = m.get_params()
    except AttributeError:
        if "get_params" in traceback.format_exc():
            params = m._get_params()
        else:
            raise

    assert param in params, f"{m} doesn't have parameter '{param}'."
