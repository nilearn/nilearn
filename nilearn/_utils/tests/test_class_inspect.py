"""
Test the class_inspect module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

import pytest
from sklearn.base import BaseEstimator

from nilearn._utils import class_inspect

##############################################################################
# Helpers for the tests


class A(BaseEstimator):
    def __init__(self, a=1):
        self.a = a


class B(A):
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b


##############################################################################
# The tests themselves


def test_get_params():
    b = B()
    params_a_in_b = class_inspect.get_params(A, b)
    assert params_a_in_b == {"a": 1}
    params_a_in_b = class_inspect.get_params(A, b, ignore=["a"])
    assert params_a_in_b == {}


def test_check_estimator_has_sklearn_is_fitted():
    """Check errors are thrown for unfitted estimator.

    Check that before fitting
    - estimlator has a __sklearn_is_fitted__ method that returns false
    - running sklearn check_is_fitted on masker throws an error
    """

    class DummyEstimator:
        def __init__(self):
            pass

    with pytest.raises(
        TypeError, match="must have __sklearn_is_fitted__ method"
    ):
        class_inspect.check_estimator_has_sklearn_is_fitted(DummyEstimator())

    class DummyEstimator:
        def __init__(self):
            pass

        def __sklearn_is_fitted__(self):
            return True

    with pytest.raises(ValueError, match="must return False before fit"):
        class_inspect.check_estimator_has_sklearn_is_fitted(DummyEstimator())
