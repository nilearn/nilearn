"""Test the class_inspect module."""

from nilearn._base import NilearnBaseEstimator
from nilearn._utils import class_inspect

##############################################################################
# Helpers for the tests


class A(NilearnBaseEstimator):
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
