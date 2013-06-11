"""
Test the class_inspect module
"""
from nose.tools import assert_equal

from sklearn.base import BaseEstimator

from nisl._utils import class_inspect


class A(BaseEstimator):

    def __init__(self, a=1):
        self.a = a


class B(A):

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

def test_get_params():
    b = B()
    params_a_in_b = class_inspect.get_params(A, b)
    assert_equal(params_a_in_b, dict(a=1))
