"""
Test the class_inspect module
"""
from nose.tools import assert_equal

from sklearn.base import BaseEstimator

from nisl._utils import class_inspect

###############################################################################
# Helpers for the tests

class A(BaseEstimator):

    def __init__(self, a=1):
        self.a = a


class B(A):

    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def get_scope_name(self):
        return get_scope_name()


def get_scope_name():
    return class_inspect.enclosing_scope_name()


###############################################################################
# The tests themselves

def test_get_params():
    b = B()
    params_a_in_b = class_inspect.get_params(A, b)
    assert_equal(params_a_in_b, dict(a=1))


def test_enclosing_scope_name():
    b = B()
    name = b.get_scope_name()
    assert_equal(name, '[B.get_scope_name]')
