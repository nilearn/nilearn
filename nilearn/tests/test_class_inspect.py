"""
Test the class_inspect module

This test file is in nilearn/tests because nosetests seems to ignore modules whose
name starts with an underscore
"""
from nose.tools import assert_equal

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

    def get_scope_name(self, stack=0, *args, **kwargs):
        c = C()
        return c.get_scope_name(stack=stack, *args, **kwargs)


class C:

    def get_scope_name(self, *args, **kwargs):
        return get_scope_name(*args, **kwargs)


def get_scope_name(stack=0, *args, **kwargs):
    if stack == 0:
        return class_inspect.enclosing_scope_name(*args, **kwargs)
    return get_scope_name(stack - 1, *args, **kwargs)


##############################################################################
# The tests themselves

def test_get_params():
    b = B()
    params_a_in_b = class_inspect.get_params(A, b)
    assert_equal(params_a_in_b, dict(a=1))
    params_a_in_b = class_inspect.get_params(A, b, ignore=['a'])
    assert_equal(params_a_in_b, {})


def test_enclosing_scope_name():
    b = B()
    name = b.get_scope_name()
    assert_equal(name, 'B.get_scope_name')
    name = b.get_scope_name(stack=3)
    assert_equal(name, 'B.get_scope_name')
    name = b.get_scope_name(ensure_estimator=False)
    assert_equal(name, 'C.get_scope_name')
    name = b.get_scope_name(stack=3, ensure_estimator=False)
    assert_equal(name, 'get_scope_name')
    name = b.get_scope_name(ensure_estimator=False, stack_level=120)
    assert_equal(name, 'Unknown')
