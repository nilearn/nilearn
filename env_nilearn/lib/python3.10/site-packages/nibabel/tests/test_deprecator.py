"""Testing deprecator module / Deprecator class"""

import sys
import warnings
from functools import partial
from textwrap import indent

import pytest

from nibabel.deprecator import (
    TESTCLEANUP,
    TESTSETUP,
    Deprecator,
    ExpiredDeprecationError,
    _add_dep_doc,
    _dedent_docstring,
    _ensure_cr,
)

from ..testing import clear_and_catch_warnings

_OWN_MODULE = sys.modules[__name__]

func_docstring = (
    f'A docstring\n   \n   foo\n   \n{indent(TESTSETUP, "   ", lambda x: True)}'
    f'   Some text\n{indent(TESTCLEANUP, "   ", lambda x: True)}'
)

if sys.version_info >= (3, 13):
    func_docstring = _dedent_docstring(func_docstring)


def test__ensure_cr():
    # Make sure text ends with carriage return
    assert _ensure_cr('  foo') == '  foo\n'
    assert _ensure_cr('  foo\n') == '  foo\n'
    assert _ensure_cr('  foo  ') == '  foo\n'
    assert _ensure_cr('foo  ') == 'foo\n'
    assert _ensure_cr('foo  \n bar') == 'foo  \n bar\n'
    assert _ensure_cr('foo  \n\n') == 'foo\n'


def test__add_dep_doc():
    # Test utility function to add deprecation message to docstring
    assert _add_dep_doc('', 'foo') == 'foo\n'
    assert _add_dep_doc('bar', 'foo') == 'bar\n\nfoo\n'
    assert _add_dep_doc('   bar', 'foo') == '   bar\n\nfoo\n'
    assert _add_dep_doc('   bar', 'foo\n') == '   bar\n\nfoo\n'
    assert _add_dep_doc('bar\n\n', 'foo') == 'bar\n\nfoo\n'
    assert _add_dep_doc('bar\n    \n', 'foo') == 'bar\n\nfoo\n'
    assert (
        _add_dep_doc(' bar\n\nSome explanation', 'foo\nbaz')
        == ' bar\n\nfoo\nbaz\n\nSome explanation\n'
    )
    assert (
        _add_dep_doc(' bar\n\n  Some explanation', 'foo\nbaz')
        == ' bar\n  \n  foo\n  baz\n  \n  Some explanation\n'
    )


class CustomError(Exception):
    """Custom error class for testing expired deprecation errors"""


def cmp_func(v):
    """Comparison func tests against version 2.0"""
    return (float(v) > 2) - (float(v) < 2)


def func_no_doc():
    pass


def func_doc(i):
    """A docstring"""


def func_doc_long(i, j):
    """A docstring\n\n   Some text"""


class TestDeprecatorFunc:
    """Test deprecator function specified in ``dep_func``"""

    dep_func = Deprecator(cmp_func)

    def test_dep_func(self):
        # Test function deprecation
        dec = self.dep_func
        func = dec('foo')(func_no_doc)
        with pytest.deprecated_call():
            assert func() is None
        assert func.__doc__ == 'foo\n'
        func = dec('foo')(func_doc)
        with pytest.deprecated_call() as w:
            assert func(1) is None
            assert len(w) == 1
        assert func.__doc__ == 'A docstring\n\nfoo\n'
        func = dec('foo')(func_doc_long)
        with pytest.deprecated_call() as w:
            assert func(1, 2) is None
            assert len(w) == 1
        assert func.__doc__ == func_docstring

        # Try some since and until versions
        func = dec('foo', '1.1')(func_no_doc)
        assert func.__doc__ == 'foo\n\n* deprecated from version: 1.1\n'
        with pytest.deprecated_call() as w:
            assert func() is None
            assert len(w) == 1
        func = dec('foo', until='99.4')(func_no_doc)
        with pytest.deprecated_call() as w:
            assert func() is None
            assert len(w) == 1
        assert (
            func.__doc__ == f'foo\n\n* Will raise {ExpiredDeprecationError} as of version: 99.4\n'
        )
        func = dec('foo', until='1.8')(func_no_doc)
        with pytest.raises(ExpiredDeprecationError):
            func()
        assert func.__doc__ == f'foo\n\n* Raises {ExpiredDeprecationError} as of version: 1.8\n'
        func = dec('foo', '1.2', '1.8')(func_no_doc)
        with pytest.raises(ExpiredDeprecationError):
            func()
        assert (
            func.__doc__ == 'foo\n\n* deprecated from version: 1.2\n* Raises '
            f'{ExpiredDeprecationError} as of version: 1.8\n'
        )
        func = dec('foo', '1.2', '1.8')(func_doc_long)
        assert (
            func.__doc__
            == f"""\
A docstring

foo

* deprecated from version: 1.2
* Raises {ExpiredDeprecationError} as of version: 1.8
"""
        )
        with pytest.raises(ExpiredDeprecationError):
            func()

        # Check different warnings and errors
        func = dec('foo', warn_class=UserWarning)(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert func() is None
            assert len(w) == 1
            assert w[0].category is UserWarning

        func = dec('foo', error_class=CustomError)(func_no_doc)
        with pytest.deprecated_call():
            assert func() is None

        func = dec('foo', until='1.8', error_class=CustomError)(func_no_doc)
        with pytest.raises(CustomError):
            func()


class TestDeprecatorMaker:
    """Test deprecator class creation with custom warnings and errors"""

    dep_maker = partial(Deprecator, cmp_func)

    def test_deprecator_maker(self):
        dec = self.dep_maker(warn_class=UserWarning)
        func = dec('foo')(func_no_doc)
        with pytest.warns(UserWarning) as w:
            # warnings.simplefilter('always')
            assert func() is None
            assert len(w) == 1

        dec = self.dep_maker(error_class=CustomError)
        func = dec('foo')(func_no_doc)
        with pytest.deprecated_call():
            assert func() is None

        func = dec('foo', until='1.8')(func_no_doc)
        with pytest.raises(CustomError):
            func()
