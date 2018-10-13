import warnings

from nose.tools import assert_true
from sklearn.externals import six
from nilearn import _py2_deprecation_warning


def test_py2_deprecation_warning():
    if six.PY2:
        with warnings.catch_warnings(record=True) as raised_warnings:
            _py2_deprecation_warning()
            assert_true(raised_warnings[0].category is DeprecationWarning)
