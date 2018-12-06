import sys
import warnings

from nose.tools import assert_true
from nistats import _py2_deprecation_warning, _py34_deprecation_warning


def test_py2_deprecation_warning():
    if sys.version_info.major == 2:
        with warnings.catch_warnings(record=True) as raised_warnings:
            _py2_deprecation_warning()
            assert_true(raised_warnings[0].category is DeprecationWarning)


def test_py34_deprecation_warning():
    if sys.version_info.major == 3 and sys.version_info.minor == 4:
        with warnings.catch_warnings(record=True) as raised_warnings:
            _py34_deprecation_warning()
            assert_true(raised_warnings[0].category is DeprecationWarning)
