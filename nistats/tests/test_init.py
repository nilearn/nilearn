import sys
import warnings

from nose.tools import assert_true


def test_py2_deprecation_warning():
    from nistats import _py2_deprecation_warning
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py2_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)


def test_py34_deprecation_warning():
    from nistats import _py34_deprecation_warning
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py34_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)


def test_python_deprecation_warnings():
    from nistats import _python_deprecation_warnings
    with warnings.catch_warnings(record=True) as raised_warnings:
        _python_deprecation_warnings()
    if sys.version_info.major == 2:
        assert_true(raised_warnings[0].category is DeprecationWarning)
        assert_true(
                str(raised_warnings[0].message).startswith(
                        'Python2 support is deprecated')
                )
    elif sys.version_info.major == 3 and sys.version_info.minor == 4:
        assert_true(raised_warnings[0].category is DeprecationWarning)
        assert_true(
                str(raised_warnings[0].message).startswith(
                        'Python 3.4 support is deprecated')
                )
    else:
        assert_true(len(raised_warnings) == 0)
