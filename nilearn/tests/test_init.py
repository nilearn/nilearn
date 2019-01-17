import sys
import warnings
from pprint import pprint

from nose.tools import assert_true


def test_py2_deprecation_warning():
    # Importing module within function's scope, preventing artifacts with
    # raising the warning due to warning filter set to 'once' due to
    # multiple similar tests in a test run.
    from nilearn import _py2_deprecation_warning
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py2_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)


def test_py34_deprecation_warning():
    # Importing module within function's scope, preventing artifacts with
    # raising the warning due to warning filter set to 'once' due to
    # multiple similar tests in a test run.
    from nilearn import _py34_deprecation_warning
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py34_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)


def test_python_deprecation_warnings():
    # Importing module within function's scope, preventing artifacts with
    # raising the warning due to warning filter set to 'once' due to
    # multiple similar tests in a test run.
    from nilearn import _python_deprecation_warnings
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


def test_warnings_filter_scope():
    """
    Tests that warnings generated at Nilearn import in Python 2, 3.4 envs
    do not change the warnings filter for subsequent warnings.
    """
    with warnings.catch_warnings(record=True) as raised_warnings:
        warnings.warn('Dummy warning 1')  # Will be raised.
        warnings.filterwarnings("ignore", message="Dummy warning")
        warnings.warn('Dummy warning 2')  # Will not be raised.
        import nilearn  # Possible warning raised during Nilearn import.
        warnings.warn('Dummy warning 3')  # Should not be raised.
    pprint([warning_.message for warning_ in raised_warnings])
    assert len(raised_warnings) < 3
    assert str(raised_warnings[0].message) == 'Dummy warning 1'
    assert str(raised_warnings[1].message) != 'Dummy warning 2'
    if raised_warnings[1].category is DeprecationWarning:
        assert str(raised_warnings[-1].message) != 'Dummy warning 3'


if __name__ == '__main__':
    test_warnings_filter_scope()
