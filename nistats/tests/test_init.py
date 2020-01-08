import sys
import warnings


# import time warnings don't interfere with warning's tests
import pytest

with warnings.catch_warnings(record=True):
    from nistats import _py34_deprecation_warning
    from nistats import _py2_deprecation_warning
    from nistats import _python_deprecation_warnings


def test_py2_deprecation_warning():
    with pytest.warns(DeprecationWarning, match='Python2 support is deprecated'):
        _py2_deprecation_warning()


def test_py34_deprecation_warning():
    with pytest.warns(DeprecationWarning, match='Python 3.4 support is deprecated'):
        _py34_deprecation_warning()


def test_python_deprecation_warnings():
    if sys.version_info.major == 2:
        with pytest.warns(DeprecationWarning, match='Python2 support is deprecated'):
            _python_deprecation_warnings()
    elif sys.version_info.major == 3 and sys.version_info.minor == 4:
        with pytest.warns(DeprecationWarning, match='Python 3.4 support is deprecated'):
            _python_deprecation_warnings()


def test_warnings_filter_scope():
    """
    Tests that warnings generated at Nistats import in Python 2, 3.4 envs
    do not change the warnings filter for subsequent warnings.
    """
    with warnings.catch_warnings(record=True) as raised_warnings:
        warnings.warn('Dummy warning 1')  # Will be raised.
        warnings.filterwarnings("ignore", message="Dummy warning")
        warnings.warn('Dummy warning 2')  # Will not be raised.
        import nistats  # Irrespective of warning raised in py2, py3.4 ...
        warnings.warn('Dummy warning 3')  # ...this should not be raised.
    assert str(raised_warnings[0].message) == 'Dummy warning 1'
    assert str(raised_warnings[-1].message) != 'Dummy warning 3'
