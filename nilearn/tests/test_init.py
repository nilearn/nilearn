import sys
import warnings

import pytest

# import time warnings don't interfere with warning's tests
with warnings.catch_warnings(record=True):
    from nilearn import _py35_deprecation_warning
    from nilearn import _python_deprecation_warnings


def test_py35_deprecation_warning():
    with pytest.warns(FutureWarning,
                      match='Python 3.5 support is deprecated'
                      ):
        _py35_deprecation_warning()


def test_python_deprecation_warnings():
    if sys.version_info.major == 3 and sys.version_info.minor == 5:
        with pytest.warns(FutureWarning,
                          match='Python 3.5 support is deprecated'
                          ):
            _python_deprecation_warnings()


def test_warnings_filter_scope():
    """
    Tests that warnings generated at Nilearn import in Python 3.5 envs
    do not change the warnings filter for subsequent warnings.
    """
    with warnings.catch_warnings(record=True) as raised_warnings:
        warnings.warn('Dummy warning 1')  # Will be raised.
        warnings.filterwarnings("ignore", message="Dummy warning")
        warnings.warn('Dummy warning 2')  # Will not be raised.
        import nilearn  # noqa: F401 # Irrespective of warning raised in py3.5
        warnings.warn('Dummy warning 3')  # ...this should not be raised.
    assert str(raised_warnings[0].message) == 'Dummy warning 1'
    assert str(raised_warnings[-1].message) != 'Dummy warning 3'
