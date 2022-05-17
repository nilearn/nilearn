import sys
import warnings
import pytest


with warnings.catch_warnings(record=True):
    from nilearn import (
        _py36_deprecation_warning, _python_deprecation_warnings
    )


def test_py36_deprecation_warning():
    with pytest.warns(FutureWarning,
                      match="Python 3.6 support is deprecated"):
        _py36_deprecation_warning()


def test_python_deprecation_warnings():
    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        with pytest.warns(FutureWarning,
                          match="Python 3.6 support is deprecated"):
            _python_deprecation_warnings()
