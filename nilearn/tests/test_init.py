import sys
import warnings

import pytest

import nilearn


def test_version_number():
    try:
        assert nilearn.__version__ == nilearn._version.__version__
    except AttributeError:
        assert nilearn.__version__ == "0+unknown"


with warnings.catch_warnings(record=True):
    from nilearn import _py37_deprecation_warning, _python_deprecation_warnings


def test_py37_deprecation_warning():
    with pytest.warns(FutureWarning, match="Python 3.7 support is deprecated"):
        _py37_deprecation_warning()


def test_python_deprecation_warnings():
    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        with pytest.warns(
            FutureWarning, match="Python 3.6 support is deprecated"
        ):
            _python_deprecation_warnings()
