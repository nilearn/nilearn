"""nilearn version, required package versions, and utilities for checking."""

# Author: Loic Esteve, Ben Cipollini
# License: simplified BSD

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
import operator

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "0+unknown"


OPTIONAL_MATPLOTLIB_MIN_VERSION = "3.3.0"

VERSION_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def _compare_version(version_a, operator, version_b):
    """Compare two version strings via a user-specified operator.

    Note: This function is inspired from MNE-Python.
    See https://github.com/mne-tools/mne-python/blob/main/mne/fixes.py

    Parameters
    ----------
    version_a : :obj:`str`
        First version string.

    operator : {'==', '!=','>', '<', '>=', '<='}
        Operator to compare ``version_a`` and ``version_b`` in the form of
        ``version_a operator version_b``.

    version_b : :obj:`str`
        Second version string.

    Returns
    -------
    result : :obj:`bool`
        The result of the version comparison.

    """
    from packaging.version import parse

    if operator not in VERSION_OPERATORS:
        error_msg = "'_compare_version' received an unexpected operator "
        raise ValueError(error_msg + operator + ".")
    return VERSION_OPERATORS[operator](parse(version_a), parse(version_b))
