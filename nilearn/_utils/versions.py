"""Utilities and constants for version comparisons."""

import operator
from typing import Literal

from packaging.version import Version, parse
from sklearn import __version__ as sklearn_version

from nilearn._utils.testing import (
    are_tests_running,
    baseline_generation_running,
)
from nilearn._version import __version__

OPTIONAL_MATPLOTLIB_MIN_VERSION = "3.8.0"

VERSION_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

# We clean up the version number
# to make it more stable during tests
# and less dependent if git tags are present or not.
__short_version__ = (
    "TEST"
    if are_tests_running() or baseline_generation_running()
    else str(Version(__version__).__replace__(local=None, dev=None, pre=None))
)


def compare_version(
    version_a: str,
    operator: Literal["==", "!=", ">", "<", ">=", "<="],
    version_b: str,
) -> bool:
    """Compare two version strings via a user-specified operator.

    .. note::

        This function is inspired from MNE-Python.
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
    if operator not in VERSION_OPERATORS:
        error_msg = "'compare_version' received an unexpected operator "
        raise ValueError(error_msg + operator + ".")
    return VERSION_OPERATORS[operator](parse(version_a), parse(version_b))


SKLEARN_LT_1_6 = compare_version(sklearn_version, "<", "1.6.0")
SKLEARN_GTE_1_7 = compare_version(sklearn_version, ">=", "1.7.0")
SKLEARN_GTE_1_8 = compare_version(sklearn_version, ">=", "1.8.0")
