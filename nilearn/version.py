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
__version__ = "0.10.1.dev"

_NILEARN_INSTALL_MSG = "See %s for installation information." % (
    "https://nilearn.github.io/stable/introduction.html#installation"
)

import operator

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    (
        "numpy",
        {
            "min_version": "1.19",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "scipy",
        {
            "min_version": "1.6",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "sklearn",
        {
            "min_version": "1.0.0",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "joblib",
        {
            "min_version": "1.0.0",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    ("nibabel", {"min_version": "3.2", "required_at_installation": False}),
    (
        "pandas",
        {
            "min_version": "1.1.5",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    ("requests", {"min_version": "2.25", "required_at_installation": False}),
)

OPTIONAL_MATPLOTLIB_MIN_VERSION = "3.3.0"


def _import_module_with_version_check(
    module_name, minimum_version, install_info=None
):
    """Check that module is installed with a recent enough version."""
    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{}" could not be found. {}').format(
            module_name,
            install_info or "Please install it properly to use nilearn.",
        )
        exc.args += (user_friendly_info,)
        # Necessary for Python 3 because the repr/str of ImportError
        # objects was changed in Python 3
        if hasattr(exc, "msg"):
            exc.msg += ". " + user_friendly_info
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, "__version__", "0.0.0")

    version_too_old = not _compare_version(
        module_version, ">=", minimum_version
    )

    if version_too_old:
        message = (
            "A {module_name} version of at least {minimum_version} "
            "is required to use nilearn. {module_version} was found. "
            "Please upgrade {module_name}"
        ).format(
            module_name=module_name,
            minimum_version=minimum_version,
            module_version=module_version,
        )

        raise ImportError(message)

    return module


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


def _check_module_dependencies(is_nilearn_installing=False):
    """Throw an exception if nilearn dependencies are not installed.

    Parameters
    ----------
    is_nilearn_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.

    Throws
    -------
    ImportError
    """
    for module_name, module_metadata in REQUIRED_MODULE_METADATA:
        if not (
            is_nilearn_installing
            and not module_metadata["required_at_installation"]
        ):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata["min_version"],
                install_info=module_metadata.get("install_info"),
            )
