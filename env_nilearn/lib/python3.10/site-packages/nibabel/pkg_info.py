from __future__ import annotations

import sys
from contextlib import suppress
from subprocess import run

from packaging.version import Version

try:
    from ._version import __version__
except ImportError:
    __version__ = '0+unknown'


COMMIT_HASH = '$Format:%h$'


def _cmp(a: Version, b: Version) -> int:
    """Implementation of ``cmp`` for Python 3"""
    return (a > b) - (a < b)


def cmp_pkg_version(version_str: str, pkg_version_str: str = __version__) -> int:
    """Compare ``version_str`` to current package version

    This comparator follows `PEP-440`_ conventions for determining version
    ordering.

    To be valid, a version must have a numerical major version. It may be
    optionally followed by a dot and a numerical minor version, which may,
    in turn, optionally be followed by a dot and a numerical micro version,
    and / or by an "extra" string.
    The extra string may further contain a "+". Any value to the left of a "+"
    labels the version as pre-release, while values to the right indicate a
    post-release relative to the values to the left. That is,
    ``1.2.0+1`` is post-release for ``1.2.0``, while ``1.2.0rc1+1`` is
    post-release for ``1.2.0rc1`` and pre-release for ``1.2.0``.

    Parameters
    ----------
    version_str : str
        Version string to compare to current package version
    pkg_version_str : str, optional
        Version of our package.  Optional, set from ``__version__`` by default.

    Returns
    -------
    version_cmp : int
        1 if `version_str` is a later version than `pkg_version_str`, 0 if
        same, -1 if earlier.

    Examples
    --------
    >>> cmp_pkg_version('1.2.1', '1.2.0')
    1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0rc1')
    -1
    >>> cmp_pkg_version('1.2.0rc1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0rc1')
    1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0.post1', '1.2.0')
    1

    .. _`PEP-440`: https://www.python.org/dev/peps/pep-0440/
    """
    return _cmp(Version(version_str), Version(pkg_version_str))


def pkg_commit_hash(pkg_path: str | None = None) -> tuple[str, str]:
    """Get short form of commit hash

    In this file is a variable called COMMIT_HASH. This contains a substitution
    pattern that may have been filled by the execution of ``git archive``.

    We get the commit hash from (in order of preference):

    * A substituted value in ``archive_subst_hash``
    * A truncated commit hash value that is part of the local portion of the
      version
    * git's output, if we are in a git repository

    If all these fail, we return a not-found placeholder tuple

    Parameters
    ----------
    pkg_path : str
       directory containing package

    Returns
    -------
    hash_from : str
       Where we got the hash from - description
    hash_str : str
       short form of hash
    """
    if not COMMIT_HASH.startswith('$Format'):  # it has been substituted
        return 'archive substitution', COMMIT_HASH
    ver = Version(__version__)
    if ver.local is not None and ver.local.startswith('g'):
        return 'installation', ver.local[1:8]
    # maybe we are in a repository, but consider that we may not have git
    with suppress(FileNotFoundError):
        proc = run(
            ('git', 'rev-parse', '--short', 'HEAD'),
            capture_output=True,
            cwd=pkg_path,
        )
        if proc.stdout:
            return 'repository', proc.stdout.decode().strip()

    return '(none found)', '<not found>'


def get_pkg_info(pkg_path: str) -> dict[str, str]:
    """Return dict describing the context of this package

    Parameters
    ----------
    pkg_path : str
       path containing __init__.py for package

    Returns
    -------
    context : dict
       with named parameters of interest
    """
    src, hsh = pkg_commit_hash(pkg_path)
    import numpy

    return dict(
        pkg_path=pkg_path,
        commit_source=src,
        commit_hash=hsh,
        sys_version=sys.version,
        sys_executable=sys.executable,
        sys_platform=sys.platform,
        np_version=numpy.__version__,
    )
