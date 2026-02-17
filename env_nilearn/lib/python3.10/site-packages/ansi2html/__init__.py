from __future__ import annotations

from ansi2html.converter import Ansi2HTMLConverter

try:
    # pyright: reportMissingImport=false
    from ansi2html._version import __version__  # mypy: disable
except ImportError:  # pragma: no branch
    try:
        import pkg_resources

        __version__ = pkg_resources.get_distribution("ansi2html").version
    except Exception:  # pylint: disable=broad-except
        # this is the fallback SemVer version picked by setuptools_scm when tag
        # information is not available.
        __version__ = "0.1.dev1"

__all__ = ("Ansi2HTMLConverter", "__version__")
