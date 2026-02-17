"""Class for recording and reporting deprecations"""

from __future__ import annotations

import functools
import re
import sys
import typing as ty
import warnings
from textwrap import dedent

if ty.TYPE_CHECKING:
    T = ty.TypeVar('T')
    P = ty.ParamSpec('P')

_LEADING_WHITE = re.compile(r'^(\s*)')


def _dedent_docstring(docstring):
    """Compatibility with Python 3.13+.

    xref: https://github.com/python/cpython/issues/81283
    """
    return '\n'.join([dedent(line) for line in docstring.split('\n')])


TESTSETUP = """

.. testsetup::

    >>> import pytest
    >>> import warnings
    >>> _suppress_warnings = pytest.deprecated_call()
    >>> _ = _suppress_warnings.__enter__()

"""

TESTCLEANUP = """

.. testcleanup::

    >>> warnings.warn("Avoid error if no doctests to run...", DeprecationWarning)
    >>> _ = _suppress_warnings.__exit__(None, None, None)

"""

if sys.version_info >= (3, 13):
    TESTSETUP = _dedent_docstring(TESTSETUP)
    TESTCLEANUP = _dedent_docstring(TESTCLEANUP)


class ExpiredDeprecationError(RuntimeError):
    """Error for expired deprecation

    Error raised when a called function or method has passed out of its
    deprecation period.
    """

    pass


def _ensure_cr(text: str) -> str:
    """Remove trailing whitespace and add carriage return

    Ensures that `text` always ends with a carriage return
    """
    return text.rstrip() + '\n'


def _add_dep_doc(
    old_doc: str,
    dep_doc: str,
    setup: str = '',
    cleanup: str = '',
) -> str:
    """Add deprecation message `dep_doc` to docstring in `old_doc`

    Parameters
    ----------
    old_doc : str
        Docstring from some object.
    dep_doc : str
        Deprecation warning to add to top of docstring, after initial line.
    setup : str, optional
        Doctest setup text
    cleanup : str, optional
        Doctest teardown text

    Returns
    -------
    new_doc : str
        `old_doc` with `dep_doc` inserted after any first lines of docstring.
    """
    dep_doc = _ensure_cr(dep_doc)
    if not old_doc:
        return dep_doc
    old_doc = _ensure_cr(old_doc)
    old_lines = old_doc.splitlines()
    new_lines = []
    for line_no, line in enumerate(old_lines):
        if line.strip():
            new_lines.append(line)
        else:
            break
    next_line = line_no + 1
    if next_line >= len(old_lines):
        # nothing following first paragraph, just append message
        return old_doc + '\n' + dep_doc
    leading_white = _LEADING_WHITE.match(old_lines[next_line])
    assert leading_white is not None  # Type narrowing, since this always matches
    indent = leading_white.group()
    setup_lines = [indent + L for L in setup.splitlines()]
    dep_lines = [indent + L for L in [''] + dep_doc.splitlines() + ['']]
    cleanup_lines = [indent + L for L in cleanup.splitlines()]
    return '\n'.join(
        new_lines + dep_lines + setup_lines + old_lines[next_line:] + cleanup_lines + ['']
    )


class Deprecator:
    """Class to make decorator marking function or method as deprecated

    The decorated function / method will:

    * Raise the given `warning_class` warning when the function / method gets
      called, up to (and including) version `until` (if specified);
    * Raise the given `error_class` error when the function / method gets
      called, when the package version is greater than version `until` (if
      specified).

    Parameters
    ----------
    version_comparator : callable
        Callable accepting string as argument, and return 1 if string
        represents a higher version than encoded in the `version_comparator`, 0
        if the version is equal, and -1 if the version is lower.  For example,
        the `version_comparator` may compare the input version string to the
        current package version string.
    warn_class : class, optional
        Class of warning to generate for deprecation.
    error_class : class, optional
        Class of error to generate when `version_comparator` returns 1 for a
        given argument of ``until`` in the ``__call__`` method (see below).
    """

    def __init__(
        self,
        version_comparator: ty.Callable[[str], int],
        warn_class: type[Warning] = DeprecationWarning,
        error_class: type[Exception] = ExpiredDeprecationError,
    ) -> None:
        self.version_comparator = version_comparator
        self.warn_class = warn_class
        self.error_class = error_class

    def is_bad_version(self, version_str: str) -> bool:
        """Return True if `version_str` is too high

        Tests `version_str` with ``self.version_comparator``

        Parameters
        ----------
        version_str : str
            String giving version to test

        Returns
        -------
        is_bad : bool
            True if `version_str` is for version below that expected by
            ``self.version_comparator``, False otherwise.
        """
        return self.version_comparator(version_str) == -1

    def __call__(
        self,
        message: str,
        since: str = '',
        until: str = '',
        warn_class: type[Warning] | None = None,
        error_class: type[Exception] | None = None,
    ) -> ty.Callable[[ty.Callable[P, T]], ty.Callable[P, T]]:
        """Return decorator function function for deprecation warning / error

        Parameters
        ----------
        message : str
            Message explaining deprecation, giving possible alternatives.
        since : str, optional
            Released version at which object was first deprecated.
        until : str, optional
            Last released version at which this function will still raise a
            deprecation warning.  Versions higher than this will raise an
            error.
        warn_class : None or class, optional
            Class of warning to generate for deprecation (overrides instance
            default).
        error_class : None or class, optional
            Class of error to generate when `version_comparator` returns 1 for a
            given argument of ``until`` (overrides class default).

        Returns
        -------
        deprecator : func
            Function returning a decorator.
        """
        exception = error_class if error_class is not None else self.error_class
        warning = warn_class if warn_class is not None else self.warn_class
        messages = [message]
        if (since, until) != ('', ''):
            messages.append('')
        if since:
            messages.append('* deprecated from version: ' + since)
        if until:
            messages.append(
                f"* {'Raises' if self.is_bad_version(until) else 'Will raise'} "
                f'{exception} as of version: {until}'
            )
        message = '\n'.join(messages)

        def deprecator(func: ty.Callable[P, T]) -> ty.Callable[P, T]:
            @functools.wraps(func)
            def deprecated_func(*args: P.args, **kwargs: P.kwargs) -> T:
                if until and self.is_bad_version(until):
                    raise exception(message)
                warnings.warn(message, warning, stacklevel=2)
                return func(*args, **kwargs)

            keep_doc = deprecated_func.__doc__
            if keep_doc is None:
                keep_doc = ''
            setup = TESTSETUP
            cleanup = TESTCLEANUP
            # After expiration, remove all but the first paragraph.
            # The details are no longer relevant, but any code will likely
            # raise exceptions we don't need.
            if keep_doc and until and self.is_bad_version(until):
                lines = '\n'.join(line.rstrip() for line in keep_doc.splitlines())
                keep_doc = lines.split('\n\n', 1)[0]
                setup = ''
                cleanup = ''
            deprecated_func.__doc__ = _add_dep_doc(keep_doc, message, setup, cleanup)
            return deprecated_func

        return deprecator
