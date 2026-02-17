# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Contexts for *with* statement providing temporary directories"""

import os
import tempfile
from contextlib import contextmanager

try:
    from contextlib import chdir as _chdir
except ImportError:  # PY310

    @contextmanager  # type: ignore[no-redef]
    def _chdir(path):
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


from .deprecated import deprecate_with_version


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    @deprecate_with_version(
        'Please use the standard library tempfile.TemporaryDirectory',
        '5.0',
        '7.0',
    )
    def __init__(self, suffix='', prefix=tempfile.template, dir=None):
        """
        Examples
        --------
        >>> import os
        >>> with TemporaryDirectory() as tmpdir:
        ...     fname = os.path.join(tmpdir, 'example_file.txt')
        ...     with open(fname, 'wt') as fobj:
        ...         _ = fobj.write('a string\\n')
        >>> os.path.exists(tmpdir)
        False
        """
        super().__init__(suffix, prefix, dir)


@contextmanager
def InTemporaryDirectory():
    """Create, return, and change directory to a temporary directory

    Notes
    -----
    As its name suggests, the class temporarily changes the working
    directory of the Python process, and this is not thread-safe.  We suggest
    using it only for tests.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> my_cwd = os.getcwd()
    >>> with InTemporaryDirectory() as tmpdir:
    ...     _ = Path('test.txt').write_text('some text')
    ...     assert os.path.isfile('test.txt')
    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
    >>> os.path.exists(tmpdir)
    False
    >>> os.getcwd() == my_cwd
    True
    """
    with tempfile.TemporaryDirectory() as tmpdir, _chdir(tmpdir):
        yield tmpdir


@contextmanager
def InGivenDirectory(path=None):
    """Change directory to given directory for duration of ``with`` block

    Useful when you want to use `InTemporaryDirectory` for the final test, but
    you are still debugging.  For example, you may want to do this in the end:

    >>> with InTemporaryDirectory() as tmpdir:
    ...     # do something complicated which might break
    ...     pass

    But indeed the complicated thing does break, and meanwhile the
    ``InTemporaryDirectory`` context manager wiped out the directory with the
    temporary files that you wanted for debugging.  So, while debugging, you
    replace with something like:

    >>> with InGivenDirectory() as tmpdir: # Use working directory by default
    ...     # do something complicated which might break
    ...     pass

    You can then look at the temporary file outputs to debug what is happening,
    fix, and finally replace ``InGivenDirectory`` with ``InTemporaryDirectory``
    again.

    Parameters
    ----------
    path : None or str, optional
        path to change directory to, for duration of ``with`` block.
        Defaults to ``os.getcwd()`` if None
    """
    if path is None:
        path = os.getcwd()
    os.makedirs(path, exist_ok=True)
    with _chdir(path):
        yield os.path.abspath(path)
