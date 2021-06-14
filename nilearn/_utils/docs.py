# -*- coding: utf-8 -*-
"""Functions related to the documentation.

docdict contains the standard documentation entries
used accross Nilearn.

source: Eric Larson and MNE-python team.
https://github.com/mne-tools/mne-python/blob/main/mne/utils/docs.py
"""

import sys

###################################
# Standard documentation entries
#
docdict = dict()

# Verbose
docdict['verbose'] = """
verbose : int, optional
    Verbosity level (0 means no message).
    Default=1."""

# Resume
docdict['resume'] = """
resume : bool, optional
    Whether to resumed download of a partly-downloaded file.
    Default=True."""

# Data_dir
docdict['data_dir'] = """
data_dir : string, optional
    Path where data should be downloaded. By default,
    files are downloaded in home directory."""

# URL
docdict['url'] = """
url : string, optional
    URL of file to download.
    Override download URL. Used for test only (or if you
    setup a mirror of the data).
    Default: None."""

docdict_indented = {}

def _indentcount_lines(lines):
    """Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0

    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.

    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f

