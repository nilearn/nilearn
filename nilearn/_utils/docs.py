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
    Verbosity level (0 means no message)."""

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
    Default=None."""

# Smoothing_fwhm
docdict['smoothing_fwhm'] = """
smoothing_fwhm : float, optional.
    If smoothing_fwhm is not None, it gives the size in millimeters of the
    spatial smoothing to apply to the signal.
    Default=None."""

# Standardize
docdict['standardize'] = """
standardize : bool, optional.
    If standardize is True, the time-series are centered and normed:
    their variance is put to 1 in the time dimension.
    Default=True."""

# Target_affine
docdict['target_affine'] = """
target_affine: 3x3 or 4x4 matrix, optional.
    This parameter is passed to image.resample_img. Please see the
    related documentation for details.
    Default=None."""

# Target_shape
docdict['target_shape'] = """
target_shape: 3-tuple of int, optional.
    This parameter is passed to image.resample_img. Please see the
    related documentation for details.
    Default=None."""

# Low_pass
docdict['low_pass'] = """
low_pass: float, optional
    Low cutoff frequency in Hertz.
    Default=None."""

# High pass
docdict['high_pass'] = """
high_pass: float, optional
    High cutoff frequency in Hertz.
    Default=None."""

# t_r
docdict['t_r'] = """
t_r: float, optional
    Repetition time, in second (sampling period). Set to None if not.
    Default=None."""

# Memory
docdict['memory'] = """
memory : instance of joblib.Memory or str
    Used to cache the masking process.
    By default, no caching is done. If a str is given, it is the
    path to the caching directory."""

# Memory_level
docdict['memory_level'] = """
memory_level: int, optional.
    Rough estimator of the amount of memory used by caching. Higher value
    means more memory for caching.
    Default=0."""

# n_jobs
docdict['n_jobs'] = """
n_jobs : int, optional.
    The number of CPUs to use to do the computation. -1 means 'all CPUs'.."""


docdict_indented = {}


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(['    '])
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
