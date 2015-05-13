""" Misc utilities for the library

Authors: Bertrand Thirion, Matthew Brett, 2015
"""
import sys

py3 = sys.version_info[0] >= 3

def open4csv(fname, mode):
    """ Open filename `fname` for CSV IO in read or write `mode`

    Parameters
    ----------
    fname : str
        filename to open
    mode : {'r', 'w'}
        Mode to open file.  Don't specify binary or text modes; we need to
        chose these according to python version.

    Returns
    -------
    fobj : file object
        open file object; needs to be closed by the caller
    """
    if mode not in ('r', 'w'):
        raise ValueError('Only "r" and "w" allowed for mode')
    if not py3: # Files for csv reading and writing should be binary mode
        return open(fname, mode + 'b')
    return open(fname, mode, newline='')
