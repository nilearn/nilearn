# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Helper utilities to be used in cmdline applications
"""

# global verbosity switch
import re
from io import StringIO
from math import ceil

import numpy as np

verbose_level = 0


def _err(msg=None):
    """To return a string to signal "error" in output table"""
    if msg is None:
        msg = 'error'
    return '!' + msg


def verbose(thing, msg):
    """Print `s` if `thing` is less than the `verbose_level`"""
    # TODO: consider using nibabel's logger
    if thing <= verbose_level:
        print(' ' * thing + msg)


def table2string(table, out=None):
    """Given list of lists figure out their common widths and print to out

    Parameters
    ----------
    table : list of lists of strings
      What is aimed to be printed
    out : None or stream
      Where to print. If None -- will print and return string

    Returns
    -------
    string if out was None
    """

    print2string = out is None
    if print2string:
        out = StringIO()

    # equalize number of elements in each row
    nelements_max = len(table) and max(len(x) for x in table)

    for i, table_ in enumerate(table):
        table[i] += [''] * (nelements_max - len(table_))

    # figure out lengths within each column
    atable = np.asarray(table)
    # eat whole entry while computing width for @w (for wide)
    markup_strip = re.compile('^@([lrc]|w.*)')
    col_width = [max(len(markup_strip.sub('', x)) for x in column) for column in atable.T]
    string = ''
    for i, table_ in enumerate(table):
        string_ = ''
        for j, item in enumerate(table_):
            item = str(item)
            if item.startswith('@'):
                align = item[1]
                item = item[2:]
                if align not in ('l', 'r', 'c', 'w'):
                    raise ValueError(f'Unknown alignment {align}. Known are l,r,c')
            else:
                align = 'c'

            nspacesl = max(ceil((col_width[j] - len(item)) / 2.0), 0)
            nspacesr = max(col_width[j] - nspacesl - len(item), 0)

            if align in ('w', 'c'):
                pass
            elif align == 'l':
                nspacesl, nspacesr = 0, nspacesl + nspacesr
            elif align == 'r':
                nspacesl, nspacesr = nspacesl + nspacesr, 0
            else:
                raise RuntimeError(f'Should not get here with align={align}')

            string_ += '%%%ds%%s%%%ds ' % (nspacesl, nspacesr) % ('', item, '')
        string += string_.rstrip() + '\n'
    out.write(string)

    if print2string:
        value = out.getvalue()
        out.close()
        return value


def ap(helplist, format_, sep=', '):
    """Little helper to enforce consistency"""
    if helplist == '-':
        return helplist
    ls = [format_ % x for x in helplist]
    return sep.join(ls)


def safe_get(obj, name):
    """A getattr which would return '-' if getattr fails"""
    try:
        f = getattr(obj, 'get_' + name)
        return f()
    except Exception as e:
        verbose(2, f'get_{name}() failed -- {e}')
        return '-'
