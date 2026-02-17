# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Settings from the system environment relevant to NIPY"""

import os
from os.path import join as pjoin


def get_home_dir():
    """Return the closest possible equivalent to a 'home' directory.

    The path may not exist; code using this routine should not
    expect the directory to exist.

    Parameters
    ----------
    None

    Returns
    -------
    home_dir : string
       best guess at location of home directory
    """
    return os.path.expanduser('~')


def get_nipy_user_dir():
    """Get the NIPY user directory

    This uses the logic in `get_home_dir` to find the home directory
    and the adds either .nipy or _nipy to the end of the path.

    We check first in environment variable ``NIPY_USER_DIR``, otherwise
    returning the default of ``<homedir>/.nipy`` (Unix) or
    ``<homedir>/_nipy`` (Windows)

    The path may well not exist; code using this routine should not
    expect the directory to exist.

    Parameters
    ----------
    None

    Returns
    -------
    nipy_dir : string
       path to user's NIPY configuration directory

    Examples
    --------
    >>> pth = get_nipy_user_dir()

    """
    try:
        return os.path.abspath(os.environ['NIPY_USER_DIR'])
    except KeyError:
        pass
    home_dir = get_home_dir()
    if os.name == 'posix':
        sdir = '.nipy'
    else:
        sdir = '_nipy'
    return pjoin(home_dir, sdir)


def get_nipy_system_dir():
    r"""Get systemwide NIPY configuration file directory

    On posix systems this will be ``/etc/nipy``.
    On Windows, the directory is less useful, but by default it will be
    ``C:\etc\nipy``

    The path may well not exist; code using this routine should not
    expect the directory to exist.

    Parameters
    ----------
    None

    Returns
    -------
    nipy_dir : string
       path to systemwide NIPY configuration directory

    Examples
    --------
    >>> pth = get_nipy_system_dir()
    """
    if os.name == 'nt':
        return r'C:\etc\nipy'
    if os.name == 'posix':
        return '/etc/nipy'
