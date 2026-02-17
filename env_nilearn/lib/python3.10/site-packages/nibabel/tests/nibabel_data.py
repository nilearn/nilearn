"""Functions / decorators for finding / requiring nibabel-data directory"""

import unittest
from os import environ, listdir
from os.path import dirname, exists, isdir, realpath
from os.path import join as pjoin


def get_nibabel_data():
    """Return path to nibabel-data or empty string if missing

    First use ``NIBABEL_DATA_DIR`` environment variable.

    If this variable is missing then look for data in directory below package
    directory.
    """
    nibabel_data = environ.get('NIBABEL_DATA_DIR')
    if nibabel_data is None:
        mod = __import__('nibabel')
        containing_path = dirname(dirname(realpath(mod.__file__)))
        nibabel_data = pjoin(containing_path, 'nibabel-data')
    return nibabel_data if isdir(nibabel_data) else ''


def needs_nibabel_data(subdir=None):
    """Decorator for tests needing nibabel-data

    Parameters
    ----------
    subdir : None or str
        Subdirectory we need in nibabel-data directory.  If None, only require
        nibabel-data directory itself.

    Returns
    -------
    skip_dec : decorator
        Decorator skipping tests if required directory not present
    """
    nibabel_data = get_nibabel_data()
    if nibabel_data == '':
        return unittest.skip('Need nibabel-data directory for this test')
    if subdir is None:
        return lambda x: x
    required_path = pjoin(nibabel_data, subdir)
    # Path should not be empty (as is the case for not-updated submodules)
    have_files = exists(required_path) and len(listdir(required_path)) > 0
    return unittest.skipUnless(have_files, f'Need files in {required_path} for these tests')
