import shutil

from distutils.version import LooseVersion

import numpy as np
import pytest

from _pytest.doctest import DoctestItem

try:
    import matplotlib
except ImportError:
    collect_ignore = ['plotting']
else:
    matplotlib  # Prevents flake8 erring due to unused entities.


def pytest_collection_modifyitems(items):

    # numpy changed the str/repr formatting of numpy arrays in 1.14. We want to
    # run doctests only for numpy >= 1.14.Adapted from scikit-learn
    if LooseVersion(np.__version__) < LooseVersion('1.14'):
        reason = 'doctests are only run for numpy >= 1.14'
        skip_doctests = True
    else:
        skip_doctests = False

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)


@pytest.fixture
def temp_dir_path(tmp_path):
    """ Fixture to create a temporary directory path for tests.

    Cleans up after the tests:
        deletes locally scoped objects from memory;
        removes the tree and its files.

    Yields
    ------

    temp_path: string
        Temporary directory path.
    """
    temp_path = str(tmp_path)
    yield temp_path
    local_objects = locals()
    for local_object_ in local_objects:
        del local_object_
    shutil.rmtree(temp_path)
