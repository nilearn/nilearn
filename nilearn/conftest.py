from distutils.version import LooseVersion

import numpy as np
import pytest

from _pytest.doctest import DoctestItem

try:
    import matplotlib  # noqa: F401
except ImportError:
    collect_ignore = ['plotting']


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
