from distutils.version import LooseVersion

import numpy as np
import pytest

from _pytest.doctest import DoctestItem
from nilearn.datasets import func, utils
from nilearn.datasets.tests import test_utils as tst

try:
    import matplotlib
except ImportError:
    collect_ignore = ['reporting',
                      'tests/test_glm_reporter.py',
                      'tests/test_reporting.py',
                      'tests/test_sphinx_report.py',
                      ]
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


@pytest.fixture()
def request_mocker():
    tst.setup_mock(utils, func)
    yield
    tst.teardown_mock(utils, func)
