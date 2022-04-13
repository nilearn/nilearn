
import numpy as np
import nibabel
import pytest

from _pytest.doctest import DoctestItem

# we need to import these fixtures even if not used in this module
from nilearn.datasets._testing import request_mocker  # noqa: F401
from nilearn.datasets._testing import temp_nilearn_data_dir  # noqa: F401
from nilearn import image
from nilearn.version import _compare_version


collect_ignore = ["datasets/data/convert_templates.py"]


try:
    import matplotlib  # noqa: F401
except ImportError:
    collect_ignore.extend(['plotting', 'reporting'])
    matplotlib = None


def pytest_configure(config):
    """Use Agg so that no figures pop up."""
    if matplotlib is not None:
        matplotlib.use('Agg', force=True)


@pytest.fixture(autouse=True)
def no_int64_nifti(monkeypatch):
    to_filename = nibabel.Nifti1Image.to_filename

    def checked_to_filename(img, filename):
        assert image.get_data(img).dtype != np.int64
        return to_filename(img, filename)

    monkeypatch.setattr("nibabel.Nifti1Image.to_filename", checked_to_filename)


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib figures."""
    yield
    if matplotlib is not None:
        import matplotlib.pyplot as plt
        plt.close('all')  # takes < 1 us so just always do it


def pytest_collection_modifyitems(items):
    # numpy changed the str/repr formatting of numpy arrays in 1.14.
    # We want to run doctests only for numpy >= 1.14.Adapted from scikit-learn
    if _compare_version(np.__version__, '<', '1.14'):
        reason = 'doctests are only run for numpy >= 1.14'
        skip_doctests = True
    else:
        skip_doctests = False

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
