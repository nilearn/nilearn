"""Configuration and extra fixtures for pytest."""
import nibabel
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from nilearn import image

# we need to import these fixtures even if not used in this module
from nilearn.datasets._testing import request_mocker  # noqa: F401
from nilearn.datasets._testing import temp_nilearn_data_dir  # noqa: F401
from nilearn.version import _compare_version

collect_ignore = ["datasets/data/convert_templates.py"]


try:
    import matplotlib  # noqa: F401
except ImportError:
    collect_ignore.extend(["plotting", "reporting"])
    matplotlib = None


def pytest_configure(config):
    """Use Agg so that no figures pop up."""
    if matplotlib is not None:
        matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def no_int64_nifti(monkeypatch):
    """Prevent creating or writing a Nift1Image containing 64-bit ints.

    It is easy to create such images by mistake because Numpy uses int64 by
    default, but tools like FSL fail to read them and Nibabel will refuse to
    write them in the future.

    For tests that do need to manipulate int64 images, it is always possible to
    disable this fixture by parametrizing a test to override it:

    @pytest.mark.parametrize("no_int64_nifti", [None])
    def test_behavior_when_user_provides_int64_img():
        # ...

    But by default it is used automatically so that Nilearn doesn't create such
    images by mistake.

    """
    forbidden_types = (np.int64, np.uint64)
    error_msg = (
        "Creating or saving an image " "containing 64-bit ints is forbidden."
    )

    to_filename = nibabel.nifti1.Nifti1Image.to_filename

    def checked_to_filename(img, filename):
        assert image.get_data(img).dtype not in forbidden_types, error_msg
        return to_filename(img, filename)

    monkeypatch.setattr(
        "nibabel.nifti1.Nifti1Image.to_filename", checked_to_filename
    )

    init = nibabel.nifti1.Nifti1Image.__init__

    def checked_init(self, dataobj, *args, **kwargs):
        assert dataobj.dtype not in forbidden_types, error_msg
        return init(self, dataobj, *args, **kwargs)

    monkeypatch.setattr("nibabel.nifti1.Nifti1Image.__init__", checked_init)


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib figures."""
    yield
    if matplotlib is not None:
        import matplotlib.pyplot as plt

        plt.close("all")  # takes < 1 us so just always do it


def pytest_collection_modifyitems(items):
    """Run doctests only for numpy >= 1.14.Adapted from scikit-learn.

    numpy changed the str/repr formatting of numpy arrays in 1.14.
    """
    if _compare_version(np.__version__, "<", "1.14"):
        reason = "doctests are only run for numpy >= 1.14"
        skip_doctests = True
    else:
        skip_doctests = False

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
