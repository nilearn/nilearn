"""Configuration and extra fixtures for pytest."""
import nibabel
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn import image

# we need to import these fixtures even if not used in this module
from nilearn.datasets._testing import request_mocker  # noqa: F401
from nilearn.datasets._testing import temp_nilearn_data_dir  # noqa: F401

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
        "Creating or saving an image containing 64-bit ints is forbidden."
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


MNI_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture()
def mni_affine():
    """Return an affine corresponding to 2mm isotropic MNI template."""
    return MNI_AFFINE


@pytest.fixture()
def testdata_3d_for_plotting():
    """A random 3D image for testing figures."""
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    img_3d = Nifti1Image(data_positive, MNI_AFFINE)
    # TODO: return img_3D directly and not a dict
    return {"img": img_3d}


@pytest.fixture()
def testdata_4d_for_plotting():
    """Random 4D images for testing figures for multivolume data."""
    rng = np.random.RandomState(42)
    img_4d = Nifti1Image(rng.uniform(size=(7, 7, 3, 10)), MNI_AFFINE)
    img_4d_long = Nifti1Image(rng.uniform(size=(7, 7, 3, 1777)), MNI_AFFINE)
    img_mask = Nifti1Image(np.ones((7, 7, 3), dtype="uint8"), MNI_AFFINE)
    atlas = np.ones((7, 7, 3), dtype="int32")
    atlas[2:5, :, :] = 2
    atlas[5:8, :, :] = 3
    img_atlas = Nifti1Image(atlas, MNI_AFFINE)
    atlas_labels = {
        "gm": 1,
        "wm": 2,
        "csf": 3,
    }
    # TODO: split into several fixtures
    return {
        "img_4d": img_4d,
        "img_4d_long": img_4d_long,
        "img_mask": img_mask,
        "img_atlas": img_atlas,
        "atlas_labels": atlas_labels,
    }
