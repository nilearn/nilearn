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


# ------------------------ AFFINES ------------------------#

AFFINE_MNI = np.array(
    [
        [2.0, 0.0, 0.0, -98.0],
        [0.0, 2.0, 0.0, -134.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture()
def affine_mni():
    """Return an affine corresponding to 2mm isotropic MNI template."""
    return np.array(
        [
            [2.0, 0.0, 0.0, -98.0],
            [0.0, 2.0, 0.0, -134.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


AFFINE_EYE = np.eye(4)


def _affine_eye():
    return np.eye(4)


@pytest.fixture()
def affine_eye():
    """Return an identity matrix affine."""
    return _affine_eye()


# ------------------------ SHAPES ------------------------#

SHAPE_3D_DEFAULT = (10, 10, 10)


def _shape_3d_default():
    return (10, 10, 10)


@pytest.fixture()
def shape_3d_default():
    """Return default shape for a 3D image."""
    return _shape_3d_default()


SHAPE_4D_DEFAULT = (10, 10, 10, 10)


def _shape_4d_default():
    return (10, 10, 10, 10)


@pytest.fixture()
def shape_4d_default():
    """Return default shape for a 4D image."""
    return _shape_4d_default()


def _img_zeros(shape, affine):
    return Nifti1Image(np.zeros(shape), affine)


def _img_ones(shape, affine):
    return Nifti1Image(np.ones(shape), affine)


# ------------------------ 3D IMAGES ------------------------#


def _img_3d_rand(affine=AFFINE_EYE):
    rng = np.random.RandomState(42)
    data = rng.rand(*SHAPE_3D_DEFAULT)
    return Nifti1Image(data, affine)


@pytest.fixture()
def img_3d_rand_eye():
    """Return random 3D Nifti1Image in MNI space."""
    return _img_3d_rand()


def _img_3d_mni(affine=AFFINE_MNI):
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.rand(7, 7, 3)
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    return Nifti1Image(data_positive, affine)


@pytest.fixture()
def img_3d_mni():
    """Return a default random 3D Nifti1Image in MNI space."""
    return _img_3d_mni()


def _img_3d_zeros(shape=SHAPE_3D_DEFAULT, affine=AFFINE_EYE):
    return _img_zeros(shape, affine)


@pytest.fixture
def img_3d_zeros_eye():
    """Return a default zeros filled 3D Nifti1Image (identity affine)."""
    return _img_3d_zeros()


def _img_3d_ones(shape=SHAPE_3D_DEFAULT, affine=AFFINE_EYE):
    return _img_ones(shape, affine)


# ------------------------ 4D IMAGES ------------------------#


def _img_4d_zeros(shape=SHAPE_4D_DEFAULT, affine=AFFINE_EYE):
    return _img_zeros(shape, affine)


@pytest.fixture
def img_4d_zeros_eye():
    """Return a default zeros filled 4D Nifti1Image (identity affine)."""
    return _img_4d_zeros()


@pytest.fixture
def img_4d_ones_eye():
    """Return a default ones filled 4D Nifti1Image (identity affine)."""
    return _img_ones(SHAPE_4D_DEFAULT, AFFINE_EYE)


@pytest.fixture
def img_4D_rand_eye():
    """Return a default random filled 4D Nifti1Image (identity affine)."""
    rng = np.random.RandomState(42)
    data = rng.rand(*SHAPE_4D_DEFAULT)
    return Nifti1Image(data, AFFINE_EYE)


@pytest.fixture()
def testdata_4d_for_plotting():
    """Random 4D images for testing figures for multivolume data."""
    rng = np.random.RandomState(42)
    img_4d = Nifti1Image(rng.uniform(size=(7, 7, 3, 10)), AFFINE_MNI)
    img_4d_long = Nifti1Image(rng.uniform(size=(7, 7, 3, 1777)), AFFINE_MNI)
    img_mask = Nifti1Image(np.ones((7, 7, 3), dtype="uint8"), AFFINE_MNI)
    atlas = np.ones((7, 7, 3), dtype="int32")
    atlas[2:5, :, :] = 2
    atlas[5:8, :, :] = 3
    img_atlas = Nifti1Image(atlas, AFFINE_MNI)
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
