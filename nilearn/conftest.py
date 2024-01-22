"""Configuration and extra fixtures for pytest."""
import warnings

import nibabel
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn import image

# we need to import these fixtures even if not used in this module
from nilearn.datasets.tests._testing import request_mocker  # noqa: F401
from nilearn.datasets.tests._testing import temp_nilearn_data_dir  # noqa: F401
from nilearn.surface import InMemoryMesh, SurfaceImage

collect_ignore = ["datasets/data/convert_templates.py"]
collect_ignore_glob = ["reporting/_visual_testing/*"]


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


@pytest.fixture(autouse=True)
def suppress_specific_warning():
    """Ignore internal deprecation warnings."""
    with warnings.catch_warnings():
        messages = (
            "The `darkness` parameter will be deprecated.*|"
            "`legacy_format` will default to `False`.*|"
            "In release 0.13, this fetcher will return a dictionary.*|"
            "The default strategy for standardize.*|"
            "The 'fetch_bids_langloc_dataset' function will be removed.*|"
        )
        warnings.filterwarnings(
            "ignore",
            message=messages,
            category=DeprecationWarning,
        )
        yield


# ------------------------   RNG   ------------------------#


def _rng(seed=42):
    return np.random.default_rng(seed)


@pytest.fixture()
def rng():
    """Return a seeded random number generator."""
    return _rng()


# ------------------------ AFFINES ------------------------#


def _affine_mni():
    """Return an affine corresponding to 2mm isotropic MNI template.

    Mostly used for set up in other fixtures in other testing modules.
    """
    return np.array(
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
    return _affine_mni()


def _affine_eye():
    """Return an identity matrix affine.

    Mostly used for set up in other fixtures in other testing modules.
    """
    return np.eye(4)


@pytest.fixture()
def affine_eye():
    """Return an identity matrix affine."""
    return _affine_eye()


# ------------------------ SHAPES ------------------------#


def _shape_3d_default():
    """Return default shape for a 3D image.

    Mostly used for set up in other fixtures in other testing modules.
    """
    return (10, 10, 10)


def _shape_4d_default():
    """Return default shape for a 4D image.

    Mostly used for set up in other fixtures in other testing modules.
    """
    return (10, 10, 10, 10)


def _shape_4d_long():
    """Return default shape for a long 4D image."""
    return (10, 10, 10, 1500)


@pytest.fixture()
def shape_3d_default():
    """Return default shape for a 3D image."""
    return _shape_3d_default()


@pytest.fixture()
def shape_4d_default():
    """Return default shape for a 4D image."""
    return _shape_4d_default()


@pytest.fixture()
def shape_4d_long():
    """Return long shape for a 4D image."""
    return _shape_4d_long()


def _img_zeros(shape, affine):
    return Nifti1Image(np.zeros(shape), affine)


def _img_ones(shape, affine):
    return Nifti1Image(np.ones(shape), affine)


# ------------------------ 3D IMAGES ------------------------#


def _img_3d_rand(affine=_affine_eye()):
    """Return random 3D Nifti1Image in MNI space.

    Mostly used for set up in other fixtures in other testing modules.
    """
    data = _rng().random(_shape_3d_default())
    return Nifti1Image(data, affine)


@pytest.fixture()
def img_3d_rand_eye():
    """Return random 3D Nifti1Image in MNI space."""
    return _img_3d_rand()


def _img_3d_mni(affine=_affine_mni()):
    data_positive = np.zeros((7, 7, 3))
    rng = _rng()
    data_rng = rng.random((7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    return Nifti1Image(data_positive, affine)


@pytest.fixture()
def img_3d_mni():
    """Return a default random 3D Nifti1Image in MNI space."""
    return _img_3d_mni()


def _img_3d_zeros(shape=_shape_3d_default(), affine=_affine_eye()):
    """Return a default zeros filled 3D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    return _img_zeros(shape, affine)


@pytest.fixture
def img_3d_zeros_eye():
    """Return a zeros-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_zeros()


def _img_3d_ones(shape=_shape_3d_default(), affine=_affine_eye()):
    """Return a ones-filled 3D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    return _img_ones(shape, affine)


@pytest.fixture
def img_3d_ones_eye():
    """Return a ones-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_ones()


@pytest.fixture
def img_3d_ones_mni():
    """Return a ones-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_ones(shape=_shape_3d_default(), affine=_affine_mni())


# ------------------------ 4D IMAGES ------------------------#


def _img_4d_zeros(shape=_shape_4d_default(), affine=_affine_eye()):
    """Return a default zeros filled 4D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    return _img_zeros(shape, affine)


def _img_4d_mni(shape=_shape_4d_default(), affine=_affine_mni()):
    return Nifti1Image(_rng().uniform(size=shape), affine=affine)


@pytest.fixture
def img_4d_zeros_eye():
    """Return a default zeros filled 4D Nifti1Image (identity affine)."""
    return _img_4d_zeros()


@pytest.fixture
def img_4d_ones_eye():
    """Return a default ones filled 4D Nifti1Image (identity affine)."""
    return _img_ones(_shape_4d_default(), _affine_eye())


@pytest.fixture
def img_4d_rand_eye():
    """Return a default random filled 4D Nifti1Image (identity affine)."""
    data = _rng().random(_shape_4d_default())
    return Nifti1Image(data, _affine_eye())


@pytest.fixture
def img_4d_mni():
    """Return a default random filled 4D Nifti1Image."""
    return _img_4d_mni()


@pytest.fixture
def img_4d_long_mni(rng, shape_4d_long, affine_mni):
    """Return a default random filled long 4D Nifti1Image."""
    return Nifti1Image(rng.uniform(size=shape_4d_long), affine=affine_mni)


@pytest.fixture()
def img_atlas(shape_3d_default, affine_mni):
    """Return an atlas and its labels."""
    atlas = np.ones(shape_3d_default, dtype="int32")
    atlas[2:5, :, :] = 2
    atlas[5:8, :, :] = 3
    return {
        "img": Nifti1Image(atlas, affine_mni),
        "labels": {
            "gm": 1,
            "wm": 2,
            "csf": 3,
        },
    }


# ------------------------ SURFACE ------------------------#


@pytest.fixture
def mini_mesh():
    """Small mesh for tests with 2 parts with different numbers of vertices."""
    left_coords = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    left_faces = np.asarray([[1, 0, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    right_coords = (
        np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
        + 2.0
    )
    right_faces = np.asarray(
        [
            [0, 1, 4],
            [0, 3, 1],
            [1, 3, 2],
            [1, 2, 4],
            [2, 3, 4],
            [0, 4, 3],
        ]
    )
    return {
        "left_hemisphere": InMemoryMesh(left_coords, left_faces),
        "right_hemisphere": InMemoryMesh(right_coords, right_faces),
    }


@pytest.fixture
def make_mini_img(mini_mesh):
    """Small surface image for tests."""

    def f(shape=()):
        data = {}
        for i, (key, val) in enumerate(mini_mesh.items()):
            data_shape = tuple(shape) + (val.n_vertices,)
            data_part = (
                np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
            ) * 10**i
            data[key] = data_part
        return SurfaceImage(mini_mesh, data)

    return f


@pytest.fixture
def mini_mask(mini_img):
    """Small surface mask."""
    data = {k: (v > v.ravel()[0]) for k, v in mini_img.data.items()}
    return SurfaceImage(mini_img.mesh, data)


@pytest.fixture
def mini_img(make_mini_img):
    """Small surface image for tests."""
    return make_mini_img()


@pytest.fixture
def flip():
    """Reorder dictionary keys by putting the last key as the first."""

    def f(parts):
        if not parts:
            return {}
        keys = list(parts.keys())
        keys = [keys[-1]] + keys[:-1]
        return dict(zip(keys, parts.values()))

    return f


@pytest.fixture
def flip_img(flip):
    """Reorder dictionary keys by putting the last key as the first for mesh \
       and data parts of SurfaceImage object."""

    def f(img):
        return SurfaceImage(flip(img.mesh), flip(img.data))

    return f


@pytest.fixture
def assert_img_equal():
    """Check two surface image objects are equivalent."""

    def f(img_1, img_2):
        assert set(img_1.data.keys()) == set(img_2.data.keys())
        for key in img_1.data:
            assert np.array_equal(img_1.data[key], img_2.data[key])

    return f


@pytest.fixture
def drop_img_part():
    """Drop right hemisphere part of surface image object."""

    def f(img, part_name="right_hemisphere"):
        mesh = img.mesh.copy()
        mesh.pop(part_name)
        data = img.data.copy()
        data.pop(part_name)
        return SurfaceImage(mesh, data)

    return f
