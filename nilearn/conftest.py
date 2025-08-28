"""Configuration and extra fixtures for pytest."""

import warnings

import nibabel
import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image

from nilearn import image
from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
)
from nilearn._utils.helpers import is_matplotlib_installed

# we need to import these fixtures even if not used in this module
from nilearn.datasets.tests._testing import (
    request_mocker,  # noqa: F401
    temp_nilearn_data_dir,  # noqa: F401
)
from nilearn.surface import (
    InMemoryMesh,
    PolyMesh,
    SurfaceImage,
)

collect_ignore = ["datasets/data/convert_templates.py"]
collect_ignore_glob = ["reporting/_visual_testing/*"]

# Plotting tests are skipped if matplotlib is missing.
# If the version is greater than the minimum one we support
# We skip the tests where the generated figures are compared to a baseline.

if is_matplotlib_installed():
    import matplotlib

    from nilearn._utils.helpers import (
        OPTIONAL_MATPLOTLIB_MIN_VERSION,
        compare_version,
    )

    if compare_version(
        matplotlib.__version__, ">", OPTIONAL_MATPLOTLIB_MIN_VERSION
    ):
        # the tests that compare plotted figures
        # against their expected baseline is only run
        # with the oldest version of matplolib
        collect_ignore.extend(
            [
                "plotting/tests/test_baseline_comparisons.py",
                "reporting/tests/test_baseline_comparisons.py",
            ]
        )

else:
    collect_ignore.extend(
        [
            "_utils/plotting.py",
            "plotting",
            "reporting/html_report.py",
            "reporting/tests/test_html_report.py",
            "reporting/tests/test_baseline_comparisons.py",
        ]
    )
    matplotlib = None  # type: ignore[assignment]


def pytest_configure(config):  # noqa: ARG001
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
        # TODO (nilearn >= 0.13.0) deprecate nearest interpolation
        messages = (
            "The `darkness` parameter will be deprecated.*|"
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
    # avoid having identical shapes values,
    # because this fails to detect if the code does not handle dimensions well.
    return (7, 8, 9)


def _shape_3d_large():
    """Shape usually used for maps images.

    Mostly used for set up in other fixtures in other testing modules.
    """
    # avoid having identical shapes values,
    # because this fails to detect if the code does not handle dimensions well.
    return (29, 30, 31)


def _shape_4d_default():
    """Return default shape for a 4D image.

    Mostly used for set up in other fixtures in other testing modules.
    """
    # avoid having identical shapes values,
    # because this fails to detect if the code does not handle dimensions well.
    return (7, 8, 9, 5)


def _shape_4d_medium():
    """Return default shape for a long 4D image."""
    # avoid having identical shapes values,
    # because this fails to detect if the code does not handle dimensions well.
    return (7, 8, 9, 100)


def _shape_4d_long():
    """Return default shape for a long 4D image."""
    # avoid having identical shapes values,
    # because this fails to detect if the code does not handle dimensions well.
    return (7, 8, 9, 1500)


@pytest.fixture()
def shape_3d_default():
    """Return default shape for a 3D image."""
    return _shape_3d_default()


@pytest.fixture
def shape_3d_large():
    """Shape usually used for maps images."""
    return _shape_3d_large()


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


def _img_3d_rand(affine=None):
    """Return random 3D Nifti1Image in MNI space.

    Mostly used for set up in other fixtures in other testing modules.
    """
    if affine is None:
        affine = _affine_eye()
    data = _rng().random(_shape_3d_default())
    return Nifti1Image(data, affine)


@pytest.fixture()
def img_3d_rand_eye():
    """Return random 3D Nifti1Image in MNI space."""
    return _img_3d_rand()


def _img_3d_mni(affine=None):
    if affine is None:
        affine = _affine_mni()
    data_positive = np.zeros((7, 7, 3))
    rng = _rng()
    data_rng = rng.random((7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    return Nifti1Image(data_positive, affine)


@pytest.fixture()
def img_3d_mni():
    """Return a default random 3D Nifti1Image in MNI space."""
    return _img_3d_mni()


@pytest.fixture()
def img_3d_mni_as_file(tmp_path):
    """Return path to a random 3D Nifti1Image in MNI space saved to disk."""
    filename = tmp_path / "img.nii"
    _img_3d_mni().to_filename(filename)
    return filename


def _img_3d_zeros(shape=None, affine=None):
    """Return a default zeros filled 3D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    if shape is None:
        shape = _shape_3d_default()
    if affine is None:
        affine = _affine_eye()
    return _img_zeros(shape, affine)


@pytest.fixture
def img_3d_zeros_eye():
    """Return a zeros-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_zeros()


def _img_3d_ones(shape=None, affine=None):
    """Return a ones-filled 3D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    if shape is None:
        shape = _shape_3d_default()
    if affine is None:
        affine = _affine_eye()
    return _img_ones(shape, affine)


@pytest.fixture
def img_3d_ones_eye():
    """Return a ones-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_ones()


@pytest.fixture
def img_3d_ones_mni():
    """Return a ones-filled 3D Nifti1Image (identity affine)."""
    return _img_3d_ones(shape=_shape_3d_default(), affine=_affine_mni())


def _mask_data():
    mask_data = np.zeros(_shape_3d_default(), dtype="int32")
    mask_data[3:6, 3:6, 3:6] = 1
    return mask_data


def _img_mask_mni():
    """Return a 3D nifti mask in MNI space with some 1s in the center."""
    return Nifti1Image(_mask_data(), _affine_mni())


@pytest.fixture
def img_mask_mni():
    """Return a 3D nifti mask in MNI space with some 1s in the center."""
    return _img_mask_mni()


def _img_mask_eye():
    """Return a 3D nifti mask with identity affine with 1s in the center."""
    return Nifti1Image(_mask_data(), _affine_eye())


@pytest.fixture
def img_mask_eye():
    """Return a 3D nifti mask with identity affine with 1s in the center."""
    return _img_mask_eye()


# ------------------------ 4D IMAGES ------------------------#


def _img_4d_zeros(shape=None, affine=None):
    """Return a default zeros filled 4D Nifti1Image (identity affine).

    Mostly used for set up in other fixtures in other testing modules.
    """
    if shape is None:
        shape = _shape_4d_default()
    if affine is None:
        affine = _affine_eye()
    return _img_zeros(shape, affine)


def _img_4d_rand_eye():
    """Return a default random filled 4D Nifti1Image (identity affine)."""
    data = _rng().random(_shape_4d_default())
    return Nifti1Image(data, _affine_eye())


def _img_4d_rand_eye_medium():
    """Return a random 4D Nifti1Image (identity affine, many volumes)."""
    data = _rng().random(_shape_4d_medium())
    return Nifti1Image(data, _affine_eye())


def _img_4d_mni(shape=None, affine=None):
    if shape is None:
        shape = _shape_4d_default()
    if affine is None:
        affine = _affine_mni()
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
    return _img_4d_rand_eye()


@pytest.fixture
def img_4d_mni():
    """Return a default random filled 4D Nifti1Image."""
    return _img_4d_mni()


@pytest.fixture
def img_4d_rand_eye_medium():
    """Return a default random filled 4D Nifti1Image of medium length."""
    return _img_4d_rand_eye_medium()


@pytest.fixture
def img_4d_long_mni(rng, shape_4d_long, affine_mni):
    """Return a default random filled long 4D Nifti1Image."""
    return Nifti1Image(rng.uniform(size=shape_4d_long), affine=affine_mni)


# ------------------------ ATLAS, LABELS, MAPS ------------------------#


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


def _n_regions():
    """Return a default number of regions for maps."""
    return 9


@pytest.fixture
def n_regions():
    """Return a default number of regions for maps."""
    return _n_regions()


def _img_maps(n_regions=None):
    """Generate a default map image."""
    if n_regions is None:
        n_regions = _n_regions()
    return generate_maps(
        shape=_shape_3d_default(), n_regions=n_regions, affine=_affine_eye()
    )[0]


@pytest.fixture
def img_maps():
    """Generate fixture for default map image."""
    return _img_maps()


def _img_labels():
    """Generate fixture for default label image.

    DO NOT CHANGE n_regions (some tests expect this value).
    """
    return generate_labeled_regions(
        shape=_shape_3d_default(),
        affine=_affine_eye(),
        n_regions=_n_regions(),
    )


@pytest.fixture
def img_labels():
    """Generate fixture for default label image."""
    return _img_labels()


@pytest.fixture
def length():
    """Return a default length for 4D images."""
    return 10


@pytest.fixture
def img_fmri(shape_3d_default, affine_eye, length):
    """Return a default length for fmri images."""
    return generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )[0]


# ------------------------ SURFACE ------------------------#
@pytest.fixture
def single_mesh(rng):
    """Create random coordinates and faces for a single mesh.

    This does not generate meaningful surfaces.
    """
    coords = rng.random((20, 3))
    faces = rng.integers(coords.shape[0], size=(30, 3))
    return [coords, faces]


@pytest.fixture
def in_memory_mesh(single_mesh):
    """Create a random InMemoryMesh.

    This does not generate meaningful surfaces.
    """
    coords, faces = single_mesh
    return InMemoryMesh(coordinates=coords, faces=faces)


def _make_mesh():
    """Create a sample mesh with two parts: left and right, and total of
    9 vertices and 10 faces.

    The left part is a tetrahedron with four vertices and four faces.
    The right part is a pyramid with five vertices and six faces.
    """
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
    return PolyMesh(
        left=InMemoryMesh(left_coords, left_faces),
        right=InMemoryMesh(right_coords, right_faces),
    )


@pytest.fixture()
def surf_mesh():
    """Return _make_mesh as a function allowing it to be used as a fixture."""
    return _make_mesh()


def _make_surface_img(n_samples=1):
    mesh = _make_mesh()
    data = {}
    for i, (key, val) in enumerate(mesh.parts.items()):
        data_shape = (val.n_vertices, n_samples)
        data_part = (
            np.arange(np.prod(data_shape)).reshape(data_shape[::-1])
        ) * 10**i
        data[key] = data_part.astype(float).T
    return SurfaceImage(mesh, data)


@pytest.fixture
def surf_img_2d():
    """Create a sample surface image using the sample mesh.
    This will add some random data to the vertices of the mesh.
    The shape of the data will be (n_vertices, n_samples).
    n_samples by default is 1.
    """
    return _make_surface_img


@pytest.fixture
def surf_img_1d():
    """Create a sample surface image using the sample mesh.
    This will add some random data to the vertices of the mesh.
    The shape of the data will be (n_vertices,).
    """
    img = _make_surface_img(n_samples=1)
    img.data.parts["left"] = np.squeeze(img.data.parts["left"])
    img.data.parts["right"] = np.squeeze(img.data.parts["right"])
    return img


def _make_surface_mask(n_zeros=4):
    mesh = _make_mesh()
    data = {}
    for key, val in mesh.parts.items():
        data_shape = (val.n_vertices, 1)
        data_part = np.ones(data_shape, dtype=int)
        for i in range(n_zeros // 2):
            data_part[i, ...] = 0
        data_part = data_part.astype(bool)
        data[key] = data_part
    return SurfaceImage(mesh, data)


def _surf_mask_1d():
    """Create a sample surface mask using the sample mesh.
    This will create a mask with n_zeros zeros (default is 4) and the
    rest ones.

    The shape of the data will be (n_vertices,).
    """
    mask = _make_surface_mask()
    mask.data.parts["left"] = np.squeeze(mask.data.parts["left"])
    mask.data.parts["right"] = np.squeeze(mask.data.parts["right"])

    return mask


@pytest.fixture
def surf_mask_1d():
    """Create a sample surface mask using the sample mesh.
    This will create a mask with n_zeros zeros (default is 4) and the
    rest ones.

    The shape of the data will be (n_vertices,).
    """
    return _surf_mask_1d()


@pytest.fixture
def surf_mask_2d():
    """Create a sample surface mask using the sample mesh.
    This will create a mask with n_zeros zeros (default is 4) and the
    rest ones.

    The shape of the data will be (n_vertices, 1). Could be useful for testing
    input validation where we throw an error if the mask is not 1D.
    """
    return _make_surface_mask


@pytest.fixture
def surf_label_img(surf_mesh):
    """Return a sample surface label image using the sample mesh.
    Has two regions with values 0 and 1 respectively.
    """
    data = {
        "left": np.asarray([0, 0, 1, 1]),
        "right": np.asarray([1, 1, 0, 0, 0]),
    }
    return SurfaceImage(surf_mesh, data)


@pytest.fixture
def surf_three_labels_img(surf_mesh):
    """Return a sample surface label image using the sample mesh.
    Has 3 regions with values 0, 1 and 2.
    """
    data = {
        "left": np.asarray([0, 0, 1, 1]),
        "right": np.asarray([1, 1, 0, 2, 0]),
    }
    return SurfaceImage(surf_mesh, data)


def _surf_maps_img():
    """Return a sample surface map image using the sample mesh.
    Has 6 regions in total: 3 in both, 1 only in left and 2 only in right.
    Later we multiply the data with random "probability" values to make it
    more realistic.
    """
    data = {
        "left": np.asarray(
            [
                [1, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]
        ),
        "right": np.asarray(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
            ]
        ),
    }
    # multiply with random "probability" values
    data = {
        part: data[part] * _rng().random(data[part].shape) for part in data
    }
    return SurfaceImage(_make_mesh(), data)


@pytest.fixture
def surf_maps_img():
    """Return a sample surface map as fixture."""
    return _surf_maps_img()


def _flip_surf_img_parts(poly_obj):
    """Flip hemispheres of a surface image data or mesh."""
    keys = list(poly_obj.parts.keys())
    keys = [keys[-1]] + keys[:-1]
    return dict(zip(keys, poly_obj.parts.values()))


@pytest.fixture
def flip_surf_img_parts():
    """Flip hemispheres of a surface image data or mesh."""
    return _flip_surf_img_parts


def _flip_surf_img(img):
    """Flip hemispheres of a surface image."""
    return SurfaceImage(
        _flip_surf_img_parts(img.mesh), _flip_surf_img_parts(img.data)
    )


@pytest.fixture
def flip_surf_img():
    """Flip hemispheres of a surface image."""
    return _flip_surf_img


def _drop_surf_img_part(img, part_name="right"):
    """Remove one hemisphere from a SurfaceImage."""
    mesh_parts = img.mesh.parts.copy()
    mesh_parts.pop(part_name)
    data_parts = img.data.parts.copy()
    data_parts.pop(part_name)
    return SurfaceImage(mesh_parts, data_parts)


@pytest.fixture
def drop_surf_img_part():
    """Remove one hemisphere from a SurfaceImage."""
    return _drop_surf_img_part


def _make_surface_img_and_design(n_samples=5):
    des = pd.DataFrame(
        _rng().standard_normal((n_samples, 3)), columns=["", "", ""]
    )
    return _make_surface_img(n_samples), des


@pytest.fixture()
def surface_glm_data():
    """Create a surface image and design matrix for testing."""
    return _make_surface_img_and_design


# ------------------------ PLOTTING ------------------------#


@pytest.fixture(scope="function")
def matplotlib_pyplot():
    """Set up and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")
    yield pyplot
    pyplot.close("all")


@pytest.fixture(scope="function")
def plotly():
    """Check if we can import plotly.

    If not, the tests will be skipped.

    Returns
    -------
    plotly : module
        The ``plotly`` module.
    """
    yield pytest.importorskip(
        "plotly", reason="Plotly is not installed; required to run the tests!"
    )


@pytest.fixture
def transparency_image(rng, affine_mni):
    """Return 3D image to use as transparency image.

    Make sure that values are not just between 0 and 1.
    """
    data_positive = np.zeros((7, 7, 3))
    data_rng = rng.random((7, 7, 3)) * 10 - 5
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    return Nifti1Image(data_positive, affine_mni)
