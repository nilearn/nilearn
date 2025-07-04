"""Fixtures for decomposition tests."""

import warnings
from typing import Union

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

SHAPE_SURF = {"left": (15, 5), "right": (10, 4)}
RANDOM_STATE = 42
N_SUBJECTS = 3
# TODO
# some fixtures or tests start breaking if some of those values below
# are changed
N_SAMPLES = 5
N_COMPONENTS = 4


@pytest.fixture(autouse=True)
def suppress_specific_decoding_warning():
    """Ignore internal decoding warnings."""
    with warnings.catch_warnings():
        messages = "Objective did not converge.*|"
        warnings.filterwarnings(
            "ignore",
            message=messages,
            category=UserWarning,
        )
        yield


def _decomposition_mesh() -> PolyMesh:
    """Return a mesh to use for decomposition tests."""
    return PolyMesh(
        left=flat_mesh(*SHAPE_SURF["left"]),
        right=flat_mesh(*SHAPE_SURF["right"]),
    )


@pytest.fixture
def decomposition_mesh() -> PolyMesh:
    """Return a mesh to use for decomposition tests."""
    return _decomposition_mesh()


@pytest.fixture
def decomposition_mask_img(
    data_type: str,
    decomposition_mesh: PolyMesh,
    affine_eye: np.ndarray,
    shape_3d_large,
) -> Union[SurfaceImage, Nifti1Image]:
    """Return a mask for decomposition."""
    if data_type == "surface":
        mask_data = {
            "left": np.ones(
                (decomposition_mesh.parts["left"].coordinates.shape[0],)
            ),
            "right": np.ones(
                (decomposition_mesh.parts["right"].coordinates.shape[0],)
            ),
        }
        return SurfaceImage(mesh=decomposition_mesh, data=mask_data)

    # TODO
    # setting the shape of the mask to be a bit different
    # shape_3d_large that is used for the data
    # to force resampling
    # shape = (
    #     shape_3d_large[0] - 1,
    #     shape_3d_large[1] - 1,
    #     shape_3d_large[2] - 1,
    # )
    shape = shape_3d_large
    mask = np.ones(shape, dtype=np.int8)
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    mask[..., -2:] = 0
    mask[..., :2] = 0
    return Nifti1Image(mask, affine_eye)


@pytest.fixture
def decomposition_masker(
    decomposition_mask_img: Union[SurfaceImage, Nifti1Image],
    img_3d_ones_eye: Nifti1Image,
    data_type: str,
) -> Union[SurfaceMasker, MultiNiftiMasker]:
    """Return the proper masker for test with volume of surface.

    Use detrend=True to check how masker parameters are passed to estimators.
    """
    if data_type == "surface":
        return SurfaceMasker(
            mask_img=decomposition_mask_img, standardize=True
        ).fit()
    return MultiNiftiMasker(mask_img=img_3d_ones_eye, standardize=True).fit()


def _decomposition_images_surface(rng, decomposition_mesh, with_activation):
    return [
        _decomposition_img(
            "surface",
            rng=rng,
            mesh=decomposition_mesh,
            with_activation=with_activation,
        )
        for _ in range(N_SUBJECTS)
    ]


def _decomposition_img(
    data_type,
    rng,
    mesh=None,
    shape=None,
    affine=None,
    with_activation: bool = True,
) -> Union[SurfaceImage, Nifti1Image]:
    """Return a single image for decomposition."""
    if data_type == "surface":
        data = {
            "left": rng.standard_normal(
                size=(
                    mesh.parts["left"].coordinates.shape[0],
                    N_SAMPLES,
                )
            ),
            "right": rng.standard_normal(
                size=(
                    mesh.parts["right"].coordinates.shape[0],
                    N_SAMPLES,
                )
            ),
        }
        if with_activation:
            data["left"][2:4, :] += 10
            data["right"][2:4, :] += 10

        return SurfaceImage(mesh=mesh, data=data)

    shape = (*shape, N_SAMPLES)
    this_img = rng.normal(size=shape)
    if with_activation:
        this_img[2:4, 2:4, 2:4, :] += 10

    return Nifti1Image(this_img, affine)


@pytest.fixture
def decomposition_images(
    data_type,
    rng,
    decomposition_mesh,
    shape_3d_large,
    affine_eye,
    with_activation=True,
):
    """Create "multi-subject" dataset with fake activation."""
    return [
        _decomposition_img(
            data_type,
            rng,
            decomposition_mesh,
            shape_3d_large,
            affine_eye,
            with_activation,
        )
        for _ in range(N_SUBJECTS)
    ]


@pytest.fixture
def decomposition_img(
    data_type,
    rng,
    decomposition_mesh,
    shape_3d_large,
    affine_eye,
    with_activation: bool = True,
) -> Union[SurfaceImage, Nifti1Image]:
    """Return a single image for decomposition."""
    return _decomposition_img(
        data_type,
        rng,
        decomposition_mesh,
        shape_3d_large,
        affine_eye,
        with_activation,
    )


@pytest.fixture
def canica_data(
    rng,
    _make_canica_components: np.ndarray,
    shape_3d_large,
    affine_eye,
    decomposition_mesh,
    data_type: str,
    n_subjects=N_SUBJECTS,
) -> Union[list[Nifti1Image], list[SurfaceImage]]:
    """Create a "multi-subject" dataset."""
    if data_type == "nifti":
        return _make_volume_data_from_components(
            _make_canica_components,
            affine_eye,
            shape_3d_large,
            rng,
            n_subjects,
        )

    else:
        # TODO for now we generate random data
        # rather than data based on actual components.
        return _decomposition_images_surface(
            rng, decomposition_mesh, with_activation=True
        )


@pytest.fixture
def _make_canica_components(
    decomposition_mesh, shape_3d_large, data_type
) -> np.ndarray:
    """Create 4 components.

    3D images unraveled for volume, 2D for surface
    """
    if data_type == "nifti":
        return _canica_components_volume(shape_3d_large)

    else:
        shape = (decomposition_mesh.n_vertices, 1)

        component1 = np.zeros(shape)
        component1[:5] = 1
        component1[5:10] = -1

        component2 = np.zeros(shape)
        component2[:5] = 1
        component2[5:10] = -1

        component3 = np.zeros(shape)
        component3[-5:] = 1
        component3[-10:-5] = -1

        component4 = np.zeros(shape)
        component4[-5:] = 1
        component4[-10:-5] = -1

        return np.vstack(
            (
                component1.ravel(),
                component2.ravel(),
                component3.ravel(),
                component4.ravel(),
            )
        )


def _canica_components_volume(shape):
    """Create 4 volume components."""
    component1 = np.zeros(shape)
    component1[:5, :10] = 1
    component1[5:10, :10] = -1

    component2 = np.zeros(shape)
    component2[:5, -10:] = 1
    component2[5:10, -10:] = -1

    component3 = np.zeros(shape)
    component3[-5:, -10:] = 1
    component3[-10:-5, -10:] = -1

    component4 = np.zeros(shape)
    component4[-5:, :10] = 1
    component4[-10:-5, :10] = -1

    return np.vstack(
        (
            component1.ravel(),
            component2.ravel(),
            component3.ravel(),
            component4.ravel(),
        )
    )


def _make_volume_data_from_components(
    components,
    affine,
    shape,
    rng,
    n_subjects,
):
    """Create a "multi-subject" dataset of volume data."""
    background = -0.01 * rng.normal(size=shape) - 2
    background = background[..., np.newaxis]

    data = []

    # TODO
    # changing this value leads makes tests overall faster but makes
    # test_canica_square_img to fail
    magic_number = 40

    for _ in range(n_subjects):
        this_data = np.dot(
            rng.normal(size=(magic_number, N_COMPONENTS)), components
        )
        this_data += 0.01 * rng.normal(size=this_data.shape)

        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (magic_number, *shape))
        this_data = np.rollaxis(this_data, 0, N_COMPONENTS)

        # Put the border of the image to zero, to mimic a brain image
        this_data[:5] = background[:5]
        this_data[-5:] = background[-5:]
        this_data[:, :5] = background[:, :5]
        this_data[:, -5:] = background[:, -5:]

        data.append(Nifti1Image(this_data, affine))

    return data


@pytest.fixture
def canica_components(rng, _make_canica_components) -> np.ndarray:
    """Create noisy non-positive components data."""
    components = _make_canica_components
    components[rng.standard_normal(components.shape) > 0.8] *= -2.0

    for mp in components:
        assert mp.max() <= -mp.min()  # Goal met ?

    return components


@pytest.fixture
def canica_data_single_img(canica_data) -> Nifti1Image:
    """Create a canonical ICA data for testing purposes."""
    return canica_data[0]


def check_decomposition_estimator(estimator, data_type):
    """Run several standard checks on decomposition estimators."""
    assert estimator.mask_img_ == estimator.masker_.mask_img_
    assert estimator.components_.shape[0] == estimator.n_components

    if data_type == "nifti":
        assert isinstance(estimator.mask_img_, Nifti1Image)
        assert isinstance(estimator.components_img_, Nifti1Image)
        assert isinstance(estimator.masker_, MultiNiftiMasker)
        check_shape = (*estimator.mask_img_.shape, estimator.n_components)

    elif data_type == "surface":
        assert isinstance(estimator.mask_img_, SurfaceImage)
        assert isinstance(estimator.components_img_, SurfaceImage)
        assert isinstance(estimator.masker_, SurfaceMasker)
        check_shape = (estimator.mask_img_.shape[0], estimator.n_components)

    assert estimator.components_img_.shape == check_shape
