"""Fixtures for decomposition tests."""

from typing import Union

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _affine_eye, _rng
from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

SHAPE_SURF = {"left": (10, 8), "right": (9, 7)}
N_SUBJECTS = 4
N_SAMPLES = 5


@pytest.fixture
def decomposition_mesh() -> PolyMesh:
    """Return a mesh to use for decomposition tests."""
    return PolyMesh(
        left=flat_mesh(*SHAPE_SURF["left"]),
        right=flat_mesh(*SHAPE_SURF["right"]),
    )


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

    # setting the shape of the mask to be a bit different
    # shape_3d_large that is used for the data
    # to force resampling
    shape = (
        shape_3d_large[0] - 1,
        shape_3d_large[1] - 1,
        shape_3d_large[2] - 1,
    )
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
    """Return the proper masker for test with volume of surface."""
    if data_type == "surface":
        return SurfaceMasker(mask_img=decomposition_mask_img).fit()
    return MultiNiftiMasker(mask_img=img_3d_ones_eye).fit()


@pytest.fixture
def decomposition_data(
    rng,
    affine_eye: np.ndarray,
    data_type: str,
    shape_3d_large,
    with_activation: bool = True,
) -> Union[list[SurfaceImage], list[Nifti1Image]]:
    """Create "multi-subject" dataset with fake activation."""
    if data_type == "surface":
        surf_imgs = []
        mesh = {
            "left": flat_mesh(*SHAPE_SURF["left"]),
            "right": flat_mesh(*SHAPE_SURF["right"]),
        }
        for _ in range(N_SUBJECTS):
            data = {
                "left": rng.standard_normal(
                    size=(mesh["left"].coordinates.shape[0], N_SAMPLES)
                ),
                "right": rng.standard_normal(
                    size=(mesh["right"].coordinates.shape[0], N_SAMPLES)
                ),
            }
            if with_activation:
                data["left"][2:4, :] += 10
                data["right"][2:4, :] += 10
            surf_imgs.append(SurfaceImage(mesh=mesh, data=data))

        return surf_imgs

    nii_imgs = []
    shape = (*shape_3d_large, N_SAMPLES)
    for _ in range(N_SUBJECTS):
        this_img = rng.normal(size=shape)
        if with_activation:
            this_img[2:4, 2:4, 2:4, :] += 10
        nii_imgs.append(Nifti1Image(this_img, affine_eye))

    return nii_imgs


@pytest.fixture
def decomposition_data_single_img(
    decomposition_data,
) -> Union[SurfaceImage, Nifti1Image]:
    """Return a single image for decomposition."""
    return decomposition_data[0]


def _make_data_from_components(
    components: np.ndarray,
    shape: tuple[int, int, int],
    n_subjects=N_SUBJECTS,
) -> list[Nifti1Image]:
    affine = _affine_eye()
    rng = _rng()

    background = -0.01 * rng.normal(size=shape) - 2
    background = background[..., np.newaxis]

    data = []
    for _ in range(n_subjects):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += 0.01 * rng.normal(size=this_data.shape)

        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40, *shape))
        this_data = np.rollaxis(this_data, 0, 4)

        # Put the border of the image to zero, to mimic a brain image
        this_data[:5] = background[:5]
        this_data[-5:] = background[-5:]
        this_data[:, :5] = background[:, :5]
        this_data[:, -5:] = background[:, -5:]

        data.append(Nifti1Image(this_data, affine))
    return data


def _make_canica_components(shape: tuple[int, int, int]) -> np.ndarray:
    """Create two images with 'activated regions'."""
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


@pytest.fixture
def canica_components(rng, shape_3d_large) -> np.ndarray:
    """Create noisy non positive data."""
    components = _make_canica_components(shape_3d_large)
    components[rng.standard_normal(components.shape) > 0.8] *= -2.0
    for mp in components:
        assert mp.max() <= -mp.min()  # Goal met ?
    return components


@pytest.fixture
def canica_data(shape_3d_large, n_subjects=N_SUBJECTS) -> list[Nifti1Image]:
    """Create a "multi-subject" dataset."""
    components = _make_canica_components(shape_3d_large)
    data = _make_data_from_components(
        components, shape_3d_large, n_subjects=n_subjects
    )
    return data


@pytest.fixture
def canica_data_single_img(canica_data) -> Nifti1Image:
    """Create a canonical ICA data for testing purposes."""
    return canica_data[0]
