"""Fixtures for decomposition tests."""

from typing import Union

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _affine_eye
from nilearn.maskers import MultiNiftiMasker, NiftiMasker, SurfaceMasker
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

SHAPE_SURF = {"left": (15, 5), "right": (10, 4)}
N_COMPONENTS = 4
N_SUBJECTS = 4
N_SAMPLES = 5
RANDOM_STATE = 42


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
    """Return the proper masker for test with volume of surface."""
    if data_type == "surface":
        return SurfaceMasker(mask_img=decomposition_mask_img).fit()
    return MultiNiftiMasker(mask_img=img_3d_ones_eye).fit()


def _decomposition_data_surface(rng, decomposition_mesh, with_activation):
    surf_imgs = []
    for _ in range(N_SUBJECTS):
        data = {
            "left": rng.standard_normal(
                size=(
                    decomposition_mesh.parts["left"].coordinates.shape[0],
                    N_SAMPLES,
                )
            ),
            "right": rng.standard_normal(
                size=(
                    decomposition_mesh.parts["right"].coordinates.shape[0],
                    N_SAMPLES,
                )
            ),
        }
        if with_activation:
            data["left"][2:4, :] += 10
            data["right"][2:4, :] += 10
        surf_imgs.append(SurfaceImage(mesh=decomposition_mesh, data=data))

    return surf_imgs


@pytest.fixture
def decomposition_data(
    decomposition_mesh,
    rng,
    affine_eye: np.ndarray,
    data_type: str,
    shape_3d_large,
    with_activation: bool = True,
) -> Union[list[SurfaceImage], list[Nifti1Image]]:
    """Create "multi-subject" dataset with fake activation."""
    if data_type == "surface":
        surf_imgs = []
        for _ in range(N_SUBJECTS):
            data = {
                "left": rng.standard_normal(
                    size=(
                        decomposition_mesh.parts["left"].coordinates.shape[0],
                        N_SAMPLES,
                    )
                ),
                "right": rng.standard_normal(
                    size=(
                        decomposition_mesh.parts["right"].coordinates.shape[0],
                        N_SAMPLES,
                    )
                ),
            }
            if with_activation:
                data["left"][2:4, :] += 10
                data["right"][2:4, :] += 10
            surf_imgs.append(SurfaceImage(mesh=decomposition_mesh, data=data))

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


@pytest.fixture
def canica_data(
    rng,
    _make_canica_components: np.ndarray,
    shape_3d_large,
    decomposition_mesh,
    data_type: str,
    n_subjects=N_SUBJECTS,
) -> Union[list[Nifti1Image], list[SurfaceImage]]:
    """Create a "multi-subject" dataset."""
    if data_type == "nifti":
        data_nii = []
        affine = _affine_eye()

        background = -0.01 * rng.normal(size=shape_3d_large) - 2
        background = background[..., np.newaxis]

        for _ in range(n_subjects):
            this_data = np.dot(
                rng.normal(size=(40, N_COMPONENTS)), _make_canica_components
            )
            this_data += 0.01 * rng.normal(size=this_data.shape)

            # Get back into 3D for CanICA
            this_data = np.reshape(this_data, (40, *shape_3d_large))
            this_data = np.rollaxis(this_data, 0, N_COMPONENTS)

            # Put the border of the image to zero, to mimic a brain image
            this_data[:5] = background[:5]
            this_data[-5:] = background[-5:]
            this_data[:, :5] = background[:, :5]
            this_data[:, -5:] = background[:, -5:]

            data_nii.append(Nifti1Image(this_data, affine))

        return data_nii

    else:
        # TODO for now we generate random data
        # rather than data based on actual components.
        return _decomposition_data_surface(
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
        shape = shape_3d_large

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
        assert isinstance(estimator.masker_, NiftiMasker)
        check_shape = (*estimator.mask_img_.shape, estimator.n_components)

    elif data_type == "surface":
        assert isinstance(estimator.mask_img_, SurfaceImage)
        assert isinstance(estimator.components_img_, SurfaceImage)
        assert isinstance(estimator.masker_, SurfaceMasker)
        check_shape = (estimator.mask_img_.shape[0], estimator.n_components)

    assert estimator.components_img_.shape == check_shape
