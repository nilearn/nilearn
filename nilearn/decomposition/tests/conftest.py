"""Fixtures for decomposition tests."""

from typing import Union

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn.surface.tests.test_surface import flat_mesh

SHAPE_NIFTI = (6, 8, 10)
SHAPE_SURF = {"left": (10, 8), "right": (9, 7)}
N_SUBJECTS = 4
N_SAMPLES = 5


@pytest.fixture
def _mesh() -> PolyMesh:
    return PolyMesh(
        left=flat_mesh(*SHAPE_SURF["left"]),
        right=flat_mesh(*SHAPE_SURF["right"]),
    )


@pytest.fixture
def mask_img(
    data_type: str, _mesh: PolyMesh, affine_eye: np.ndarray
) -> Union[SurfaceImage, Nifti1Image]:
    if data_type == "surface":
        mask_data = {
            "left": np.ones((_mesh.parts["left"].coordinates.shape[0],)),
            "right": np.ones((_mesh.parts["right"].coordinates.shape[0],)),
        }
        return SurfaceImage(mesh=_mesh, data=mask_data)

    shape = (*SHAPE_NIFTI, N_SAMPLES)
    return Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine_eye)


@pytest.fixture
def input_imgs(
    rng, affine_eye: np.ndarray, data_type: str, with_activation: bool = True
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
    shape = (*SHAPE_NIFTI, N_SAMPLES)
    for _ in range(N_SUBJECTS):
        this_img = rng.normal(size=shape)
        if with_activation:
            this_img[2:4, 2:4, 2:4, :] += 10
        nii_imgs.append(Nifti1Image(this_img, affine_eye))

    return nii_imgs


@pytest.fixture
def masker(
    mask_img: SurfaceImage, img_3d_ones_eye: Nifti1Image, data_type: str
) -> Union[SurfaceMasker, MultiNiftiMasker]:
    """Return the proper masker for test with volume of surface."""
    if data_type == "surface":
        return SurfaceMasker(mask_img=mask_img).fit()
    return MultiNiftiMasker(mask_img=img_3d_ones_eye).fit()
