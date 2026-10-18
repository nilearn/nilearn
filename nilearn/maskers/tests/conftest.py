"""Fixtures specific for maskers."""

from collections.abc import Callable

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _make_mesh
from nilearn.maskers import NiftiMasker
from nilearn.surface import SurfaceImage


@pytest.fixture
def make_fitted_nifti_masker() -> Callable[[Nifti1Image], NiftiMasker]:
    """Return a factory for new maskers fitted with an explicit mask."""

    def _make_masker(mask_img: Nifti1Image) -> NiftiMasker:
        """Fit a fresh masker with the provided mask."""
        return NiftiMasker(mask_img=mask_img).fit()

    return _make_masker


@pytest.fixture
def data_1(shape_3d_default) -> np.ndarray:
    """Return 3D zeros with a few 10 in the center."""
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def mask_img_1(data_1, affine_eye) -> Nifti1Image:
    """Return a mask image."""
    return Nifti1Image(data_1.astype("uint8"), affine_eye)


@pytest.fixture
def shape_mask() -> tuple[int, int, int]:
    """Shape for masks."""
    return (13, 14, 15)


def sklearn_surf_label_img(n_regions: int = 2) -> SurfaceImage:
    """Create a sample surface label image using the sample mesh,
    just to use for scikit-learn and nilearn checks.
    """
    if n_regions not in [1, 2]:
        raise ValueError(
            f"'n_regions' must be one of [1, 2]. Got {n_regions=}"
        )

    labels = {
        "left": np.asarray([1, 1, 1, 1]),
        "right": np.asarray([2, 2, 2, 1, 1]),
    }
    labels["left"][labels["left"] > n_regions] = 0
    labels["right"][labels["right"] > n_regions] = 0

    return SurfaceImage(_make_mesh(), labels)
