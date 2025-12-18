"""Fixtures specific for maskers."""

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _make_mesh
from nilearn.surface import SurfaceImage


@pytest.fixture
def data_1(shape_3d_default):
    """Return 3D zeros with a few 10 in the center."""
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def mask_img_1(data_1, affine_eye):
    """Return a mask image."""
    return Nifti1Image(data_1.astype("uint8"), affine_eye)


@pytest.fixture
def shape_mask():
    """Shape for masks."""
    return (13, 14, 15)


def sklearn_surf_label_img(n_regions: int = 2) -> SurfaceImage:
    """Create a sample surface label image using the sample mesh,
    just to use for scikit-learn and nilearn checks.
    """
    if n_regions not in [1, 2]:
        raise ValueError(f"'n_regions' must be '1' or '2'. Got {n_regions=}")

    labels = {
        "left": np.asarray([1, 1, 2, 2]),
        "right": np.asarray([1, 1, 2, 2, 2]),
    }
    labels["left"][labels["left"] > n_regions] = 0
    labels["right"][labels["right"] > n_regions] = 0

    return SurfaceImage(_make_mesh(), labels)
