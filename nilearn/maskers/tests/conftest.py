"""Fixtures specific for maskers."""

import numpy as np
import pytest
from nibabel import Nifti1Image


def check_valid_for_all_maskers():
    """Return list of names of sklearn checks valid for all maskers."""
    return []


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


@pytest.fixture
def shape_maps():
    """Shape for maps."""
    return (16, 17, 18)
