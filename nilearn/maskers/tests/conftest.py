import numpy as np
import pytest
from nibabel import Nifti1Image


@pytest.fixture
def data_1(shape_3d_default):
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def mask_img_1(data_1, affine_eye):
    return Nifti1Image(data_1.astype("uint8"), affine_eye)
