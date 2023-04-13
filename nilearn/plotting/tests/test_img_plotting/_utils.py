"""Configuration and extra fixtures for pytest."""
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import MNI_AFFINE


@pytest.fixture()
def testdata_3d_for_plotting():
    """A random 3D image for testing figures."""
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    img_3d = Nifti1Image(data_positive, MNI_AFFINE)
    # TODO: return img_3D directly and not a dict
    return {"img": img_3d}


@pytest.fixture()
def testdata_4d_for_plotting():
    """Random 4D images for testing figures for multivolume data."""
    rng = np.random.RandomState(42)
    img_4d = Nifti1Image(rng.uniform(size=(7, 7, 3, 10)), MNI_AFFINE)
    img_4d_long = Nifti1Image(rng.uniform(size=(7, 7, 3, 1777)), MNI_AFFINE)
    img_mask = Nifti1Image(np.ones((7, 7, 3), dtype="uint8"), MNI_AFFINE)
    atlas = np.ones((7, 7, 3), dtype="int32")
    atlas[2:5, :, :] = 2
    atlas[5:8, :, :] = 3
    img_atlas = Nifti1Image(atlas, MNI_AFFINE)
    atlas_labels = {
        "gm": 1,
        "wm": 2,
        "csf": 3,
    }
    # TODO: split into several fixtures
    return {
        "img_4d": img_4d,
        "img_4d_long": img_4d_long,
        "img_mask": img_mask,
        "img_atlas": img_atlas,
        "atlas_labels": atlas_labels,
    }
