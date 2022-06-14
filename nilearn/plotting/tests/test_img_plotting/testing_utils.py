"""Utilities for testing image plotting functions."""

import numpy as np
import pytest
from nibabel import Nifti1Image


MNI_AFFINE = np.array([[-2., 0., 0., 90.],
                       [0., 2., 0., -126.],
                       [0., 0., 2., -72.],
                       [0., 0., 0., 1.]])


@pytest.fixture()
def testdata_3d():
    """A random 3D image for testing figures."""
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    img_3d = Nifti1Image(data_positive, MNI_AFFINE)
    return {'img': img_3d}


@pytest.fixture()
def testdata_4d():
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
    return ({
        'img_4d': img_4d,
        'img_4d_long': img_4d_long,
        'img_mask': img_mask,
        'img_atlas': img_atlas,
        'atlas_labels': atlas_labels,
    })
