"""Pytest fixtures for testing copied headers in nilearn.image functions."""

import pytest
from nibabel.nifti1 import Nifti1Image

from nilearn import image


@pytest.fixture
def img_4d_ones_eye_default_header(img_4d_ones_eye):
    """Return a 4D Nifti1Image with default header.

    The header is created by new_img_like and is not modified. The image is
    filled with ones and has an identity affine.
    """
    img = image.new_img_like(
        img_4d_ones_eye,
        data=img_4d_ones_eye.get_fdata(),
        copy_header=False,
    )
    return img


@pytest.fixture
def img_4d_ones_eye_tr2(img_4d_ones_eye):
    """Return a 4D Nifti1Image with otherwise default header, except TR 2.0.

    The header is the default one created by new_img_like, but the TR is
    changed to 2.0. The image is filled with ones and has an identity affine.
    """
    img = image.new_img_like(
        img_4d_ones_eye,
        data=img_4d_ones_eye.get_fdata(),
        copy_header=True,
    )
    # Change the TR
    header = img.header.copy()
    header["pixdim"][4] = 2.0
    return Nifti1Image(img.get_fdata(), img.affine, header=header)


@pytest.fixture
def img_4d_mni_tr2(img_4d_mni):
    """Return a 4D Nifti1Image with MNI affine and header, and TR 2.0.

    The header has the MNI affine, and the TR is changed to 2.0. The image is
    filled with random numbers.
    """
    img = image.new_img_like(
        img_4d_mni, data=img_4d_mni.get_fdata(), copy_header=True
    )
    # Add fake description
    header = img.header.copy()
    header["descrip"] = b"Fake description"
    # Change the TR
    header["pixdim"][4] = 2.0
    return Nifti1Image(img.get_fdata(), img.affine, header=header)
