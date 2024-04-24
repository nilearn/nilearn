"""Pytest fixtures for testing copied headers in nilearn.image functions."""

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from numpy.testing import assert_array_equal

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
    # Change the TR
    header = img.header.copy()
    header["pixdim"][4] = 2.0
    return Nifti1Image(img.get_fdata(), img.affine, header=header)


def match_headers_keys(source, target, except_keys):
    """Check if header fields of two Nifti images match, except for some keys.

    Parameters
    ----------
    source : Nifti1Image
        Source image to compare headers with.
    target : Nifti1Image
        Target image to compare headers from.
    except_keys : list of str
        List of keys that should from comparison.
    """
    for key in source.header.keys():
        if key in except_keys:
            assert (target.header[key] != source.header[key]).any()
        else:
            if isinstance(target.header[key], np.ndarray):
                assert_array_equal(
                    target.header[key],
                    source.header[key],
                )
            else:
                assert target.header[key] == source.header[key]
