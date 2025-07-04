"""Test the base_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal

from nilearn import image
from nilearn.maskers.nifti_masker import filter_and_mask


def test_cropping_code_paths(rng):
    """Mask data with an identically sampled mask and with a smaller mask.

    The results must be identical.
    """
    data = np.zeros([20, 30, 40, 5])
    data[10:15, 5:20, 10:30, :] = 1.0 + rng.uniform(size=(5, 15, 20, 5))

    affine = np.eye(4)

    img = Nifti1Image(data, affine=affine)

    mask = (data[..., 0] > 0).astype("uint8")
    mask_img = Nifti1Image(mask, affine=affine)

    # the mask in mask_img has the same shape and affine as the
    # data and should thus avoid resampling

    # we now crop the mask to its non-zero part. Masking with this
    # mask must yield the same result

    cropped_mask_img = image.crop_img(mask_img, copy_header=True)

    parameters = {
        "smoothing_fwhm": None,
        "high_pass": None,
        "low_pass": None,
        "t_r": None,
        "detrend": False,
        "standardize": "zscore",
        "standardize_confounds": True,
        "clean_kwargs": {},
    }

    # Now do the two maskings
    out_data_uncropped = filter_and_mask(img, mask_img, parameters)
    out_data_cropped = filter_and_mask(img, cropped_mask_img, parameters)

    assert_array_almost_equal(out_data_cropped, out_data_uncropped)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, "auto"])
def test_filter_and_mask_dtype(img_3d_rand_eye, dtype, img_3d_ones_eye):
    """Ensure filter_and_mask conserves dtype."""
    data = img_3d_rand_eye.get_fdata()

    dtype_input = np.float32 if dtype == "auto" else dtype
    img = image.new_img_like(img_3d_rand_eye, data.astype(dtype_input))

    parameters = {
        "smoothing_fwhm": None,
        "high_pass": None,
        "low_pass": None,
        "t_r": None,
        "detrend": False,
        "standardize": "zscore",
        "standardize_confounds": True,
        "clean_kwargs": {},
    }

    output = filter_and_mask(img, img_3d_ones_eye, parameters)

    assert output.dtype == dtype_input

    output = filter_and_mask(
        img, img_3d_ones_eye, parameters, dtype=np.float64
    )

    assert output.dtype == np.float64
