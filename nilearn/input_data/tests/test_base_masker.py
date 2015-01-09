"""
Test the base_masker module
"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import nibabel

from ..base_masker import filter_and_mask
from ... import image
from ..._utils.testing import assert_raises_regexp

def test_cropping_code_paths():
    # Will mask data with an identically sampled mask and
    # with a smaller mask. The results must be identical
    rng = np.random.RandomState(42)
    data = np.zeros([20, 30, 40, 5])
    data[10:15, 5:20, 10:30, :] = 1. + rng.rand(5, 15, 20, 5)

    affine = np.eye(4)

    img = nibabel.Nifti1Image(data, affine=affine)

    mask = (data[..., 0] > 0).astype(int)
    mask_img = nibabel.Nifti1Image(mask, affine=affine)

    # the mask in mask_img has the same shape and affine as the
    # data and should thus avoid resampling

    # we now crop the mask to its non-zero part. Masking with this
    # mask must yield the same result

    cropped_mask_img = image.crop_img(mask_img)

    parameters = {"smoothing_fwhm": None,
                  "high_pass": None,
                  "low_pass": None,
                  "t_r": None,
                  "detrend": None,
                  "standardize": None
                                 }

    # Now do the two maskings
    out_data_uncropped, affine_uncropped = filter_and_mask(img,
                                mask_img, parameters)
    out_data_cropped, affine_cropped = filter_and_mask(img,
                                cropped_mask_img, parameters)

    assert_array_almost_equal(out_data_cropped, out_data_uncropped)


def test_filter_and_mask():
    data = np.zeros([20, 30, 40, 5])
    mask = np.zeros([20, 30, 40, 2])
    mask[10, 15, 20, :] = 1

    assert_raises_regexp(TypeError, "A 3D image is expected", filter_and_mask,
                         data, mask, {})
