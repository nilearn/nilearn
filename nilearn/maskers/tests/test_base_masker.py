"""Test the base_masker module."""

import contextlib
import io

import numpy as np
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal

from nilearn import image
from nilearn.maskers.base_masker import mask_logger
from nilearn.maskers.nifti_masker import NiftiMasker, filter_and_mask


def test_mask_logger(img_3d_mni, img_3d_mni_as_file, surf_img_1d):
    """Check verbosity of mask_logger."""
    # verbose = 0 --> no output
    buffer = io.StringIO()
    with (
        contextlib.redirect_stdout(buffer),
    ):
        mask_logger("load_data", img=img_3d_mni, verbose=0)
        mask_logger("load_data", img=surf_img_1d, verbose=0)
    output_verbose = buffer.getvalue()

    assert output_verbose == ""

    # SurfaceImage: no shorten repr
    output = {}
    for verbose in [1, 2, 3]:
        buffer = io.StringIO()
        with (
            contextlib.redirect_stdout(buffer),
        ):
            mask_logger("load_data", img=surf_img_1d, verbose=verbose)
        output[verbose] = buffer.getvalue()

    assert len(output[1]) == len(output[2]) == len(output[3])

    # nifti file or nifti object or list nifti file or  SurfaceImage list
    output = {}
    for img in [img_3d_mni_as_file, img_3d_mni, [img_3d_mni_as_file] * 5]:
        for verbose in [1, 2, 3]:
            buffer = io.StringIO()
            with (
                contextlib.redirect_stdout(buffer),
            ):
                mask_logger("load_data", img=img, verbose=verbose)
            output[verbose] = buffer.getvalue()

        # verbose 2 gives fullpath or affine matrix or expands list
        assert len(output[1]) < len(output[2])
        assert len(output[2]) == len(output[3])

    # list nifti object
    output = {}
    for verbose in [1, 2, 3]:
        buffer = io.StringIO()
        with (
            contextlib.redirect_stdout(buffer),
        ):
            mask_logger("load_data", img=[img_3d_mni] * 5, verbose=verbose)
        output[verbose] = buffer.getvalue()

    # verbose 2: expands list
    assert len(output[1]) < len(output[2])
    # verbose 2: expands list
    assert len(output[2]) == len(output[3])


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

    cropped_mask_img = image.crop_img(mask_img)

    parameters = {
        "smoothing_fwhm": None,
        "high_pass": None,
        "low_pass": None,
        "t_r": None,
        "detrend": False,
        "standardize": "zscore_sample",
        "standardize_confounds": True,
        "clean_kwargs": {},
    }

    # Now do the two maskings
    out_data_uncropped = filter_and_mask(img, mask_img, parameters)
    out_data_cropped = filter_and_mask(img, cropped_mask_img, parameters)

    assert_array_almost_equal(out_data_cropped, out_data_uncropped)


def test_get_masker_params():
    """Test for private method to return params of an instance as dict."""
    masker = NiftiMasker()
    assert masker._get_masker_params() == {
        "clean_args": None,
        "cmap": "gray",
        "detrend": False,
        "dtype": None,
        "high_pass": None,
        "high_variance_confounds": False,
        "low_pass": None,
        "mask_args": None,
        "mask_img": None,
        "mask_strategy": "background",
        "reports": True,
        "runs": None,
        "smoothing_fwhm": None,
        "standardize": False,
        "standardize_confounds": True,
        "t_r": None,
        "target_affine": None,
        "target_shape": None,
    }

    assert masker._get_masker_params(ignore=["t_r"]) == {
        "clean_args": None,
        "cmap": "gray",
        "detrend": False,
        "dtype": None,
        "high_pass": None,
        "high_variance_confounds": False,
        "low_pass": None,
        "mask_args": None,
        "mask_img": None,
        "mask_strategy": "background",
        "reports": True,
        "runs": None,
        "smoothing_fwhm": None,
        "standardize": False,
        "standardize_confounds": True,
        "target_affine": None,
        "target_shape": None,
    }
