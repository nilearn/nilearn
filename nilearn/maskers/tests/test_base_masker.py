"""Test the base_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal
from sklearn import __version__ as sklearn_version

from nilearn import image
from nilearn._utils import compare_version
from nilearn._utils.class_inspect import check_estimator
from nilearn.maskers.base_masker import BaseMasker
from nilearn.maskers.nifti_masker import _filter_and_mask

extra_valid_checks = [
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_overwrite_params",
    "check_estimators_unfitted",
    "check_dont_overwrite_parameters",
    "check_get_params_invariance",
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]

if compare_version(sklearn_version, ">", "1.5.2"):
    extra_valid_checks.append("check_positive_only_tag_during_fit")


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[BaseMasker()],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[BaseMasker()],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_cropping_code_paths(rng):
    # Will mask data with an identically sampled mask and
    # with a smaller mask. The results must be identical
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
        "detrend": None,
        "standardize": "zscore",
        "standardize_confounds": True,
        "clean_kwargs": {},
    }

    # Now do the two maskings
    out_data_uncropped = _filter_and_mask(img, mask_img, parameters)
    out_data_cropped = _filter_and_mask(img, cropped_mask_img, parameters)

    assert_array_almost_equal(out_data_cropped, out_data_uncropped)
