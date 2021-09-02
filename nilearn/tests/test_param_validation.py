"""
Test the _utils.param_validation module
"""

import numpy as np
import warnings
import os

import nibabel
import pytest

from sklearn.base import BaseEstimator

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (MNI152_BRAIN_VOLUME,
                                             _get_mask_volume,
                                             check_feature_screening,
                                             check_threshold)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_check_threshold():
    matrix = np.array([[1., 2.],
                       [2., 1.]])

    name = 'threshold'
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ['0.1', '10', '10.2.3%', 'asdf%']
    for wrong_threshold in wrong_thresholds:
        with pytest.raises(ValueError,
                           match='{0}.+should be a number followed by '
                                 'the percent sign'.format(name)):
            check_threshold(wrong_threshold, matrix,
                            fast_abs_percentile, name)

    threshold = object()
    with pytest.raises(TypeError,
                       match='{0}.+should be either a number '
                             'or a string'.format(name)):
        check_threshold(threshold, matrix,
                        fast_abs_percentile, name)

    # Test threshold as int, threshold=2 should return as it is
    # since it is not string
    assert check_threshold(2, matrix, fast_abs_percentile) == 2

    # check whether raises a warning if given threshold is higher than expected
    with pytest.warns(UserWarning):
        check_threshold(3., matrix, fast_abs_percentile)

    # test with numpy scalar as argument
    threshold = 2.
    threshold_numpy_scalar = np.float64(threshold)
    assert (
        check_threshold(threshold, matrix, fast_abs_percentile)
        == check_threshold(threshold_numpy_scalar, matrix,
                           fast_abs_percentile))

    # Test for threshold provided as a percentile of the data (str ending with a
    # %)
    assert 1. < check_threshold("50%", matrix, fast_abs_percentile,
                                name=name) <= 2.


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert MNI152_BRAIN_VOLUME == _get_mask_volume(nibabel.load(
            mni152_brain_mask))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (mni152_brain_mask))


def test_feature_screening():
    # dummy
    mask_img_data = np.zeros((182, 218, 182))
    mask_img_data[30:-30, 30:-30, 30:-30] = 1
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(mask_img_data, affine=affine)

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:

            if screening_percentile == 100 or screening_percentile is None:
                assert check_feature_screening(
                    screening_percentile, mask_img, is_classif) == None
            elif screening_percentile == 101 or screening_percentile == -1:
                pytest.raises(ValueError, check_feature_screening,
                              screening_percentile, mask_img, is_classif)
            elif screening_percentile == 20:
                assert isinstance(check_feature_screening(
                    screening_percentile, mask_img, is_classif),
                    BaseEstimator)
