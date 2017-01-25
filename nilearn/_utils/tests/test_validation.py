
from nose.tools import assert_equal, assert_true, assert_raises
import warnings
import os
import numpy as np
import nibabel
from sklearn.base import BaseEstimator

from nilearn.input_data import NiftiMasker
from nilearn._utils.testing import assert_warns
from nilearn._utils.validation import check_masker, check_feature_screening
from nilearn._utils.param_validation import (MNI152_BRAIN_VOLUME,
                                             _get_mask_volume)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_check_masking():
    # Create toy mask_img
    mask = np.ones((5, 5, 5), np.bool)
    mask_img = nibabel.Nifti1Image(mask.astype(np.int),
                                   np.eye(4))
    # Using two different smoothing_fwhm to compare overriding of this param
    # after masker fitting
    smoothing_fwhm = 4
    smoothing_fwhm_test = 8

    kwargs = {'target_affine': np.eye(4),
              'target_shape': (5, 5, 5),
              'standardize': True,
              'mask_strategy': 'epi',
              'memory': None,
              'memory_level': 1}

    # the masker shold be fitted
    masker_test_1 = NiftiMasker(smoothing_fwhm=smoothing_fwhm_test, **kwargs)
    # assigning a mask_img
    masker_test_2 = NiftiMasker(mask_img=mask_img,
                                smoothing_fwhm=smoothing_fwhm_test, **kwargs)
    # assigning all the params
    masker_test_3 = NiftiMasker(mask_img=mask_img,
                                smoothing_fwhm=smoothing_fwhm_test,
                                **kwargs).fit()
    # Testing various mask inputs
    masks = [None, masker_test_1, masker_test_2, masker_test_3]
    for mask in masks:
        masker = check_masker(mask, smoothing_fwhm=smoothing_fwhm, **kwargs)
        if mask is None:
            assert_true(isinstance(masker, BaseEstimator))
        else:
            assert_true(masker.smoothing_fwhm == smoothing_fwhm_test)


def test_feature_screening():
    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:

            if screening_percentile == 100 or screening_percentile is None:
                assert_equal(check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    None)
            elif screening_percentile == 101 or screening_percentile == -1:
                assert_raises(ValueError, check_feature_screening,
                              screening_percentile, MNI152_BRAIN_VOLUME,
                              is_classif)
            elif screening_percentile == 20:
                assert_true(isinstance(check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    BaseEstimator))
            else:
                assert_warns(UserWarning, check_feature_screening,
                             screening_percentile, MNI152_BRAIN_VOLUME * 2,
                             is_classif)


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (mni152_brain_mask))

